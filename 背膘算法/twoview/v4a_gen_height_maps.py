# -*- coding: utf-8 -*-
"""
v4a_gen_height_maps.py — 标准化高度图（俯视）：地面拟合 → 离地高度 → 固定物理分辨率，保留绝对体尺

来源 / 背景
  05 §3 ②「几何标准化：拟合地面 → 离地高度 → 等物理尺度 → 统一朝向」（借鉴 PigFormer），一直未做。
  现有俯视深度图（generate_depthmap_dataset.py）每帧按掩膜外框裁剪后单独缩放到 256×256，**绝对体尺丢失**，
  像素值是「相机到猪背距离」而不是离地高度（相机高度不同的批次之间不可比）。本人 2026-10-12 决定做此改进。

方法（每头猪均匀 20 帧，与俯视深度图相同的帧与分割）
  1. 分割：直接 import generate_depthmap_dataset 的 get_frame_pairs / get_pig_mask（YOLO11n + FastSAM-s）
  2. 反投影：深度（mm，已配准到彩色 1280×720）+ 彩色相机内参（index_v2 master_fx/fy/cx/cy 或 intrinsic.json）
     → 相机坐标 (X, Y, Z) mm
  3. 地面平面（每头猪拟合一次：录制期间相机固定，汇总 20 帧的候选点；2026-10-12 演示数据测试中逐帧拟合会
     偶尔误选栏位侧面，倾角约 26°）：候选点 = 有效深度、不在（膨胀后的）猪掩膜内、且比猪背中位深度远 FLOOR_MARGIN mm 以上；
     RANSAC（阈值 20 mm，法向与相机光轴夹角 < 20°）→ 内点最小二乘精修；法向指向相机。
     离地高度 h = n·p + d（相机原点 h = d = 相机离地高度）
  4. 栅格化：在地面平面上取正交基，猪体点投影后用 PCA 把体长方向摆成水平；窗口 2000 × 1000 mm，
     256 × 128 像素（7.8125 mm/px），像素值 = 该格最大离地高度（mm），背景 0；猪体占据区域内的小空洞用
     3×3 灰度膨胀补齐。头尾方向不定（训练时用水平 + 垂直翻转增广）
  5. 每帧体尺（物理单位）：体长 / 体宽（主 / 次轴 2–98% 分位跨度）、背高（h 的 98% 分位）、俯视面积、
     体积（Σ h × 格面积）、相机离地高度、地面内点数、地面倾角

质控字段（写入 metadata，不在生成时过滤）：camera_h_mm、floor_inliers、tilt_deg、back_h_mm、mask_frac

输入：twoview/index_v2.csv（master_dir 及内参）；或 --demo 目录（edge_deploy 演示子集布局：
      <pig>/{color,depth,intrinsic.json}），仅用于在没有原始数据盘时测试逻辑
输出（新目录，不覆盖已有数据）：yolo+FastSAM/hmap_dataset/{uid}/frame_XXX.npy（float16，mm）、metadata.csv、
      _preview/{uid}.png、gen_log.txt；中断后重跑跳过已完成的猪（最后一头重做）

用法：
  D:\\Anaconda\\envs\\pytorch\\python.exe v4a_gen_height_maps.py --farm all [--drive G:H]
  D:\\Anaconda\\envs\\pytorch\\python.exe v4a_gen_height_maps.py --demo D:/jetson_demo/20260819/1 --out <临时目录>
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
SEG_DIR = os.path.normpath(os.path.join(HERE, '..', 'yolo+FastSAM'))
sys.path.insert(0, SEG_DIR)
import generate_depthmap_dataset as G  # noqa: E402

INDEX = os.path.join(HERE, 'index_v2.csv')
OUT_DEFAULT = os.path.join(SEG_DIR, 'hmap_dataset')

MIN_D, MAX_D = 300.0, 3000.0     # 有效深度 mm
FLOOR_MARGIN = 300.0             # 地面候选点须比猪背中位深度远这么多 mm
RANSAC_THR, RANSAC_ITERS = 20.0, 200
MAX_TILT = 20.0                  # 地面法向与光轴最大夹角（度）；演示数据正常倾角约 7.5°，35° 时会误选栏位侧面
WIN_L, WIN_W = 2000.0, 1000.0    # 窗口（mm）：体长方向 × 体宽方向
GRID_L, GRID_W = 256, 128
CELL = WIN_L / GRID_L            # 7.8125 mm/px
FLOOR_STRIDE = 3                 # 地面候选点抽稀步长（像素）

META_COLS = ['frame_path', 'uid', 'farm', 'batch', 'frame_idx', 'ok', 'reason', 'mask_frac', 'camera_h_mm',
             'floor_inliers', 'tilt_deg', 'back_h_mm', 'length_mm', 'width_mm', 'area_cm2', 'volume_l']


def backproject(depth, fx, fy, cx, cy, ys, xs):
    z = depth[ys, xs].astype(np.float64)
    return np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], 1)


def fit_floor(P, rng, d_range=None):
    """RANSAC 平面；返回 (n, d, n_inliers) 使 h = n·p + d，相机原点 h = d > 0；失败返回 None。
    d_range=(lo, hi)：只接受相机离地高度 |d| 在该范围内的候选平面（--fix-outliers 用）。"""
    if len(P) < 500:
        return None
    best, best_in = None, None
    cos_max = np.cos(np.deg2rad(MAX_TILT))
    for _ in range(RANSAC_ITERS):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if abs(n[2]) < cos_max:
            continue
        d = -n @ a
        if d_range is not None and not (d_range[0] <= abs(d) <= d_range[1]):
            continue
        inl = np.abs(P @ n + d) < RANSAC_THR
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (n, d), inl
    if best is None or best_in.sum() < 300:
        return None
    Q = P[best_in]                         # 最小二乘精修（SVD）
    c0 = Q.mean(0)
    n = np.linalg.svd(Q - c0, full_matrices=False)[2][-1]
    d = -n @ c0
    if d < 0:                              # 法向指向相机
        n, d = -n, -d
    inl = np.abs(P @ n + d) < RANSAC_THR
    return n, d, int(inl.sum())


def rasterize(Pp, n, d):
    """猪体点 → (高度图 GRID_W×GRID_L, 体尺 dict)。"""
    h = Pp @ n + d
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.stack([Pp @ e1, Pp @ e2], 1)
    keep = h > 50                          # 去掉贴地噪点
    uv, h = uv[keep], h[keep]
    if len(h) < 500:
        return None, None
    c = uv.mean(0)
    _, V = np.linalg.eigh(np.cov((uv - c).T))
    ab = (uv - c) @ V[:, ::-1]             # 第 0 列 = 主轴（体长）
    lo, hi = np.percentile(ab, [2, 98], axis=0)
    ab -= (lo + hi) / 2                    # 以稳健外接框中心对齐
    col = np.floor(ab[:, 0] / CELL + GRID_L / 2).astype(int)
    row = np.floor(ab[:, 1] / CELL + GRID_W / 2).astype(int)
    m = (col >= 0) & (col < GRID_L) & (row >= 0) & (row < GRID_W)
    hm = np.zeros((GRID_W, GRID_L), np.float32)
    np.maximum.at(hm, (row[m], col[m]), h[m].astype(np.float32))
    occ = (hm > 0).astype(np.uint8)
    foot = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fill = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    hm = np.where((hm == 0) & (foot > 0), fill, hm)
    feats = dict(back_h_mm=float(np.percentile(h, 98)), length_mm=float(hi[0] - lo[0]),
                 width_mm=float(hi[1] - lo[1]), area_cm2=float((hm > 0).sum() * CELL * CELL / 100),
                 volume_l=float(hm.sum() * CELL * CELL / 1e6))
    return hm, feats


def segment_frame(color, depth, K, yolo, fsam, rng, n_floor=2000):
    """分割一帧，返回 (pig_mask, 该帧地面候选点（抽样）, info)；失败时 pig_mask 为 None。"""
    fx, fy, cx, cy = K
    mask = G.get_pig_mask(color, yolo, fsam)
    if mask is None:
        return None, None, dict(ok=0, reason='no_mask')
    if mask.shape != depth.shape:
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = (depth > MIN_D) & (depth < MAX_D)
    pig = (mask > 0) & valid
    info = dict(mask_frac=round(float(mask.mean()), 4))
    if pig.sum() < 2000:
        return None, None, dict(info, ok=0, reason='small_mask')
    zpig = float(np.median(depth[pig]))
    far = valid & (cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0) & (depth > zpig + FLOOR_MARGIN)
    ys, xs = np.nonzero(far[::FLOOR_STRIDE, ::FLOOR_STRIDE])
    ys, xs = ys * FLOOR_STRIDE, xs * FLOOR_STRIDE
    if len(ys) > n_floor:
        k = rng.choice(len(ys), n_floor, replace=False)
        ys, xs = ys[k], xs[k]
    return pig, backproject(depth, fx, fy, cx, cy, ys, xs), info


def pig_list(a):
    if a.demo:
        pigs = []
        for p in sorted(os.listdir(a.demo)):
            base = os.path.join(a.demo, p)
            if not os.path.isfile(os.path.join(base, 'intrinsic.json')):
                continue
            k = json.load(open(os.path.join(base, 'intrinsic.json')))['intrinsic_matrix']
            pigs.append(dict(uid=f'demo_{p}', farm='DEMO', batch='demo', base=base, K=(k[0], k[4], k[6], k[7])))
        return pigs
    with open(INDEX, encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['master_dir'] and (a.farm == 'all' or r['farm'] == a.farm)]
    rows.sort(key=lambda r: r['uid'])
    return [dict(uid=r['uid'], farm=r['farm'], batch=r['batch'], base=r['master_dir'],
                 K=tuple(float(r[f'master_{c}']) for c in ('fx', 'fy', 'cx', 'cy'))) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--farm', default='all', choices=['FMH', 'DK', 'all'])
    ap.add_argument('--demo', default='', help='演示子集目录（测试用）')
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fix-outliers', action='store_true',
                    help='重做相机离地高度偏离本批中位数 > 150 mm 或倾角 > 12° 的猪：地面平面限定在本批中位高度 ±150 mm')
    ap.add_argument('--drive', default='', help='原始数据盘符映射，如 G:H（index_v2 记录为 G 盘，副本挂在 H 盘时）')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, '_preview'), exist_ok=True)
    logf = open(os.path.join(a.out, 'gen_log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    pigs = pig_list(a)
    if a.drive:
        src, dst = a.drive.split(':')
        for p in pigs:
            if p['base'][:2].upper() == f'{src.upper()}:':
                p['base'] = f'{dst.upper()}:' + p['base'][2:]
    if a.limit:
        pigs = pigs[:a.limit]
    meta_path = os.path.join(a.out, 'metadata.csv')
    d_ranges = {}
    if a.fix_outliers:
        with open(meta_path, encoding='utf-8') as f:
            mrows = list(csv.DictReader(f))
        first = {}
        for r in mrows:
            if r['ok'] == '1':
                first.setdefault(r['uid'], r)
        med = {}
        for r in first.values():
            med.setdefault(r['batch'], []).append(float(r['camera_h_mm']))
        med = {b: float(np.median(v)) for b, v in med.items()}
        bad = {u for u, r in first.items()
               if abs(float(r['camera_h_mm']) - med[r['batch']]) > 150 or float(r['tilt_deg']) > 12}
        d_ranges = {u: (med[first[u]['batch']] - 150, med[first[u]['batch']] + 150) for u in bad}
        with open(meta_path, 'w', newline='', encoding='utf-8') as f:
            w0 = csv.DictWriter(f, fieldnames=META_COLS)
            w0.writeheader()
            w0.writerows([r for r in mrows if r['uid'] not in bad])
        pigs = [p for p in pigs if p['uid'] in bad]
        print(f'fix-outliers: redo {sorted(bad)} with camera-height windows '
              f'{ {u: tuple(round(x) for x in v) for u, v in d_ranges.items()} }')
    done = set()
    if os.path.isfile(meta_path) and not a.fix_outliers:
        with open(meta_path, encoding='utf-8') as f:
            mrows = list(csv.DictReader(f))
        if mrows:  # 最后一头可能不完整 → 删去重做
            last = mrows[-1]['uid']
            keep = [r for r in mrows if r['uid'] != last]
            with open(meta_path, 'w', newline='', encoding='utf-8') as f:
                w0 = csv.DictWriter(f, fieldnames=META_COLS)
                w0.writeheader()
                w0.writerows(keep)
            done = {r['uid'] for r in keep}
    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] height maps: {len(pigs)} pigs, done {len(done)}, out {a.out}; '
        f'window {WIN_L:.0f}x{WIN_W:.0f} mm, grid {GRID_L}x{GRID_W}, {CELL:.4f} mm/px')
    cwd = os.getcwd()
    os.chdir(SEG_DIR)
    yolo, fsam = G.load_models()
    os.chdir(cwd)
    new = not os.path.isfile(meta_path)
    mf = open(meta_path, 'a', newline='', encoding='utf-8')
    w = csv.DictWriter(mf, fieldnames=META_COLS)
    if new:
        w.writeheader()
    rng = np.random.default_rng(0)
    for i, p in enumerate(pigs, 1):
        if p['uid'] in done:
            continue
        pairs = G.get_frame_pairs(os.path.join(p['base'], 'color'), os.path.join(p['base'], 'depth'),
                                  G.FRAMES_PER_PIG)
        os.makedirs(os.path.join(a.out, p['uid']), exist_ok=True)
        # 第 1 遍：逐帧分割，汇总全部帧的地面候选点（录制期间相机固定 → 每头猪只拟合一次地面）
        frames, floor_pts = [], []
        for j, (cp, dp) in enumerate(pairs):
            color, depth = G.imread_chinese(cp), G.imread_depth_chinese(dp)
            row = dict(frame_path=f'{p["uid"]}\\frame_{j:03d}.npy', uid=p['uid'], farm=p['farm'],
                       batch=p['batch'], frame_idx=j)
            if color is None or depth is None:
                frames.append((row, None, None, dict(ok=0, reason='read_fail')))
                continue
            pig, Pf, info = segment_frame(color, depth, p['K'], yolo, fsam, rng)
            frames.append((row, depth, pig, info))
            if Pf is not None:
                floor_pts.append(Pf)
        fl = fit_floor(np.concatenate(floor_pts), rng, d_ranges.get(p['uid'])) if floor_pts else None
        # 第 2 遍：用同一个地面平面栅格化每一帧
        n_ok, prev = 0, None
        for row, depth, pig, info in frames:
            if pig is not None and fl is None:
                info = dict(info, ok=0, reason='no_floor')
            elif pig is not None:
                n, d, n_in = fl
                ys, xs = np.nonzero(pig)
                hm, feats = rasterize(backproject(depth, *p['K'], ys, xs), n, d)
                if hm is None:
                    info = dict(info, ok=0, reason='raster_fail')
                else:
                    info = dict(info, ok=1, reason='', camera_h_mm=round(float(d), 1), floor_inliers=n_in,
                                tilt_deg=round(float(np.degrees(np.arccos(abs(n[2])))), 2),
                                **{k: round(v, 2) for k, v in feats.items()})
                    np.save(os.path.join(a.out, p['uid'], f'frame_{row["frame_idx"]:03d}.npy'), hm.astype(np.float16))
                    n_ok += 1
                    if prev is None or row['frame_idx'] == len(pairs) // 2:
                        prev = hm
            w.writerow(dict(row, **info))
        mf.flush()
        if prev is not None:
            v = np.clip(prev / 1100.0 * 255, 0, 255).astype(np.uint8)
            cv2.imencode('.png', cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS) * (prev[..., None] > 0))[1].tofile(
                os.path.join(a.out, '_preview', f'{p["uid"]}.png'))
        if i % 50 == 0 or i <= 3:
            el = time.time() - t0
            log(f'  [{i}/{len(pigs)}] {p["uid"]} ok {n_ok}/{len(pairs)}  {el / 60:.1f} min, '
                f'ETA {el / i * (len(pigs) - i) / 60:.1f} min')
    mf.close()
    log(f'done | {(time.time() - t0) / 60:.1f} min -> {a.out}')


if __name__ == '__main__':
    main()
