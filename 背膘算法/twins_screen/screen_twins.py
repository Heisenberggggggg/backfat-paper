# -*- coding: utf-8 -*-
"""
screen_twins.py — 双胞胎吉水白水 GP 场手持 RGB-D 数据的筛选辅助工具

来源 / 背景
  双胞胎数据是手持深度相机沿限位栏拍摄的，每头猪在背部、臀部各停留几秒，
  约 1160 段录像、18.5 万帧，未整理。逐帧人工删图不现实，也容易误删。

核心思路（只读原始数据，一张不删）
  1. 找出所有录像段（含 camera.json 的目录）
  2. 逐帧算自动质量指标：清晰度、中央区域深度有效率 / 中位深度 / 近距物体占比、帧间运动
     → 标出明显不能用的帧（auto_bad）
  3. 每段录像生成一张「缩略图拼板」：均匀抽 N 帧（RGB + 深度小窗 + 帧号），
     底部附全段质量时间轴 → 人工 10 秒内判断背部段 / 臀部段 / 废弃段
  4. 输出 twins_index.xlsx 给师弟师妹填：耳号、背部帧段、臀部帧段、废弃帧段、可用、备注
     （xlsx 且列为文本格式，避免 Excel 把「1-12」自动变成日期）

与旧脚本的区别
  不同于 edge_deploy/check_daily.py（面向农科院饲喂站 meta.json 目录规范），
  本脚本面向双胞胎手持数据的原始目录（rgb_%06d.png / depth_%06d.png / camera.json）。

用法
  试跑 10 段:
    python screen_twins.py --root "G:/耿来硕数据/双胞胎实验/深度和双目相机数据" --out out_trial --limit 10
  指定几段:
    python screen_twins.py --root ... --out out_trial --only c1/1,d3/20,1
  全量（建议晚上跑）:
    python screen_twins.py --root ... --out out_v1 --workers 12
  已生成过的拼板会跳过（断点续跑）；加 --force 重算。
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

SKIP_DIRS = {'.Trash-1000', '@eaDir', '@tmp', 'shuangmu', 'kinect_video', 'pointcloud_viewer'}

# ---- 自动预筛阈值（保守：宁可放过交给人工，也不误杀） ----
SHARP_REL_BAD = 0.30      # 清晰度 < 本段 P90 的 30% → 模糊
VALID_C_BAD = 0.40        # 中央区域深度有效像素 < 40%
NEAR_MM = 2000            # 「近距物体」= 深度 < 2 m
NEAR_C_BAD = 0.30         # 中央区域近距像素 < 30% → 画面中央大概率没有猪
DEPTH_MIN_MM, DEPTH_MAX_MM = 300, 2200   # 中央中位深度超出此范围视为异常

# ---- 拼板参数 ----
TILE_W, TILE_H = 320, 180
COLS = 6
DEPTH_VIS_RANGE = (400, 2500)  # 深度伪彩固定量程，各段之间可比


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def find_recordings(root):
    recs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
        if 'camera.json' in filenames:
            recs.append(os.path.relpath(dirpath, root).replace('\\', '/'))
    return sorted(recs, key=natural_key)


def frame_ids(rec_dir):
    names = os.listdir(rec_dir)
    rgb = {n[4:10] for n in names if n.startswith('rgb_') and n.endswith('.png')}
    dep = {n[6:12] for n in names if n.startswith('depth_') and n.endswith('.png')}
    return sorted(rgb & dep), len(rgb), len(dep)


def central_box(h, w, frac=0.5):
    y0, x0 = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return slice(y0, h - y0), slice(x0, w - x0)


def colorize_depth(d):
    lo, hi = DEPTH_VIS_RANGE
    v = np.clip((d.astype(np.float32) - lo) / (hi - lo), 0, 1)
    c = cv2.applyColorMap(np.uint8(255 - v * 255), cv2.COLORMAP_JET)  # 近 = 红，远 = 蓝
    c[d == 0] = 0
    return c


def imread_u(path, flag):
    # cv2.imread 不支持中文路径，用 imdecode
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, flag) if buf.size else None


def frame_metrics(rec_dir, fid, prev_gray):
    rgb = imread_u(os.path.join(rec_dir, f'rgb_{fid}.png'), cv2.IMREAD_COLOR)
    dep = imread_u(os.path.join(rec_dir, f'depth_{fid}.png'), cv2.IMREAD_UNCHANGED)
    if rgb is None or dep is None:
        return None, prev_gray, None, None
    small = cv2.resize(rgb, (480, 270), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    motion = float(np.mean(cv2.absdiff(gray, prev_gray))) if prev_gray is not None else 0.0

    ds = cv2.resize(dep, (480, 270), interpolation=cv2.INTER_NEAREST)
    ys, xs = central_box(*ds.shape)
    c = ds[ys, xs]
    valid = c > 0
    m = dict(frame=int(fid), sharp=round(sharp, 1), valid_c=round(float(valid.mean()), 3),
             depth_med_c=round(float(np.median(c[valid])) if valid.any() else 0.0, 0),
             near_c=round(float(((c > 0) & (c < NEAR_MM)).mean()), 3), motion=round(motion, 2),
             brightness=round(float(gray.mean()), 1))
    return m, gray, small, ds


def judge(rows):
    sharps = np.array([r['sharp'] for r in rows], dtype=float)
    p90 = np.percentile(sharps, 90) if len(sharps) else 1.0
    for r in rows:
        r['sharp_rel'] = round(r['sharp'] / p90, 3) if p90 > 0 else 0.0
        reasons = []
        if r['sharp_rel'] < SHARP_REL_BAD:
            reasons.append('模糊')
        if r['valid_c'] < VALID_C_BAD:
            reasons.append('深度缺失')
        if r['near_c'] < NEAR_C_BAD:
            reasons.append('中央无近物')
        if not (DEPTH_MIN_MM <= r['depth_med_c'] <= DEPTH_MAX_MM):
            reasons.append('距离异常')
        r['auto_bad'] = int(bool(reasons))
        r['reason'] = '|'.join(reasons)
    return rows


def put_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


REASON_EN = {'模糊': 'blur', '深度缺失': 'no-depth', '中央无近物': 'no-pig', '距离异常': 'dist'}


def make_tile(small_rgb, small_dep, row):
    tile = cv2.resize(small_rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    inset = cv2.resize(colorize_depth(small_dep), (TILE_W // 3, TILE_H // 3), interpolation=cv2.INTER_NEAREST)
    tile[-inset.shape[0]:, -inset.shape[1]:] = inset
    put_text(tile, f"#{row['frame']}", (6, 22), 0.7)
    if row['auto_bad']:
        cv2.rectangle(tile, (0, 0), (TILE_W - 1, TILE_H - 1), (0, 0, 255), 4)
        en = ' '.join(REASON_EN[x] for x in row['reason'].split('|'))
        put_text(tile, en, (6, 44), 0.5, (80, 80, 255))
    return tile


def timeline(rows, sampled, width):
    """全段质量时间轴：绿=相对清晰度，蓝=中央深度（400–2500 mm），红竖线=auto_bad，白三角=拼板抽样帧"""
    h = 110
    img = np.full((h, width, 3), 30, np.uint8)
    n = len(rows)
    if n < 2:
        return img
    xs = np.linspace(8, width - 8, n).astype(int)
    for x, r in zip(xs, rows):
        if r['auto_bad']:
            cv2.line(img, (int(x), 18), (int(x), h - 18), (40, 40, 160), 1)
    sr = np.clip([r['sharp_rel'] for r in rows], 0, 1.5) / 1.5
    lo, hi = DEPTH_VIS_RANGE
    dm = np.clip((np.array([r['depth_med_c'] for r in rows], float) - lo) / (hi - lo), 0, 1)
    pts_s = np.stack([xs, (h - 18 - sr * (h - 36)).astype(int)], 1).astype(np.int32)
    pts_d = np.stack([xs, (h - 18 - dm * (h - 36)).astype(int)], 1).astype(np.int32)
    cv2.polylines(img, [pts_s.reshape(-1, 1, 2)], False, (80, 220, 80), 2)
    cv2.polylines(img, [pts_d.reshape(-1, 1, 2)], False, (230, 160, 60), 2)
    for i in sampled:
        cv2.drawMarker(img, (int(xs[i]), h - 8), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 8, 1)
    put_text(img, 'green=sharpness  blue=center depth  red=auto-bad  triangle=tile above', (10, 14), 0.45)
    put_text(img, f'#{rows[0]["frame"]}', (4, h - 2), 0.4)
    put_text(img, f'#{rows[-1]["frame"]}', (width - 60, h - 2), 0.4)
    return img


def ranges_str(frames):
    if not frames:
        return ''
    out, s, p = [], frames[0], frames[0]
    for x in frames[1:]:
        if x == p + 1:
            p = x
            continue
        out.append(f'{s}-{p}' if s != p else f'{s}')
        s = p = x
    out.append(f'{s}-{p}' if s != p else f'{s}')
    return ','.join(out)


def process(args_tuple):
    root, rec, out_dir, n_tiles, force = args_tuple
    rec_dir = os.path.join(root, rec)
    safe = rec.replace('/', '__')
    sheet_path = os.path.join(out_dir, 'sheets', f'{safe}.jpg')
    part_csv = os.path.join(out_dir, 'parts', f'{safe}.csv')
    if not force and os.path.exists(sheet_path) and os.path.exists(part_csv):
        return rec, 'skip'
    try:
        try:
            with open(os.path.join(rec_dir, 'camera.json'), encoding='utf-8') as f:
                cam = json.load(f)
        except Exception:
            cam = {}
        cap_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(rec_dir, 'camera.json')))
        cap_str = '' if cap_time.year < 2000 else cap_time.strftime('%Y-%m-%d %H:%M')
        fids, n_rgb, n_dep = frame_ids(rec_dir)
        n = len(fids)
        sampled = set(np.unique(np.linspace(0, n - 1, min(n_tiles, n)).astype(int)).tolist()) if n else set()

        rows, tiles_src, prev = [], {}, None
        for i, fid in enumerate(fids):
            m, prev, small, ds = frame_metrics(rec_dir, fid, prev)
            if m is None:
                continue
            rows.append(m)
            if i in sampled:
                tiles_src[len(rows) - 1] = (small, ds)
        rows = judge(rows)

        keys = sorted(tiles_src)
        tiles = [make_tile(*tiles_src[k], rows[k]) for k in keys]
        width = COLS * TILE_W
        header = np.full((50, width, 3), 20, np.uint8)
        n_bad = sum(r['auto_bad'] for r in rows)
        put_text(header, f'{rec}   frames={n} (rgb {n_rgb}/depth {n_dep})   auto-bad={n_bad}   {cap_str}', (10, 34), 0.8)
        grid = []
        for k in range(0, len(tiles), COLS):
            rt = tiles[k:k + COLS]
            rt += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (COLS - len(rt))
            grid.append(np.hstack(rt))
        os.makedirs(os.path.dirname(sheet_path), exist_ok=True)
        cv2.imencode('.jpg', np.vstack([header] + grid + [timeline(rows, keys, width)]),
                     [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(sheet_path)

        os.makedirs(os.path.dirname(part_csv), exist_ok=True)
        fields = ['folder', 'frame', 'sharp', 'sharp_rel', 'valid_c', 'depth_med_c', 'near_c',
                  'motion', 'brightness', 'auto_bad', 'reason']
        with open(part_csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({'folder': rec, **{k: r[k] for k in fields if k != 'folder'}})
        meta = dict(folder=rec, n_frames=n, n_rgb=n_rgb, n_depth=n_dep, capture_time=cap_str,
                    cam_frames=cam.get('record', {}).get('frames', ''), n_bad=n_bad,
                    good_ratio=round(1 - n_bad / n, 3) if n else 0.0,
                    good_ranges=ranges_str([r['frame'] for r in rows if not r['auto_bad']]))
        with open(part_csv.replace('.csv', '.meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False)
        return rec, 'ok'
    except Exception as e:  # 单段出错不影响整体，记录后继续
        return rec, f'ERROR {type(e).__name__}: {e}'


def write_outputs(out_dir, recs):
    import openpyxl
    from openpyxl.styles import Font, PatternFill

    q_path = os.path.join(out_dir, 'frames_quality.csv')
    with open(q_path, 'w', newline='', encoding='utf-8-sig') as fo:
        header_done = False
        for rec in recs:
            p = os.path.join(out_dir, 'parts', rec.replace('/', '__') + '.csv')
            if not os.path.exists(p):
                continue
            with open(p, encoding='utf-8-sig') as fi:
                lines = fi.readlines()
            fo.writelines(lines if not header_done else lines[1:])
            header_done = True

    metas = []
    for rec in recs:
        p = os.path.join(out_dir, 'parts', rec.replace('/', '__') + '.meta.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                metas.append(json.load(f))
    xlsx = os.path.join(out_dir, 'twins_index.xlsx')
    if os.path.exists(xlsx):  # 绝不覆盖可能已人工填写的表
        xlsx = os.path.join(out_dir, f'twins_index_{datetime.now():%Y%m%d_%H%M%S}.xlsx')
        print(f'[提示] twins_index.xlsx 已存在，新表另存为 {os.path.basename(xlsx)}')
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '标注'
    auto_cols = ['folder', '拼板图', 'n_frames', 'capture_time', 'n_bad', 'good_ratio', 'auto_good_frames']
    fill_cols = ['耳号', '背部帧段', '臀部帧段', '废弃帧段', '可用(Y/N)', '标注人', '备注']
    ws.append(auto_cols + fill_cols)
    for m in metas:
        ws.append([m['folder'], f"sheets/{m['folder'].replace('/', '__')}.jpg", m['n_frames'],
                   m['capture_time'], m['n_bad'], m['good_ratio'], m['good_ranges']] + [''] * len(fill_cols))
    yellow = PatternFill('solid', fgColor='FFF2CC')
    all_cols = auto_cols + fill_cols
    for ci, name in enumerate(all_cols, 1):
        head = ws.cell(row=1, column=ci)
        head.font = Font(bold=True)
        if name in fill_cols:
            head.fill = yellow
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=ci).number_format = '@'
    for r in range(2, ws.max_row + 1):
        c = ws.cell(row=r, column=2)
        c.hyperlink = c.value
        c.font = Font(color='0563C1', underline='single')
    for i, wdt in enumerate([14, 26, 9, 17, 7, 10, 22, 12, 14, 14, 14, 10, 8, 24], 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = wdt
    ws.freeze_panes = 'B2'
    wb.save(xlsx)
    return q_path, xlsx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='深度和双目相机数据 根目录')
    ap.add_argument('--out', required=True, help='输出目录（不能在 root 内）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 段（试跑）')
    ap.add_argument('--only', default='', help='逗号分隔的录像段相对路径，如 c1/1,d3/20')
    ap.add_argument('--tiles', type=int, default=30, help='每张拼板抽帧数')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    root, out = os.path.abspath(a.root), os.path.abspath(a.out)
    if os.path.splitdrive(root)[0].lower() == os.path.splitdrive(out)[0].lower() \
            and os.path.commonpath([root, out]) == root:
        sys.exit('[错误] 输出目录不能在原始数据目录内')
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    recs = [s.strip() for s in a.only.split(',') if s.strip()] if a.only else find_recordings(root)
    if a.limit:
        recs = recs[:a.limit]
    print(f'录像段: {len(recs)}   输出: {out}', flush=True)
    with open(os.path.join(out, 'recordings.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(recs))

    jobs = [(root, r, out, a.tiles, a.force) for r in recs]
    errors = []
    with Pool(a.workers) as pool:
        for done, (rec, st) in enumerate(pool.imap_unordered(process, jobs), 1):
            if st.startswith('ERROR'):
                errors.append(f'{rec}\t{st}')
            if done % 10 == 0 or done == len(recs) or st.startswith('ERROR'):
                el = time.time() - t0
                print(f'  [{done}/{len(recs)}] {rec} {st}   {el / 60:.1f} min，预计剩余 '
                      f'{el / done * (len(recs) - done) / 60:.1f} min', flush=True)
    if errors:
        with open(os.path.join(out, 'errors.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(errors))
    q, x = write_outputs(out, recs)
    print(f'完成，用时 {(time.time() - t0) / 60:.1f} min，出错 {len(errors)} 段\n'
          f'  逐帧质量: {q}\n  标注表:   {x}\n  拼板图:   {os.path.join(out, "sheets")}')


if __name__ == '__main__':
    main()
