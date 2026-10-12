# -*- coding: utf-8 -*-
"""
v3a_gen_sub_depthmaps.py — V3 第一步：斜后视（sub）掩膜深度图数据集（与俯视 depthmap_dataset 同一处理）

来源 / 背景
  06 §5.1 必做第 3 / 4 项：V3 俯视 / 斜后视 / 双视角对比。俯视 ResNet-18 使用
  yolo+FastSAM/depthmap_dataset（generate_depthmap_dataset.py，2026-04-01）。sub 原始帧为整幅场景
  （猪 + 栏杆 + 地面 + 邻栏猪），需按同一流程抠出目标猪，才能在统一协议下比较视角。

与 generate_depthmap_dataset.py 的关系
  - 直接 import 其 get_frame_pairs / get_pig_mask / crop_masked_depthmap / imread_*（YOLO11n 取面积最大框
    + FastSAM 取与框重叠最大的掩膜；每猪均匀 20 帧；裁外框 → 保比例缩放 256×256 → 背景 0；float32 mm）
  - 猪列表与目录改由 index_v2.csv 提供（uid 与 V1/V6 等全部分析一致；sub_dir 已区分 sub / sub2 / sub10），
    不再用旧的耳号子串匹配；标签不写进本数据集（训练时从 index_v2 取）
  - 新增质控：每帧掩膜面积占比、掩膜质心、YOLO 框数；每猪保存一张抠图预览（中间帧）

输入：twoview/index_v2.csv；G:\\耿来硕数据\\...\\{sub_dir}\\{color,depth}
输出（新目录，不覆盖任何已有数据）：yolo+FastSAM/depthmap_dataset_sub/
  {uid}/frame_XXX.npy、metadata.csv（frame_path, uid, farm, batch, frame_idx, mask_frac, cx, cy, n_box）、
  _preview/{uid}.jpg、gen_log.txt
  中断后重跑会跳过 metadata.csv 中已有的猪。

用法：D:\\Anaconda\\envs\\pytorch\\python.exe v3a_gen_sub_depthmaps.py [--farm FMH|DK|all] [--limit N]
"""
import argparse
import csv
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
OUT = os.path.join(SEG_DIR, 'depthmap_dataset_sub')


def preview(color, mask, path):
    ov = color.copy()
    ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    s = 480 / max(ov.shape[:2])
    ov = cv2.resize(ov, (int(ov.shape[1] * s), int(ov.shape[0] * s)))
    cv2.imencode('.jpg', ov)[1].tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--farm', default='all', choices=['FMH', 'DK', 'all'])
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 头（试跑）')
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, '_preview'), exist_ok=True)
    logf = open(os.path.join(OUT, 'gen_log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    with open(INDEX, encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['sub_dir'] and (a.farm == 'all' or r['farm'] == a.farm)]
    rows.sort(key=lambda r: r['uid'])
    if a.limit:
        rows = rows[:a.limit]
    meta_path = os.path.join(OUT, 'metadata.csv')
    done = set()
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            mrows = list(csv.DictReader(f))
        if mrows:  # 中断时最后一头猪可能只写了部分帧 → 删去其记录并重做
            last = mrows[-1]['uid']
            keep = [r for r in mrows if r['uid'] != last]
            with open(meta_path, 'w', newline='', encoding='utf-8') as f:
                w0 = csv.DictWriter(f, fieldnames=list(mrows[0].keys()))
                w0.writeheader()
                w0.writerows(keep)
            done = {r['uid'] for r in keep}
            print(f'resume: redo last (possibly partial) pig {last}')
    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] sub depthmaps: {len(rows)} pigs (farm={a.farm}), already done {len(done)}')

    cwd = os.getcwd()
    os.chdir(SEG_DIR)  # 模型权重为相对路径
    yolo, fsam = G.load_models()
    os.chdir(cwd)

    new_file = not os.path.isfile(meta_path)
    mf = open(meta_path, 'a', newline='', encoding='utf-8')
    w = csv.writer(mf)
    if new_file:
        w.writerow(['frame_path', 'uid', 'farm', 'batch', 'frame_idx', 'mask_frac', 'cx', 'cy', 'n_box'])
    n_ok, n_fail = 0, []
    for i, r in enumerate(rows, 1):
        uid = r['uid']
        if uid in done:
            continue
        pairs = G.get_frame_pairs(os.path.join(r['sub_dir'], 'color'), os.path.join(r['sub_dir'], 'depth'),
                                  G.FRAMES_PER_PIG)
        os.makedirs(os.path.join(OUT, uid), exist_ok=True)
        saved = 0
        for j, (cp, dp) in enumerate(pairs):
            color = G.imread_chinese(cp)
            depth = G.imread_depth_chinese(dp) if color is not None else None
            if color is None or depth is None:
                continue
            n_box = len(yolo(color, conf=G.YOLO_CONF, verbose=False)[0].boxes)
            mask = G.get_pig_mask(color, yolo, fsam)
            if mask is None:
                continue
            if mask.shape != depth.shape:
                mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            crop = G.crop_masked_depthmap(depth, mask, G.TARGET_SIZE)
            if crop is None:
                continue
            fn = f'frame_{j:03d}.npy'
            np.save(os.path.join(OUT, uid, fn), crop)
            ys, xs = np.nonzero(mask)
            w.writerow([f'{uid}\\{fn}', uid, r['farm'], r['batch'], j, round(float(mask.mean()), 4),
                        round(float(xs.mean() / mask.shape[1]), 3), round(float(ys.mean() / mask.shape[0]), 3),
                        n_box])
            if j == len(pairs) // 2:
                preview(color if color.shape[:2] == mask.shape else cv2.resize(color, mask.shape[::-1]), mask,
                        os.path.join(OUT, '_preview', f'{uid}.jpg'))
            saved += 1
        mf.flush()
        if saved:
            n_ok += 1
        else:
            n_fail.append(uid)
        if i % 50 == 0 or i <= 2:
            el = time.time() - t0
            log(f'  [{i}/{len(rows)}] {uid} frames {saved}/{len(pairs)}  elapsed {el / 60:.1f} min, '
                f'ETA {el / i * (len(rows) - i) / 60:.1f} min')
    mf.close()
    log(f'done: ok {n_ok}, failed {len(n_fail)} {n_fail[:20]} | {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
