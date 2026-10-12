# -*- coding: utf-8 -*-
"""
v7b_dk_batch_shift.py — V7 补充：DK 外测偏差是否来自「批次 / 相机安装」差异；按批次校准能恢复多少

来源 / 背景
  v7_dk_external.py 结果：丰美禾训练的模型在 DK 上整体高估 2–4 mm（R² < 0），但各批次偏差差异大
  （ResNet-18：nkythird +5.2 mm，nkyfirst +1.7，nkysecond +1.3），且批次内 Spearman（0.49–0.58）高于整体（0.36）。
  假设：批次间相机到猪背距离不同 → 系统偏移。

内容
  1. 每批次（丰美禾各日期、DK 三批）猪背深度中位数（每帧有效像素 d≥300 mm 的中位数，再取猪级中位数）、背膘均值
  2. 每批次：各模型平均偏差、Spearman、Pearson
  3. 按批次少样本截距校准：每批随机抽 k=5/10 头 B 超校正该批截距，在本批其余猪上评估；三批合并报告 MAE / R² / Spearman
     （对照：整体截距校准，k 取三批合计相同头数）。重复 200 次。

输入：runs/v7_dk_external/preds.csv；yolo+FastSAM/depthmap_dataset（深度帧）
输出：runs/v7_dk_external/batch_analysis/{log.txt, result.json}

用法：D:\\Anaconda\\envs\\pytorch\\python.exe v7b_dk_batch_shift.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import DM_DIR, metrics  # noqa: E402

SRC = os.path.join(HERE, 'runs', 'v7_dk_external', 'preds.csv')
OUT = os.path.join(HERE, 'runs', 'v7_dk_external', 'batch_analysis')
MODELS = ('resnet18_mean3', 'dinov3_relh_ridge', 'dinov3_normal_ridge')


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] input {SRC}')
    with open(SRC, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    batch = np.array([r['batch'] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    P = {m: np.array([float(r[m]) for r in rows]) for m in MODELS}

    # 1. 猪背深度中位数（丰美禾按日期、DK 按批次）
    with open(os.path.join(DM_DIR, 'metadata.csv'), encoding='utf-8') as f:
        meta = list(csv.DictReader(f))
    per_pig = {}
    for r in meta:
        per_pig.setdefault((r['source'], r['pig_id']), []).append(r['frame_path'])
    depth = {}
    for (src, pid), fps in per_pig.items():
        meds = []
        for fp in fps[::4]:                       # 每猪抽约 5 帧
            d = np.load(os.path.join(DM_DIR, fp))
            v = d[d >= 300]
            if v.size:
                meds.append(np.median(v))
        if meds:
            depth.setdefault(src, []).append(np.median(meds))
    log('\n1. median camera-to-back depth by batch (pig-level median of frame medians)')
    dep_tab = {}
    for src in sorted(depth):
        v = np.array(depth[src])
        dep_tab[src] = dict(n=len(v), median_mm=float(np.median(v)),
                            iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))])
        log(f'   {src:12s} n={len(v):4d}  depth median {np.median(v):6.0f} mm  (IQR {np.percentile(v, 25):.0f}–'
            f'{np.percentile(v, 75):.0f})')

    # 2. DK 每批次：偏差与排序
    log('\n2. DK per-batch bias / ranking')
    per_batch = {}
    for b in sorted(set(batch)):
        mk = batch == b
        per_batch[b] = {'n': int(mk.sum()), 'y_mean': float(y[mk].mean()), 'y_sd': float(y[mk].std())}
        msg = f'   {b:10s} n={mk.sum():3d} backfat {y[mk].mean():.2f}±{y[mk].std():.2f}'
        for m in MODELS:
            bias = float(np.mean(P[m][mk] - y[mk]))
            sp = float(spearmanr(P[m][mk], y[mk])[0])
            pr = float(pearsonr(P[m][mk], y[mk])[0])
            per_batch[b][m] = dict(bias=bias, spearman=sp, pearson=pr)
            msg += f' | {m}: bias {bias:+.2f} ρ {sp:.3f} r {pr:.3f}'
        log(msg)

    # 3. 按批次 vs 整体 截距校准
    rng = np.random.default_rng(0)
    batches = sorted(set(batch))
    calib = {}
    log('\n3. few-shot intercept calibration: per-batch vs pooled (200 draws; evaluated on non-calibration pigs)')
    for m in MODELS:
        calib[m] = {}
        for k in (5, 10):
            res = {'per_batch': [], 'pooled': []}
            for _ in range(200):
                cal_idx = np.concatenate([rng.choice(np.where(batch == b)[0], k, replace=False) for b in batches])
                rest = np.setdiff1d(np.arange(len(y)), cal_idx)
                pb = P[m].copy()
                for b in batches:
                    cb = cal_idx[batch[cal_idx] == b]
                    pb[batch == b] += np.mean(y[cb] - P[m][cb])
                pp = P[m] + np.mean(y[cal_idx] - P[m][cal_idx])
                for name, pr_ in (('per_batch', pb), ('pooled', pp)):
                    mm = metrics(y[rest], pr_[rest])
                    res[name].append((mm['MAE'], mm['R2'], mm['Spearman']))
            calib[m][k] = {name: dict(zip(('MAE', 'R2', 'Spearman'),
                                          [(float(np.mean(v)), float(np.std(v))) for v in np.array(vals).T]))
                           for name, vals in res.items()}
            c = calib[m][k]
            log(f'   {m:20s} k={k:2d}/batch ({3 * k} pigs total)  per-batch: MAE {c["per_batch"]["MAE"][0]:.3f} '
                f'R2 {c["per_batch"]["R2"][0]:.3f} ρ {c["per_batch"]["Spearman"][0]:.3f}  | pooled: MAE '
                f'{c["pooled"]["MAE"][0]:.3f} R2 {c["pooled"]["R2"][0]:.3f}')

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(depth_by_batch=dep_tab, dk_per_batch=per_batch, calibration=calib), f, ensure_ascii=False,
                  indent=1, default=float)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
