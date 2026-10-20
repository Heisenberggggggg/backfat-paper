# -*- coding: utf-8 -*-
"""
v5g_line_from_shape.py — 相机体型能否区分遗传系？卡尺读数的含义是否依赖遗传系？（解释 Table 3 中「记录」对相机与卡尺的增量差异）

来源 / 背景
  本人 10-13 指出 Results 3.3 的读感矛盾：相机单独（0.378）远胜记录（0.158）与卡尺（0.205），但记录 + 相机（0.357）
  ≈ 记录 + 卡尺（0.371）。同口径下（v4h 嵌套分数）：图像 0.346 → 记录 + 图像 0.357（+0.011）；卡尺 0.205 → 记录 + 卡尺
  0.371（+0.166）。假设：卡尺读数的含义随品系变化，需要生产记录提供品系；相机体型本身已携带品系信息。
检验
  1. 体型 → 品系：6 项整体体尺（v4e）、几何特征 G（v5b），按个体分组 5 折逻辑回归的 AUC（新美系 vs 丹系）
  2. 卡尺 → 品系：卡尺读数单独区分品系的 AUC；各品系卡尺读数与背膘的均值 ± SD

输入：runs/v1_tabular/preds.csv、runs/v4_hmap/body_dims/body_dims.csv、runs/v5_caliper/features.csv、runs/v1b_caliper/preds.csv
输出：runs/v5_line_shape/{log.txt, result.json}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5g_line_from_shape.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
OUT = os.path.join(RUNS, 'v5_line_shape')
B_COLS = ('back_h_mm', 'length_mm', 'width_mm', 'area_cm2', 'volume_l', 'hw_ratio')
G_COLS = ('angle_f0.60_D150', 'curv_f0.60_D150', 'shoulder_w', 'rib_w', 'hip_w', 'hip_h')


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return {r['uid']: r for r in csv.DictReader(f)}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] v5g genetic line from body shape vs caliper')
    v1 = read(os.path.join(RUNS, 'v1_tabular', 'preds.csv'))
    body = read(os.path.join(RUNS, 'v4_hmap', 'body_dims', 'body_dims.csv'))
    feat = read(os.path.join(RUNS, 'v5_caliper', 'features.csv'))
    calr = read(os.path.join(RUNS, 'v1b_caliper', 'preds.csv'))
    uid = sorted(v1)
    nam = np.array([v1[u]['line'] == 'NAM' for u in uid]).astype(int)
    group = np.array([v1[u]['group'] for u in uid])
    y = np.array([float(v1[u]['y']) for u in uid])
    cal = np.array([fnum(calr[u]['caliper']) if calr[u]['caliper_flag'] != 'implausible' else np.nan for u in uid])
    X = {'whole-body dimensions': np.array([[fnum(body[u][c]) for c in B_COLS] for u in uid]),
         'image geometry': np.array([[fnum(feat[u][c]) for c in G_COLS] for u in uid]),
         'caliper score': cal[:, None]}
    log(f'   n={len(uid)} (American line {nam.sum()}, Danish line {len(uid) - nam.sum()})')
    res = {}
    for name, M in X.items():
        ok = ~np.isnan(M).any(1)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        p = cross_val_predict(clf, M[ok], nam[ok], groups=group[ok], cv=GroupKFold(5), method='predict_proba')[:, 1]
        auc = float(roc_auc_score(nam[ok], p))
        res[name] = dict(n=int(ok.sum()), auc=auc)
        log(f'   line from {name:22s} n={ok.sum()}  AUC {auc:.3f} (grouped 5-fold logistic regression)')
    log('\n   caliper and backfat by line (mean ± SD)')
    for nm, m in (('Danish', nam == 0), ('American', nam == 1)):
        k = m & ~np.isnan(cal)
        res[nm] = dict(caliper=[float(cal[k].mean()), float(cal[k].std())], backfat=[float(y[m].mean()), float(y[m].std())])
        log(f'   {nm:8s} caliper {cal[k].mean():.2f} ± {cal[k].std():.2f} (n={k.sum()}); '
            f'backfat {y[m].mean():.2f} ± {y[m].std():.2f} mm (n={m.sum()})')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
