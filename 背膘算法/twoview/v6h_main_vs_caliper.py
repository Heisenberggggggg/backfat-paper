# -*- coding: utf-8 -*-
"""
v6h_main_vs_caliper.py — 新主模型（解剖引导网络，5 种子平均）与人工卡尺、生产记录、ResNet-18 基线的配对比较

来源 / 背景
  DECISIONS 2026-10-15：主模型换成 v6f prof。稿件 3.4「相机 vs 卡尺」原用 v5f（ResNet-18 三种子 vs 卡尺），
  需按新主模型重算。只用已有 OOF，不训练。

比较（group bootstrap 95% CI + Wilcoxon，子集：全体、丹系头胎、新美系经产）
  - 主模型 vs 卡尺单独（C，TabPFN，来自 v6a anatomy 运行）
  - 主模型 vs 生产记录单独（R）
  - 主模型 vs ResNet-18 基线（v6f base，同代码 5 种子）

输入：runs/v6_anatomy/{prof,base}/test_preds.csv、runs/v6_camera_caliper_anatomy/preds.csv、runs/v1_tabular/preds.csv
输出：runs/v6_main_vs_caliper/{log.txt, result.json}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6h_main_vs_caliper.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402

RUNS = RUNS_ROOT
OUT = os.path.join(RUNS, 'v6_main_vs_caliper')


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return {r['uid']: r for r in csv.DictReader(f)}


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    prof = read(os.path.join(RUNS, 'v6_anatomy', 'prof', 'test_preds.csv'))
    base = read(os.path.join(RUNS, 'v6_anatomy', 'base', 'test_preds.csv'))
    tab = read(os.path.join(RUNS, 'v6_camera_caliper_anatomy', 'preds.csv'))
    v1 = read(os.path.join(RUNS, 'v1_tabular', 'preds.csv'))
    uid = sorted(prof)
    assert set(uid) == set(base) == set(tab) == set(v1)
    y = np.array([float(prof[u]['y']) for u in uid])
    assert np.allclose(y, [float(tab[u]['y']) for u in uid])
    group = np.array([prof[u]['group'] for u in uid])
    line = np.array([v1[u]['line'] for u in uid])
    parity = np.array([float(v1[u]['parity']) if v1[u]['parity'] != '' else np.nan for u in uid])
    P = {'main (anatomy-guided, 5 seeds)': np.array([float(prof[u]['pred_inner']) for u in uid]),
         'ResNet-18 baseline (5 seeds)': np.array([float(base[u]['pred_inner']) for u in uid]),
         'caliper (C)': np.array([float(tab[u]['C']) for u in uid]),
         'production records (R)': np.array([float(tab[u]['R']) for u in uid])}
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2)}
    log(f'[{time.strftime("%F %T")}] v6h | pigs {len(y)}')
    res = {'table': {}, 'comparisons': {}}
    for nm, p in P.items():
        res['table'][nm] = {s: metrics(y[m], p[m]) for s, m in subsets.items()}
        log(f'  {nm:32s} ' + ' | '.join(f'{s} R2 {res["table"][nm][s]["R2"]:.3f} MAE {res["table"][nm][s]["MAE"]:.3f}'
                                         for s in subsets))
    main_k = 'main (anatomy-guided, 5 seeds)'
    log('\npaired comparisons (Δ = main − other): group bootstrap 95% CI; Wilcoxon on |error|')
    for other in ('caliper (C)', 'production records (R)', 'ResNet-18 baseline (5 seeds)'):
        for s, m in subsets.items():
            bs = group_bootstrap(group[m], y[m], P[main_k][m], P[other][m])
            pw = float(wilcoxon(np.abs(P[main_k][m] - y[m]), np.abs(P[other][m] - y[m])).pvalue)
            res['comparisons'][f'main vs {other} [{s}]'] = dict(**bs, wilcoxon_p=pw)
            log(f'   main vs {other:30s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
