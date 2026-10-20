# -*- coding: utf-8 -*-
"""
v4c_hmap_compare.py — 标准化高度图 vs 俯视深度图（只读 OOF 预测，不训练）

来源 / 背景：v4b_hmap_models.py 的高度图 ResNet-18（3 种子平均）与 v3b 的俯视 ResNet-18（3 种子平均，
  runs/v3_views/master_s128）同一套 5 折、同一套内层验证集、同一训练设置，只换输入表示。

比较：hmap vs master；另报两者猪级预测平均（late）vs master；子集同 v3c；ΔR² / ΔMAE 按 group bootstrap
  2000 次（复用 v1_tabular_baseline.group_bootstrap），猪级 |误差| Wilcoxon。
另报每个种子单独的 R²（种子间波动）。

输入：runs/v4_hmap/hmap/test_preds.csv、runs/v3_views/master_s128/test_preds.csv、runs/v1_tabular/preds.csv
输出：runs/v4_hmap/compare/{log.txt, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4c_hmap_compare.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, wilcoxon

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402

SRC = {'master': os.path.join(RUNS_ROOT, 'v3_views', 'master_s128', 'test_preds.csv'),
       'hmap': os.path.join(RUNS_ROOT, 'v4_hmap', 'hmap', 'test_preds.csv')}
OUT = os.path.join(RUNS_ROOT, 'v4_hmap', 'compare')


def read(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] height map vs top-view depth map')
    rows = {k: read(p) for k, p in SRC.items()}
    uid = [r['uid'] for r in rows['master']]
    assert uid == [r['uid'] for r in rows['hmap']], '两份预测猪顺序不一致'
    assert [r['fold'] for r in rows['master']] == [r['fold'] for r in rows['hmap']], '折不一致'
    y = np.array([float(r['y']) for r in rows['master']])
    group = np.array([r['group'] for r in rows['master']])
    P = {k: np.array([fnum(r['pred_mean']) for r in v]) for k, v in rows.items()}
    P['late'] = (P['master'] + P['hmap']) / 2
    for k in ('master', 'hmap'):
        seeds = [c for c in rows[k][0] if c.startswith('pred_s')]
        log(f'   {k}: per-seed R² ' + ', '.join(
            f'{c[5:]} {metrics(y, np.array([fnum(r[c]) for r in rows[k]]))["R2"]:.3f}' for c in seeds))
    tab = {r['uid']: r for r in read(os.path.join(RUNS_ROOT, 'v1_tabular', 'preds.csv'))}
    line = np.array([tab[u]['line'] for u in uid])
    parity = np.array([fnum(tab[u]['parity']) for u in uid])
    prev = np.array([fnum(tab[u]['prev_bf']) for u in uid])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    res = {'table': {}, 'vs_master': {}}
    log('\nmodel   subset             n    MAE    RMSE   R2     CCC    Spearman')
    for k, p in P.items():
        res['table'][k] = {}
        for s, m in subsets.items():
            mm = metrics(y[m], p[m])
            res['table'][k][s] = mm
            log(f'{k:7s} {s:16s} {mm["n"]:4d}  {mm["MAE"]:.3f}  {mm["RMSE"]:.3f}  {mm["R2"]:.3f}  {mm["CCC"]:.3f}  '
                f'{mm["Spearman"]:.3f}')
    r = pearsonr(y - P['master'], y - P['hmap'])[0]
    res['resid_corr_master_hmap'] = float(r)
    log(f'\nr(master resid, hmap resid) = {r:.3f}')
    log('\nvs master (Δ = model − master): group bootstrap 95% CI; Wilcoxon on |error|')
    for k in ('hmap', 'late'):
        res['vs_master'][k] = {}
        for s, m in subsets.items():
            b = group_bootstrap(group[m], y[m], P[k][m], P['master'][m])
            pw = float(wilcoxon(np.abs(P[k][m] - y[m]), np.abs(P['master'][m] - y[m])).pvalue)
            res['vs_master'][k][s] = dict(**b, wilcoxon_p=pw)
            log(f'   {k:5s} [{s:16s}] ΔR² {b["dR2"][1]:+.3f} ({b["dR2"][0]:+.3f}, {b["dR2"][2]:+.3f})  '
                f'ΔMAE {b["dMAE"][1]:+.3f} ({b["dMAE"][0]:+.3f}, {b["dMAE"][2]:+.3f})  Wilcoxon p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(P))
        for i, r_ in enumerate(rows['master']):
            w.writerow([r_['uid'], r_['group'], r_['fold'], y[i]] + [round(P[k][i], 3) for k in P])
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
