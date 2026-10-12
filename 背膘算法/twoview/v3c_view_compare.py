# -*- coding: utf-8 -*-
"""
v3c_view_compare.py — V3 第三步：俯视 / 斜后视 / 双视角结果对比（只读 v3b 的 OOF 预测，不训练）

来源 / 背景：06 §5.1 必做第 4 项 → Table 3「视觉（俯视、斜后视、双视角）」行。

比较对象（均为 3 个种子平均的猪级 OOF 预测，同一套 5 折）
  master：俯视 ResNet-18；sub：斜后视 ResNet-18；dual：双分支 ResNet-18（端到端）
  late  ：master 与 sub 猪级预测直接平均（无可调参数，免配准的最简融合）
子集：全部、丹系头胎、新美系经产、首次测定、有上季记录（定义同 v1_tabular_baseline.py）
检验：相对 master 的 ΔR² / ΔMAE（按 group bootstrap 2000 次，95% CI）+ 猪级 |误差| Wilcoxon 符号秩检验；
     另报 master 与 sub 预测的相关（两视角信息重叠程度）

输入：runs/v3_views/{master,sub,dual}_s128/test_preds.csv；runs/v1_tabular/preds.csv（子集变量）
输出：runs/v3_views/compare/{log.txt, result.json, preds.csv}

用法：D:\\Anaconda\\envs\\pytorch\\python.exe v3c_view_compare.py
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
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402

VDIR = os.path.join(HERE, 'runs', 'v3_views')
OUT = os.path.join(VDIR, 'compare')
VIEWS = ('master', 'sub', 'dual')


def read(path):
    with open(path, encoding='utf-8-sig') as f:
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

    log(f'[{time.strftime("%F %T")}] V3 view comparison')
    tab = {r['uid']: r for r in read(os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv'))}
    base, uid = None, None
    P = {}
    for v in VIEWS:
        path = os.path.join(VDIR, f'{v}_s128', 'test_preds.csv')
        if not os.path.isfile(path):
            log(f'   [skip] {v}: {path} 不存在')
            continue
        rows = read(path)
        if base is None:
            base, uid = rows, [r['uid'] for r in rows]
        assert [r['uid'] for r in rows] == uid
        P[v] = np.array([fnum(r['pred_mean']) for r in rows])
        log(f'   {v}: {path}')
    y = np.array([float(r['y']) for r in base])
    group = np.array([r['group'] for r in base])
    if 'master' in P and 'sub' in P:
        P['late'] = (P['master'] + P['sub']) / 2
    line = np.array([tab[u]['line'] for u in uid])
    parity = np.array([fnum(tab[u]['parity']) for u in uid])
    prev = np.array([fnum(tab[u]['prev_bf']) for u in uid])
    ok = np.all([~np.isnan(p) for p in P.values()], axis=0)
    log(f'   pigs with all predictions: {ok.sum()}/{len(y)}')
    subsets = {
        'all': ok,
        'DAN_parity1': ok & (line == 'DAN') & (parity == 1),
        'NAM_multiparous': ok & (line == 'NAM') & (parity >= 2),
        'no_prev_season': ok & np.isnan(prev),
        'has_prev_season': ok & ~np.isnan(prev),
    }
    res = {'n': int(ok.sum()), 'table': {}, 'vs_master': {}}
    log('\nmodel      subset             n    MAE    RMSE   R2     CCC    Spearman')
    for v, p in P.items():
        res['table'][v] = {}
        for s, mk in subsets.items():
            m = metrics(y[mk], p[mk])
            res['table'][v][s] = m
            log(f'{v:10s} {s:16s} {m["n"]:4d}  {m["MAE"]:.3f}  {m["RMSE"]:.3f}  {m["R2"]:.3f}  {m["CCC"]:.3f}  '
                f'{m["Spearman"]:.3f}')
    if 'master' in P and 'sub' in P:
        r = pearsonr(P['master'][ok], P['sub'][ok])[0]
        rr = pearsonr(y[ok] - P['master'][ok], y[ok] - P['sub'][ok])[0]
        res['corr_master_sub'] = dict(pred=float(r), resid=float(rr))
        log(f'\nr(master pred, sub pred) = {r:.3f}; r(master resid, sub resid) = {rr:.3f}')
    if 'master' in P:
        log('\nvs master (Δ = model − master): ΔR² / ΔMAE group-bootstrap 95% CI; Wilcoxon on |error|')
        for v in [k for k in P if k != 'master']:
            res['vs_master'][v] = {}
            for s, mk in subsets.items():
                b = group_bootstrap(group[mk], y[mk], P[v][mk], P['master'][mk])
                ea, eb = np.abs(P[v][mk] - y[mk]), np.abs(P['master'][mk] - y[mk])
                pw = float(wilcoxon(ea, eb).pvalue)
                res['vs_master'][v][s] = dict(**b, wilcoxon_p=pw)
                log(f'   {v:5s} [{s:16s}] ΔR² {b["dR2"][1]:+.3f} ({b["dR2"][0]:+.3f}, {b["dR2"][2]:+.3f})  '
                    f'ΔMAE {b["dMAE"][1]:+.3f} ({b["dMAE"][0]:+.3f}, {b["dMAE"][2]:+.3f})  Wilcoxon p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(P))
        for i, r_ in enumerate(base):
            w.writerow([r_['uid'], r_['group'], r_['fold'], y[i]]
                       + ['' if np.isnan(P[k][i]) else round(P[k][i], 3) for k in P])
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
