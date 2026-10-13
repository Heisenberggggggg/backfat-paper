# -*- coding: utf-8 -*-
"""
v5c_virtual_caliper_fusion.py — 虚拟体况卡尺（v5b）进入融合：能否替代人工卡尺？在 CNN 之上还有没有增量？

来源 / 背景
  v5b：虚拟卡尺角（f 0.60、D 150 mm，预先指定）与人工卡尺 r 0.69，但与背膘的相关弱于人工卡尺。
  本脚本在同一套 5 折上用 TabPFN 比较（与 v4e / v4h 相同的做法与配对检验）：
    V vs C            单独使用：虚拟卡尺 vs 人工卡尺
    R+V vs R+C        有生产记录时，虚拟卡尺能否替代人工卡尺
    R+V vs R          虚拟卡尺相对生产记录的增量
    R+G vs R+B        分段体尺 + 虚拟卡尺（G）vs v4e 整体体尺（B）
    R+C+G vs R+C      图像几何特征在人工卡尺之上的增量
    R+img+G vs R+img  图像几何特征在高度图 CNN 分数之上的增量
    R+P+C+img+G vs R+P+C+img
  G = 虚拟卡尺角 + 背部曲率（f 0.60、D 150）+ 肩宽、肋宽、臀宽、臀高（v5b）。
  img = v4h 的无泄漏 CNN 分数：测试猪 = 外层模型（seed 42）预测，训练猪 = 内层 OOF（inner_scores.csv），不重新训练。

输入：runs/v1_tabular/preds.csv、runs/v1b_caliper/preds.csv、runs/v5_caliper/features.csv、
      runs/v4_hmap/nested_fusion/{preds.csv, inner_scores.csv}、hmap_dataset/metadata.csv（v4e 体尺）
输出：runs/v5_caliper/fusion/{log.txt, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5c_virtual_caliper_fusion.py --device cpu
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a3_foundation_regression as A3  # noqa: E402
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v4e_body_dims_fusion import pig_dims  # noqa: E402

V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
FEAT = os.path.join(HERE, 'runs', 'v5_caliper', 'features.csv')
NEST = os.path.join(HERE, 'runs', 'v4_hmap', 'nested_fusion')
OUT = os.path.join(HERE, 'runs', 'v5_caliper', 'fusion')
V_MAIN = 'angle_f0.60_D150'
G_COLS = (V_MAIN, 'curv_f0.60_D150', 'shoulder_w', 'rib_w', 'hip_w', 'hip_h')


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['auto', 'cpu'], default='cpu')
    a = ap.parse_args()
    A3.FORCE_CPU = a.device == 'cpu'
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(V1, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(CAL, encoding='utf-8-sig') as f:
        cal_map = {r['uid']: (fnum(r['caliper']) if r['caliper_flag'] != 'implausible' else np.nan)
                   for r in csv.DictReader(f)}
    with open(FEAT, encoding='utf-8-sig') as f:
        feat = {r['uid']: r for r in csv.DictReader(f)}
    with open(os.path.join(NEST, 'preds.csv'), encoding='utf-8-sig') as f:
        nest = {r['uid']: r for r in csv.DictReader(f)}
    with open(os.path.join(NEST, 'inner_scores.csv'), encoding='utf-8-sig') as f:
        inner = {r['uid']: r for r in csv.DictReader(f)}

    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    assert all(u in feat and u in nest and u in inner for u in uid), 'uid 不一致'
    assert np.array_equal(fold, [int(nest[u]['fold']) for u in uid]), '折不一致'
    cal = np.array([cal_map.get(u, np.nan) for u in uid])
    G = np.array([[fnum(feat[u].get(c)) for c in G_COLS] for u in uid])
    B, _ = pig_dims(uid)
    outer_score = np.array([fnum(nest[u]['img_raw (outer CNN, seed 42)']) for u in uid])
    folds = sorted(set(fold))
    inner_by_fold = {k: np.array([fnum(inner[u][f'inner_score_outer{k}']) for u in uid]) for k in folds}
    for k in folds:
        assert not np.isnan(inner_by_fold[k][fold != k]).any(), f'外层 {k} 训练猪缺内层分数'
    log(f'[{time.strftime("%F %T")}] v5c | pigs {len(y)}; caliper missing {int(np.isnan(cal).sum())}; '
        f'G missing per col {np.isnan(G).sum(0).tolist()} ({", ".join(G_COLS)})')

    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    C = cal[:, None]
    V = G[:, :1]
    sets = {
        'C': lambda I: C, 'V': lambda I: V, 'G': lambda I: G,
        'R': lambda I: R, 'R+C': lambda I: np.hstack([R, C]), 'R+V': lambda I: np.hstack([R, V]),
        'R+G': lambda I: np.hstack([R, G]), 'R+B': lambda I: np.hstack([R, B]),
        'R+C+G': lambda I: np.hstack([R, C, G]),
        'R+img': lambda I: np.hstack([R, I]), 'R+img+G': lambda I: np.hstack([R, I, G]),
        'R+P+C+img': lambda I: np.hstack([R, P, C, I]), 'R+P+C+img+G': lambda I: np.hstack([R, P, C, I, G]),
    }
    preds = {k: np.zeros(len(y)) for k in sets}
    for k in folds:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        I = np.where(fold == k, outer_score, inner_by_fold[k])[:, None]
        for name, fn in sets.items():
            X = fn(I)
            preds[name][te] = A3.fit_predict_tabpfn(X[tr], y[tr], X[te], 42 + k)
        log(f'  fold {k} done | device {sorted(A3.DEVICE_USED)} | {(time.time() - t0) / 60:.1f} min')

    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel           subset             n    MAE    RMSE   R2     CCC')
    for nm in preds:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:15s} {s:16s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:.3f}  {r["CCC"]:.3f}')
    pairs = [('V', 'C', 'virtual vs manual caliper (alone)'),
             ('R+V', 'R+C', 'virtual vs manual caliper (with records)'),
             ('R+V', 'R', 'virtual caliper over records'),
             ('R+G', 'R+B', 'segmental dims+virtual caliper vs whole-body dims'),
             ('R+C+G', 'R+C', 'image geometry over records+manual caliper'),
             ('R+img+G', 'R+img', 'image geometry over records+CNN'),
             ('R+P+C+img+G', 'R+P+C+img', 'image geometry over best fusion')]
    comp = {}
    log('\npaired comparisons (Δ = a − b): group bootstrap 95% CI; Wilcoxon on |error|')
    for a_, b_, lab in pairs:
        comp[lab] = {}
        for s in subsets:
            m = subsets[s]
            bs = group_bootstrap(group[m], y[m], preds[a_][m], preds[b_][m])
            pw = float(wilcoxon(np.abs(preds[a_][m] - y[m]), np.abs(preds[b_][m] - y[m])).pvalue)
            comp[lab][s] = dict(**bs, wilcoxon_p=pw)
            log(f'   {lab:50s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f}  p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(G_cols=G_COLS, table=table, comparisons=comp), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(preds))
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], y[i]] + [round(float(preds[k][i]), 3) for k in preds])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
