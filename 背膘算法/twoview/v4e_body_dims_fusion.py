# -*- coding: utf-8 -*-
"""
v4e_body_dims_fusion.py — 高度图得到的物理单位体尺（猪级）× 生产记录 × 卡尺 → TabPFN（同一套 5 折）

来源 / 背景
  DECISIONS 2026-10-13：标准化高度图保留绝对体尺；v4a 已为每帧输出背高、体长、体宽、面积、体积。
  胡宝玖学位论文 §6.2 展望 ③「结合反映个体差异的体型特征」；05 §8.2 A4。问题：
    B1  体尺单独能解释多少背膘变异（对照生产记录 R）
    B2  R + 体尺 vs R；R + 卡尺 + 体尺 vs R + 卡尺；R + 上季 + 卡尺 + 体尺 vs R + 上季 + 卡尺
  体尺是少数几列可解释特征，可直接进 TabPFN，无需堆叠（无泄漏）。

体尺（每头猪取通过帧质控的帧的中位数；帧质控同 v4b：ok=1、面积 ≥ 本猪中位 80%、体长 ≥ 本猪中位 85%）
  back_h_mm（背高，离地高度 98% 分位）、length_mm、width_mm、area_cm2、volume_l、hw_ratio = back_h / width（矮胖度）
  另报各体尺与背膘的 Pearson 相关（全体、丹系、新美系）。

输入：runs/v1_tabular/preds.csv（猪、折、品系、胎次、妊娠日龄、上季背膘）、runs/v1b_caliper/preds.csv（卡尺）、
      yolo+FastSAM/hmap_dataset/metadata.csv
输出：runs/v4_hmap/body_dims/{log.txt, result.json, preds.csv, body_dims.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4e_body_dims_fusion.py
"""
import argparse
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
import a3_foundation_regression as A3  # noqa: E402
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402

V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
HMETA = os.path.join(HERE, '..', 'yolo+FastSAM', 'hmap_dataset', 'metadata.csv')
OUT = os.path.join(HERE, 'runs', 'v4_hmap', 'body_dims')
DIMS = ('back_h_mm', 'length_mm', 'width_mm', 'area_cm2', 'volume_l', 'hw_ratio')


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def pig_dims(uids, area_rel=0.8, len_rel=0.85):
    by = {}
    with open(HMETA, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ok'] == '1':
                by.setdefault(r['uid'], []).append(r)
    out = np.full((len(uids), len(DIMS)), np.nan)
    nfr = np.zeros(len(uids), int)
    for i, u in enumerate(uids):
        rs = by.get(u, [])
        if not rs:
            continue
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        rs = [r for r in rs if float(r['area_cm2']) >= area_rel * ma and float(r['length_mm']) >= len_rel * ml]
        v = np.array([[float(r[k]) for k in DIMS[:5]] for r in rs])
        med = np.median(v, 0)
        out[i] = list(med) + [med[0] / med[2]]
        nfr[i] = len(rs)
    return out, nfr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['auto', 'cpu'], default='auto')
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
        cal_map = {r['uid']: fnum(r['caliper']) for r in csv.DictReader(f)}
    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    cal = np.array([cal_map.get(u, np.nan) for u in uid])
    B, nfr = pig_dims(uid)
    log(f'[{time.strftime("%F %T")}] pigs {len(y)}; body dims missing {int(np.isnan(B[:, 0]).sum())}; '
        f'frames/pig median {np.median(nfr):.0f} (min {nfr.min()}); caliper missing {int(np.isnan(cal).sum())}')
    with open(os.path.join(OUT, 'body_dims.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'line', 'parity', 'y', 'n_frames'] + list(DIMS))
        for i, u in enumerate(uid):
            w.writerow([u, line[i], parity[i], y[i], nfr[i]] + [round(float(v), 3) for v in B[i]])

    log('\nPearson r(body dim, backfat): all | Danish | American')
    corr = {}
    for j, k in enumerate(DIMS):
        c = {}
        for nm, m in (('all', np.ones(len(y), bool)), ('DAN', line == 'DAN'), ('NAM', line == 'NAM')):
            ok = m & ~np.isnan(B[:, j])
            c[nm] = float(pearsonr(B[ok, j], y[ok])[0])
        corr[k] = c
        log(f'   {k:10s} {c["all"]:+.3f} | {c["DAN"]:+.3f} | {c["NAM"]:+.3f}')

    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    C = cal[:, None]
    sets = {'B': B, 'R': R, 'R+B': np.hstack([R, B]), 'R+cal': np.hstack([R, C]), 'R+cal+B': np.hstack([R, C, B]),
            'R+prev+cal': np.hstack([R, P, C]), 'R+prev+cal+B': np.hstack([R, P, C, B])}
    preds = {k: np.zeros(len(y)) for k in sets}
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        for name, X in sets.items():
            preds[name][te] = A3.fit_predict_tabpfn(X[tr], y[tr], X[te], 42 + k)
        log(f'  fold {k} done | device {sorted(A3.DEVICE_USED)} | {(time.time() - t0) / 60:.1f} min')

    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel           subset             n    MAE    RMSE   R2     CCC')
    for nm in preds:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:15s} {s:16s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:.3f}  {r["CCC"]:.3f}')
    pairs = [('B', 'R', 'body dims alone vs records'), ('R+B', 'R', 'B2a body dims over records'),
             ('R+cal+B', 'R+cal', 'B2b body dims over records+caliper'),
             ('R+prev+cal+B', 'R+prev+cal', 'B2c body dims over records+prev+caliper')]
    comp = {}
    log('\npaired comparisons (Δ = a − b): group bootstrap 95% CI; Wilcoxon on |error|')
    for a_, b_, lab in pairs:
        comp[lab] = {}
        for s in ('all', 'DAN_parity1', 'NAM_multiparous', 'no_prev_season'):
            m = subsets[s]
            bs = group_bootstrap(group[m], y[m], preds[a_][m], preds[b_][m])
            pw = float(wilcoxon(np.abs(preds[a_][m] - y[m]), np.abs(preds[b_][m] - y[m])).pvalue)
            comp[lab][s] = dict(**bs, wilcoxon_p=pw)
            log(f'   {lab:42s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f}  p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(corr=corr, table=table, comparisons=comp), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(preds))
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], y[i]] + [round(float(preds[k][i]), 3) for k in preds])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
