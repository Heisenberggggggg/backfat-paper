# -*- coding: utf-8 -*-
"""
v4h_hmap_nested_fusion.py — 高度图 ResNet-18 的嵌套堆叠融合（无泄漏）：图像分数 × 记录 × 卡尺 × 上季背膘 × 体尺 → TabPFN

来源 / 背景
  DECISIONS 2026-10-13（二）：高度图 ResNet-18 场内 R² 0.378 为最强视觉模型；冻结 DINOv3 用在高度图上反而更差，
  所以要把训练过的高度图 CNN 无泄漏地放进融合，只能做嵌套训练。本人 10-13 决定做。

设计（外层 = 同一套 5 折；与 a3_foundation_regression.py --vis stack 同一思路，只是图像模型换成训练的 CNN）
  - 测试猪的图像分数：外层模型（v4b，seed 42）在该折测试猪上的预测 = runs/v4_hmap/hmap/test_preds.csv 的 pred_s42。
    该模型只用外层训练猪训练（内层验证集选 epoch），对测试猪无泄漏。
  - 训练猪的图像分数：外层训练猪再按个体分内层 5 折（StratifiedGroupKFold(5, shuffle, random_state=100+k)）；
    每个内层折用其余 4 份训练高度图 ResNet-18（再留约 1/7 个体作验证集选 epoch，StratifiedGroupKFold(7, seed=200+10k+j)
    第 1 份），预测该份 → 每头训练猪的分数都来自没见过它的模型。种子 42（与测试猪分数同种子）。
  - 训练函数、帧质控、增广与 v4b 完全相同（import）。
  - 第二层：TabPFN（复用 a3_foundation_regression.fit_predict_tabpfn），每个外层折在训练猪上拟合、测试折预测一次。
    特征组：img / R / R+img / R+cal / R+cal+img / R+prev+cal / R+prev+cal+img / R+prev+cal+B / R+prev+cal+img+B
    （R = 品系 + 胎次 + 妊娠日龄；prev = 上季背膘 + 是否有；cal = 卡尺；B = v4e 体尺 6 列；img = 高度图 CNN 分数）
  - 配对比较（同 A3）：Q0 R+img vs R；Q1 R+img vs R+cal；Q2 R+cal+img vs R+cal；Q3 R+prev+cal+img vs R+prev+cal；
    另报 R+prev+cal+img+B vs R+prev+cal+img（体尺在相机之上的增量）
  - 每个外层折完成后存档（内层分数），中断重跑跳过已完成的折

输出：runs/v4_hmap/nested_fusion/{log.txt, result.json, preds.csv, inner_scores.csv, _ckpt/}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4h_hmap_nested_fusion.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a3_foundation_regression as A3  # noqa: E402
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v4b_hmap_models import DM_HMAP, load_hmaps, predict, train_fold  # noqa: E402
from v4e_body_dims_fusion import pig_dims  # noqa: E402

V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
OUTER = os.path.join(HERE, 'runs', 'v4_hmap', 'hmap', 'test_preds.csv')
OUT = os.path.join(HERE, 'runs', 'v4_hmap', 'nested_fusion')


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, '_ckpt'), exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(V1, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(OUTER, encoding='utf-8-sig') as f:
        outer = {r['uid']: r for r in csv.DictReader(f)}
    with open(CAL, encoding='utf-8-sig') as f:
        cal_map = {r['uid']: fnum(r['caliper']) for r in csv.DictReader(f)}
    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows], np.float32)
    fold = np.array([int(r['fold']) for r in rows])
    assert all(int(outer[u]['fold']) == k for u, k in zip(uid, fold)), '外层折与 v4b 不一致'
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    cal = np.array([cal_map.get(u, np.nan) for u in uid])
    outer_score = np.array([float(outer[u][f'pred_s{a.seed}']) for u in uid])
    X, owner, _ = load_hmaps(uid, DM_HMAP, a.area_rel, a.len_rel)
    B, _ = pig_dims(uid)
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    log(f'[{time.strftime("%F %T")}] args={vars(a)} | pigs {len(y)}, frames {len(X)} (load {time.time() - t0:.0f}s)')

    dev = 'cuda'
    inner_by_fold = {}
    for k in sorted(set(fold)):
        ck = os.path.join(OUT, '_ckpt', f'outer{k}.npy')
        tr = np.where(fold != k)[0]
        if os.path.isfile(ck):
            inner_by_fold[k] = np.load(ck)
            log(f'  outer fold {k}: loaded checkpoint')
            continue
        sc = np.full(len(y), np.nan)
        isk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + k)
        for j, (itr_i, ite_i) in enumerate(isk.split(np.zeros(len(tr)), strat[tr], group[tr])):
            itr, ite = tr[itr_i], tr[ite_i]
            vs = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=200 + 10 * k + j)
            fit_i, val_i = next(vs.split(np.zeros(len(itr)), strat[itr], group[itr]))
            np.random.seed(a.seed + 1000 * k + j)
            torch.manual_seed(a.seed + 1000 * k + j)
            model = train_fold(X, owner, y, itr[fit_i], itr[val_i], a, dev, lambda m: None)
            pt = predict(model, X, owner, ite, dev)
            for i in ite:
                sc[i] = np.mean(pt[i])
            m = metrics(y[ite], sc[ite])
            log(f'  outer {k} inner {j + 1}: fit {len(fit_i)} / val {len(val_i)} / pred {len(ite)} | '
                f'MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} | {(time.time() - t0) / 60:.1f} min')
            del model
            torch.cuda.empty_cache()
        np.save(ck, sc)
        inner_by_fold[k] = sc
        m = metrics(y[tr], sc[tr])
        log(f'  outer fold {k}: inner-OOF on {len(tr)} training pigs MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f}')

    # 第二层 TabPFN
    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    C = cal[:, None]
    sets = {'img': lambda I: I, 'R': lambda I: R, 'R+img': lambda I: np.hstack([R, I]),
            'R+cal': lambda I: np.hstack([R, C]), 'R+cal+img': lambda I: np.hstack([R, C, I]),
            'R+prev+cal': lambda I: np.hstack([R, P, C]), 'R+prev+cal+img': lambda I: np.hstack([R, P, C, I]),
            'R+prev+cal+B': lambda I: np.hstack([R, P, C, B]),
            'R+prev+cal+img+B': lambda I: np.hstack([R, P, C, I, B])}
    preds = {k: np.zeros(len(y)) for k in sets}
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        I = np.where(fold == k, outer_score, inner_by_fold[k])[:, None]
        assert not np.isnan(I[tr]).any() and not np.isnan(I[te]).any()
        for name, fn in sets.items():
            Xs = fn(I)
            preds[name][te] = A3.fit_predict_tabpfn(Xs[tr], y[tr], Xs[te], 42 + k)
        log(f'  TabPFN outer fold {k} done | device {sorted(A3.DEVICE_USED)}')

    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    preds['img_raw (outer CNN, seed %d)' % a.seed] = outer_score
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel                         subset             n    MAE    RMSE   R2     CCC')
    for nm in preds:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:29s} {s:16s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:.3f}  {r["CCC"]:.3f}')
    pairs = [('R+img', 'R', 'Q0 imaging increment over records'),
             ('R+img', 'R+cal', 'Q1 camera vs caliper (with records)'),
             ('R+cal+img', 'R+cal', 'Q2 camera increment over records+caliper'),
             ('R+prev+cal+img', 'R+prev+cal', 'Q3 camera increment over records+prev+caliper'),
             ('R+prev+cal+img+B', 'R+prev+cal+img', 'body dims over records+prev+caliper+camera'),
             ('R+prev+cal+img', 'R+prev+cal+B', 'CNN score vs body dims (with records+prev+caliper)')]
    comp = {}
    log('\npaired comparisons (Δ = a − b): group bootstrap 95% CI; Wilcoxon on |error|')
    for a_, b_, lab in pairs:
        comp[lab] = {}
        for s in ('all', 'DAN_parity1', 'NAM_multiparous', 'no_prev_season'):
            m = subsets[s]
            bs = group_bootstrap(group[m], y[m], preds[a_][m], preds[b_][m])
            pw = float(wilcoxon(np.abs(preds[a_][m] - y[m]), np.abs(preds[b_][m] - y[m])).pvalue)
            comp[lab][s] = dict(**bs, wilcoxon_p=pw)
            log(f'   {lab:52s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), table=table, comparisons=comp), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(preds))
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], float(y[i])] + [round(float(preds[k][i]), 3) for k in preds])
    with open(os.path.join(OUT, 'inner_scores.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'fold'] + [f'inner_score_outer{k}' for k in sorted(inner_by_fold)])
        for i, u in enumerate(uid):
            w.writerow([u, fold[i]] + ['' if np.isnan(inner_by_fold[k][i]) else round(float(inner_by_fold[k][i]), 3)
                                       for k in sorted(inner_by_fold)])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
