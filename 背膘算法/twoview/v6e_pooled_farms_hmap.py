# -*- coding: utf-8 -*-
"""
v6e_pooled_farms_hmap.py — A 场（丰美禾）+ B 场（省农科院 DK）合并训练高度图 ResNet-18：加数据能否提高精度

来源 / 背景
  本人 2026-10-14 更正：「不加多个场」指不做跨场横向对比（外测 / 迁移），不是不用 B 场数据；问 B 场数据加入训练
  能否提高模型精度与泛化。本脚本只回答「加 B 场数据训练」这一问题，不做跨场迁移评估。

设计
  - A 场 663 头：外层折、内层验证集与 v4b 完全相同（折取自 framepreds_new_s128；StratifiedGroupKFold(7, seed=42+k)
    在 A 场训练猪上取第 1 份）→ A 场测试猪与 v4b 一一对应，可做配对比较
  - B 场 159 头：另分 5 折（StratifiedGroupKFold(5, shuffle, random_state=2026)，按 B 场背膘五分位分层，每头一组），
    第 k 折与 A 场第 k 折一起作测试；其余 B 场猪加入训练，其中约 1/7（StratifiedGroupKFold(7, seed=42+k) 第 1 份）
    加入验证集 → 验证集 = A 场验证猪 ∪ B 场验证猪，按合并验证集猪级 MAE 选 epoch
  - 网络、帧质控、增广、种子（42 / 43 / 44）全部复用 v4b（train_fold / predict / load_hmaps）；不输入场别
  - 评估：A 场测试猪 vs v4b（group bootstrap + Wilcoxon；全体、丹系头胎、新美系经产）；B 场测试猪的合并交叉验证指标
    （R²、MAE、平均偏差、分批次）。每个种子完成后存档，中断重跑跳过已完成种子

输出：runs/v6_pooled_farms/{log.txt, result.json, test_preds.csv, _seed_ckpt/}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6e_pooled_farms_hmap.py
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
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v4b_hmap_models import DM_HMAP, FOLDS_SRC, INDEX, load_hmaps, predict, train_fold  # noqa: E402

OUT = os.path.join(HERE, 'runs', 'v6_pooled_farms')
V4B = os.path.join(HERE, 'runs', 'v4_hmap', 'hmap', 'test_preds.csv')
V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(os.path.join(OUT, '_seed_ckpt'), exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(FOLDS_SRC, encoding='utf-8-sig') as f:
        prow = list(csv.DictReader(f))
    fu = [r['uid'] for r in prow]
    assert fu == sorted(fu)
    with open(INDEX, encoding='utf-8-sig') as f:
        idx = {r['uid']: r for r in csv.DictReader(f)}
    du = sorted(u for u, r in idx.items() if r['farm'] == 'DK')
    uids = fu + du
    nA, nB = len(fu), len(du)
    farm = np.array(['A'] * nA + ['B'] * nB)
    y = np.array([float(idx[u]['backfat']) for u in uids], np.float32)
    assert np.allclose(y[:nA], [float(r['y']) for r in prow])
    groups = np.array([r['group'] for r in prow] + du)
    batch = np.array([''] * nA + [idx[u]['batch'] for u in du])
    foldA = np.array([int(r['fold']) for r in prow])
    yB = y[nA:]
    stratB = np.digitize(yB, np.quantile(yB, [0.2, 0.4, 0.6, 0.8]))
    foldB = np.zeros(nB, int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
                                .split(np.zeros(nB), stratB, du)):
        foldB[te] = k + 1
    fold = np.concatenate([foldA, foldB])
    X, owner, n_ok = load_hmaps(uids, DM_HMAP, a.area_rel, a.len_rel)
    have = np.array([i in set(owner.tolist()) for i in range(len(uids))])
    log(f'[{time.strftime("%F %T")}] v6e args={vars(a)}')
    log(f'   A {nA} pigs (y {y[:nA].mean():.2f}±{y[:nA].std():.2f}), B {nB} pigs (y {yB.mean():.2f}±{yB.std():.2f}); '
        f'frames after QC {len(X)}/{n_ok}; pigs with frames A {have[:nA].sum()} B {have[nA:].sum()}; '
        f'B per fold {np.bincount(foldB)[1:].tolist()} (load {time.time() - t0:.0f}s)')

    stratA = np.digitize(y[:nA], np.quantile(y[:nA], [0.2, 0.4, 0.6, 0.8]))
    preds = {s: np.full(len(uids), np.nan) for s in a.seeds}
    for s in a.seeds:
        ck = os.path.join(OUT, '_seed_ckpt', f'seed{s}.npy')
        if os.path.isfile(ck):
            preds[s] = np.load(ck)
            log(f'  seed {s}: loaded checkpoint')
            continue
        np.random.seed(s)
        torch.manual_seed(s)
        for k in range(5):
            # A 场：与 v4b 完全相同的训练 / 验证划分
            trvA, teA = np.where(foldA != k + 1)[0], np.where(foldA == k + 1)[0]
            iA = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            trA_i, vaA_i = next(iA.split(np.zeros(len(trvA)), stratA[trvA], groups[:nA][trvA]))
            # B 场：其余折的 B 猪加入训练，约 1/7 加入验证
            trvB, teB = np.where(foldB != k + 1)[0], np.where(foldB == k + 1)[0]
            iB = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            trB_i, vaB_i = next(iB.split(np.zeros(len(trvB)), stratB[trvB], np.array(du)[trvB]))
            tr = np.concatenate([trvA[trA_i], nA + trvB[trB_i]])
            va = np.concatenate([trvA[vaA_i], nA + trvB[vaB_i]])
            te = np.concatenate([teA, nA + teB])
            tr, va, te = [x[have[x]] for x in (tr, va, te)]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} (B {np.sum(tr >= nA)}) / val {len(va)} '
                f'(B {np.sum(va >= nA)}) / test {len(te)} (B {np.sum(te >= nA)})')
            model = train_fold(X, owner, y, tr, va, a, dev, lambda m: None)
            pt = predict(model, X, owner, te, dev)
            for i in te:
                preds[s][i] = np.mean(pt[i])
            mA = metrics(y[te[te < nA]], preds[s][te[te < nA]])
            mB = metrics(y[te[te >= nA]], preds[s][te[te >= nA]])
            log(f'  seed {s} fold {k + 1} TEST A MAE {mA["MAE"]:.3f} R2 {mA["R2"]:.3f} | '
                f'B MAE {mB["MAE"]:.3f} R2 {mB["R2"]:.3f} | {(time.time() - t0) / 60:.1f} min')
            del model
            torch.cuda.empty_cache()
        np.save(ck, preds[s])
        for fm, sl in (('A', slice(0, nA)), ('B', slice(nA, None))):
            p_, y_ = preds[s][sl], y[sl]
            ok = ~np.isnan(p_)
            g = metrics(y_[ok], p_[ok])
            log(f'== seed {s} farm {fm}: n={ok.sum()} MAE {g["MAE"]:.3f} R2 {g["R2"]:.3f}')

    pm = np.mean([preds[s] for s in a.seeds], 0)
    with open(V4B, encoding='utf-8-sig') as f:
        v4b = {r['uid']: float(r['pred_mean']) for r in csv.DictReader(f)}
    with open(V1, encoding='utf-8-sig') as f:
        v1 = {r['uid']: r for r in csv.DictReader(f)}
    pA, yA, gA = pm[:nA], y[:nA], groups[:nA]
    base = np.array([v4b[u] for u in fu])
    line = np.array([v1[u]['line'] for u in fu])
    parity = np.array([float(v1[u]['parity']) if v1[u]['parity'] != '' else np.nan for u in fu])
    subsets = {'all': np.ones(nA, bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2)}
    res = {'args': vars(a), 'farm_A': {}, 'farm_B': {}}
    log('\n== Farm A test pigs: A-only training (v4b) vs A + B pooled training (seed mean, same A folds)')
    for s_, m in subsets.items():
        ra, rb = metrics(yA[m], base[m]), metrics(yA[m], pA[m])
        bs = group_bootstrap(gA[m], yA[m], pA[m], base[m])
        pw = float(wilcoxon(np.abs(pA[m] - yA[m]), np.abs(base[m] - yA[m])).pvalue)
        res['farm_A'][s_] = dict(A_only=ra, pooled=rb, **bs, wilcoxon_p=pw)
        log(f'   [{s_:15s}] n={m.sum()} A-only R2 {ra["R2"]:.3f} MAE {ra["MAE"]:.3f} | pooled R2 {rb["R2"]:.3f} '
            f'MAE {rb["MAE"]:.3f} | ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f}) '
            f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f}) p={pw:.3g}')
    pB, okB = pm[nA:], ~np.isnan(pm[nA:])
    gb = metrics(yB[okB], pB[okB])
    res['farm_B']['pooled_cv'] = dict(**gb, bias=float(np.mean(pB[okB] - yB[okB])))
    log(f'\n== Farm B test pigs (pooled cross-validation, seed mean): n={okB.sum()} MAE {gb["MAE"]:.3f} '
        f'R2 {gb["R2"]:.3f} CCC {gb["CCC"]:.3f} Spearman {gb["Spearman"]:.3f} bias {np.mean(pB[okB] - yB[okB]):+.2f}')
    bB = batch[nA:]
    for b in sorted(set(bB)):
        m = okB & (bB == b)
        gm = metrics(yB[m], pB[m])
        res['farm_B'][f'batch_{b}'] = dict(**gm, bias=float(np.mean(pB[m] - yB[m])))
        log(f'   batch {b}: n={m.sum()} MAE {gm["MAE"]:.3f} R2 {gm["R2"]:.3f} bias {np.mean(pB[m] - yB[m]):+.2f}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'farm', 'group', 'fold', 'batch', 'y'] + [f'pred_s{s}' for s in a.seeds] + ['pred_mean'])
        for i, u in enumerate(uids):
            w.writerow([u, farm[i], groups[i], fold[i], batch[i], float(y[i])]
                       + ['' if np.isnan(preds[s][i]) else round(float(preds[s][i]), 3) for s in a.seeds]
                       + ['' if np.isnan(pm[i]) else round(float(pm[i]), 3)])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
