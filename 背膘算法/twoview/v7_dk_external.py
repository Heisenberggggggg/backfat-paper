# -*- coding: utf-8 -*-
"""
v7_dk_external.py — V7：丰美禾训练 → 省农科院 DK 外部测试（同为丹系，跨场、跨相机）

来源 / 背景
  05 §5 / 06 §5.1 第 7 项；10/15 检查点的最后一项。DK 159 头无胎次、妊娠日龄、卡尺 → 只能检验「相机单独」跨场泛化。
  已知：DK 背膘均值 14.55 mm，丰美禾 17.9 mm（差约 3.3 mm）；DK 相机到猪背中位深度 799 mm，丰美禾 845 mm。

模型（全部只用丰美禾 663 头训练 / 拟合，DK 标签不参与任何训练与选模）
  1. ResNet-18@128：与 cv_depthmap_labelfix.py 完全相同的网络、预处理与训练函数（import 复用）；
     在丰美禾全体上按 group 留约 1/7 作验证集选 epoch；seed 42/43/44 各训练一次，另报 3 个模型的平均
  2. 冻结 DINOv3 ViT-B/16（relh、normal）+ RidgeCV（丰美禾全体，标准化后全维，alpha 内部 5 折）
  3. 丰美禾均值（常数）
指标（DK 159 头）：MAE / RMSE / R² / CCC / Spearman / Pearson / 平均偏差（预测 − 真值）
少样本校准（新猪场补测少量 B 超）：随机抽 k = 10 / 20 / 40 头 DK 猪，
  (a) 只校正截距：p + mean(y_k − p_k)；(b) 截距 + 斜率：y ~ a + b·p 在 k 头上拟合；在其余猪上评估，重复 200 次
场内参照（非外测）：DK 内部 5 折交叉验证，DINOv3-relh + RidgeCV —— 只用 DK 自己的数据能达到的水平

与旧脚本的区别：新脚本，只读已有数据 / 特征；输出 runs/v7_dk_external/。

用法
  D:\\Anaconda\\envs\\pytorch\\python.exe v7_dk_external.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import (DM_DIR, NORM_MAX, NORM_MIN, load_data, metrics,  # noqa: E402
                                  predict_pigs, train_fold)

FEAT = os.path.join(HERE, 'runs', 'a3_foundation')
OUT = os.path.join(HERE, 'runs', 'v7_dk_external')
PRE = {'nky1': 'dk_nkyfirst_', 'nky2': 'dk_nkysecond_', 'nky3': 'dk_nkythird_'}


def load_dk(size):
    idx = {}
    with open(os.path.join(HERE, 'index_v2.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['farm'] == 'DK':
                idx[r['uid']] = r
    frames = {}
    with open(os.path.join(DM_DIR, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = r['pig_id']
            if p.split('_')[0] in PRE:
                frames.setdefault(PRE[p.split('_')[0]] + p.split('_', 1)[1], []).append(r['frame_path'])
    uids = sorted(idx)
    X, owner = [], []
    for i, u in enumerate(uids):
        for fp in frames[u]:
            d = np.load(os.path.join(DM_DIR, fp)).astype(np.float32)
            d = np.clip((d - NORM_MIN) / (NORM_MAX - NORM_MIN), 0, 1)
            d = torch.from_numpy(d)[None, None]
            X.append(F.interpolate(d, size=(size, size), mode='area')[0, 0].numpy().astype(np.float16))
            owner.append(i)
    y = np.array([float(idx[u]['backfat']) for u in uids])
    batch = [idx[u]['batch'] for u in uids]
    return uids, np.stack(X), np.array(owner), y, batch


def full_metrics(y, p):
    m = metrics(y, p)
    m['Pearson'] = float(pearsonr(y, p)[0]) if np.std(p) > 0 else float('nan')
    m['bias'] = float(np.mean(p - y))
    return m


def calibration(y, p, ks=(10, 20, 40), reps=200, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        res = {'intercept': [], 'linear': []}
        for _ in range(reps):
            cal = rng.choice(len(y), k, replace=False)
            rest = np.setdiff1d(np.arange(len(y)), cal)
            pi = p[rest] + np.mean(y[cal] - p[cal])
            b, a0 = np.polyfit(p[cal], y[cal], 1)
            pl = a0 + b * p[rest]
            for name, pp in (('intercept', pi), ('linear', pl)):
                mm = metrics(y[rest], pp)
                res[name].append((mm['MAE'], mm['R2'], mm['CCC']))
        out[k] = {name: dict(zip(('MAE', 'R2', 'CCC'),
                                 [(float(np.mean(v)), float(np.std(v))) for v in np.array(vals).T]))
                  for name, vals in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seeds', default='42,43,44')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    pigs, X, owner, y = load_data('new', a.size)
    groups = np.array([p['group'] for p in pigs])
    fmh_uids = [p['uid'] for p in pigs]
    dk_uids, Xd, owner_d, yd, batch_d = load_dk(a.size)
    log(f'[{time.strftime("%F %T")}] FMH pigs={len(pigs)} frames={len(X)} y {y.mean():.2f}±{y.std():.2f} | '
        f'DK pigs={len(dk_uids)} frames={len(Xd)} y {yd.mean():.2f}±{yd.std():.2f}')
    preds = {'fmh_mean': np.full(len(yd), y.mean())}

    # ---- 1. ResNet-18（与 cv_depthmap_labelfix 相同的训练函数）----
    dev = 'cuda'
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    rn = []
    for s in [int(v) for v in a.seeds.split(',')]:
        np.random.seed(s)
        torch.manual_seed(s)
        tr_i, va_i = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=s)
                          .split(np.zeros(len(y)), strat, groups))
        log(f'  ResNet-18 seed {s}: train {len(tr_i)} / val {len(va_i)} FMH pigs')
        model = train_fold(X, owner, y.astype(np.float32), tr_i, va_i, a, dev, lambda m: None)
        p = predict_pigs(model, Xd, owner_d, np.arange(len(dk_uids)), dev)
        preds[f'resnet18_s{s}'] = p
        rn.append(p)
        m = full_metrics(yd, p)
        log(f'    DK: MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} bias {m["bias"]:+.2f} Spearman {m["Spearman"]:.3f}')
        del model
        torch.cuda.empty_cache()
    preds['resnet18_mean3'] = np.mean(rn, 0)

    # ---- 2. 冻结 DINOv3 + RidgeCV ----
    for enc in ('relh', 'normal'):
        fz = np.load(os.path.join(FEAT, f'feat_dinov3-vitb16_{enc}.npz'))
        fd = np.load(os.path.join(FEAT, f'feat_dinov3-vitb16_{enc}_dk.npz'))
        assert list(fd['uids']) == dk_uids
        order = {u: i for i, u in enumerate(fz['uids'])}
        Ff = fz['pig_feat'][[order[u] for u in fmh_uids]].astype(np.float64)
        sc = StandardScaler().fit(Ff)
        rg = RidgeCV(alphas=np.logspace(-1, 4, 20), cv=5).fit(sc.transform(Ff), y)
        preds[f'dinov3_{enc}_ridge'] = rg.predict(sc.transform(fd['pig_feat'].astype(np.float64)))
        log(f'  DINOv3-{enc} ridge alpha {rg.alpha_:.1f}')

    # ---- 汇总 ----
    table = {k: full_metrics(yd, p) for k, p in preds.items()}
    log('\nDK external test (n=%d)      MAE   RMSE     R2    CCC  Spear  Pears   bias' % len(yd))
    for k, m in table.items():
        log(f'{k:24s} {m["MAE"]:.3f}  {m["RMSE"]:.3f}  {m["R2"]:6.3f}  {m["CCC"]:.3f}  {m["Spearman"]:.3f}  '
            f'{m["Pearson"]:.3f}  {m["bias"]:+.2f}')
    by_batch = {}
    for b in sorted(set(batch_d)):
        mk = np.array(batch_d) == b
        by_batch[b] = {k: full_metrics(yd[mk], preds[k][mk]) for k in ('resnet18_mean3', 'dinov3_relh_ridge')}
        log(f'  [{b}] n={mk.sum()}  ' + '  '.join(f'{k}: Spearman {v["Spearman"]:.3f} bias {v["bias"]:+.2f}'
                                                   for k, v in by_batch[b].items()))

    cal = {k: calibration(yd, preds[k]) for k in ('resnet18_mean3', 'dinov3_relh_ridge', 'dinov3_normal_ridge')}
    log('\nfew-shot calibration on DK (mean ± SD over 200 random draws; evaluated on remaining pigs)')
    for k, d in cal.items():
        for kk, v in d.items():
            log(f'  {k:22s} k={kk:2d}  intercept: MAE {v["intercept"]["MAE"][0]:.3f}±{v["intercept"]["MAE"][1]:.3f} '
                f'R2 {v["intercept"]["R2"][0]:.3f}  | linear: MAE {v["linear"]["MAE"][0]:.3f}±{v["linear"]["MAE"][1]:.3f} '
                f'R2 {v["linear"]["R2"][0]:.3f}')

    # ---- 场内参照：DK 内部 5 折 ----
    fd = np.load(os.path.join(FEAT, 'feat_dinov3-vitb16_relh_dk.npz'))['pig_feat'].astype(np.float64)
    oof, base = np.zeros(len(yd)), np.zeros(len(yd))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(fd):
        sc = StandardScaler().fit(fd[tr])
        oof[te] = RidgeCV(alphas=np.logspace(-1, 4, 20), cv=5).fit(sc.transform(fd[tr]), yd[tr]).predict(
            sc.transform(fd[te]))
        base[te] = yd[tr].mean()
    within = {'dinov3_relh_ridge_withinDK_cv': full_metrics(yd, oof), 'withinDK_mean': full_metrics(yd, base)}
    for k, m in within.items():
        log(f'  reference {k:30s} MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} Spearman {m["Spearman"]:.3f}')

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), external=table, by_batch=by_batch, calibration=cal, within_dk=within),
                  f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        keys = list(preds)
        w.writerow(['uid', 'batch', 'y'] + keys + ['dinov3_relh_withinDK_cv'])
        for i, u in enumerate(dk_uids):
            w.writerow([u, batch_d[i], yd[i]] + [round(float(preds[k][i]), 3) for k in keys] + [round(oof[i], 3)])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
