# -*- coding: utf-8 -*-
"""
v4d_hmap_dk_external.py — 高度图 ResNet-18：丰美禾全体训练 → DK 外部测试（对照 V7 俯视深度图）

来源 / 背景
  DECISIONS 2026-10-13：标准化高度图把场内 R² 从 0.309 提到 0.378。V7（2026-10-11）：俯视深度图模型直接迁移到 DK
  高估 2–4 mm、R² < 0；每批补测 5 头 B 超校正截距后 R² 0.285。高度图消除了相机距离差异，检验跨场是否更稳。

设置（与 v7_dk_external.py 的 ResNet-18 部分一致，只换输入为高度图）
  - 丰美禾 663 头全部用于训练；验证集 = StratifiedGroupKFold(7, shuffle, random_state=seed) 第 1 份（按个体），选 epoch
  - 种子 42 / 43 / 44，另报 3 个模型平均；训练函数、帧质控、增广复用 v4b_hmap_models
  - DK 159 头只作测试（标签不参与训练与选模）
指标：MAE / RMSE / R² / CCC / Spearman / 平均偏差；分批次偏差与 Spearman；
  按批次截距校准（每批随机 k = 5 / 10 头 B 超，200 次，与 v7b 相同抽样方式）；与 V7 俯视 resnet18_mean3 并列

输出：runs/v4_hmap/dk_external/{log.txt, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4d_hmap_dk_external.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v4b_hmap_models import DM_HMAP, FOLDS_SRC, INDEX, load_hmaps, predict, train_fold  # noqa: E402

OUT = os.path.join(HERE, 'runs', 'v4_hmap', 'dk_external')
V7_PREDS = os.path.join(HERE, 'runs', 'v7_dk_external', 'preds.csv')


def full_metrics(y, p):
    m = metrics(y, p)
    m['bias'] = float(np.mean(p - y))
    return m


def batch_calibration(y, p, batch, ks=(5, 10), reps=200, seed=0):
    rng = np.random.default_rng(seed)
    bs = sorted(set(batch))
    out = {}
    for k in ks:
        vals = []
        for _ in range(reps):
            cal = np.concatenate([rng.choice(np.where(batch == b)[0], k, replace=False) for b in bs])
            rest = np.setdiff1d(np.arange(len(y)), cal)
            pc = p.copy()
            for b in bs:
                cb = cal[batch[cal] == b]
                pc[batch == b] += np.mean(y[cb] - p[cb])
            mm = metrics(y[rest], pc[rest])
            vals.append((mm['MAE'], mm['R2'], mm['Spearman']))
        v = np.array(vals)
        out[k] = {n: (float(v[:, i].mean()), float(v[:, i].std())) for i, n in enumerate(('MAE', 'R2', 'Spearman'))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(FOLDS_SRC, encoding='utf-8-sig') as f:
        prow = list(csv.DictReader(f))
    fu = [r['uid'] for r in prow]
    groups = np.array([r['group'] for r in prow])
    with open(INDEX, encoding='utf-8-sig') as f:
        idx = {r['uid']: r for r in csv.DictReader(f)}
    y = np.array([float(idx[u]['backfat']) for u in fu], np.float32)
    du = sorted(u for u, r in idx.items() if r['farm'] == 'DK')
    yd = np.array([float(idx[u]['backfat']) for u in du])
    bd = np.array([idx[u]['batch'] for u in du])
    X, owner, n1 = load_hmaps(fu, DM_HMAP, a.area_rel, a.len_rel)
    Xd, owner_d, n2 = load_hmaps(du, DM_HMAP, a.area_rel, a.len_rel)
    have_d = np.array([i in set(owner_d.tolist()) for i in range(len(du))])
    log(f'[{time.strftime("%F %T")}] args={vars(a)}')
    log(f'   FMH {len(fu)} pigs, frames {len(X)}/{n1}; DK {have_d.sum()}/{len(du)} pigs, frames {len(Xd)}/{n2} '
        f'(missing {[du[i] for i in np.where(~have_d)[0]]}) | y FMH {y.mean():.2f}±{y.std():.2f}, '
        f'DK {yd.mean():.2f}±{yd.std():.2f}')

    dev = 'cuda'
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    P = {}
    for s in a.seeds:
        np.random.seed(s)
        torch.manual_seed(s)
        tr, va = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=s)
                      .split(np.zeros(len(y)), strat, groups))
        model = train_fold(X, owner, y, tr, va, a, dev, lambda m: None)
        pt = predict(model, Xd, owner_d, np.where(have_d)[0], dev)
        P[f'hmap_s{s}'] = np.array([np.mean(pt[i]) if i in pt else np.nan for i in range(len(du))])
        m = full_metrics(yd[have_d], P[f'hmap_s{s}'][have_d])
        log(f'  seed {s}: train {len(tr)} / val {len(va)} | DK MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} '
            f'bias {m["bias"]:+.2f} Spearman {m["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')
        del model
        torch.cuda.empty_cache()
    P['hmap_mean3'] = np.mean([P[f'hmap_s{s}'] for s in a.seeds], 0)
    with open(V7_PREDS, encoding='utf-8-sig') as f:
        v7 = {r['uid']: r for r in csv.DictReader(f)}
    P['depthmap_resnet18_mean3 (V7)'] = np.array([float(v7[u]['resnet18_mean3']) for u in du])

    ok = have_d & ~np.isnan(P['hmap_mean3'])
    res = {'n': int(ok.sum()), 'external': {}, 'by_batch': {}, 'calibration': {}}
    log('\nDK external (n=%d)' % ok.sum())
    for k in ('hmap_mean3', 'depthmap_resnet18_mean3 (V7)'):
        p = P[k][ok]
        m = full_metrics(yd[ok], p)
        res['external'][k] = m
        log(f'  {k:30s} MAE {m["MAE"]:.3f} RMSE {m["RMSE"]:.3f} R2 {m["R2"]:.3f} CCC {m["CCC"]:.3f} '
            f'Spearman {m["Spearman"]:.3f} bias {m["bias"]:+.2f}')
        res['by_batch'][k] = {}
        msg = '      per batch:'
        for b in sorted(set(bd)):
            mk = bd[ok] == b
            bb = dict(n=int(mk.sum()), bias=float(np.mean(p[mk] - yd[ok][mk])),
                      spearman=float(spearmanr(p[mk], yd[ok][mk])[0]))
            res['by_batch'][k][b] = bb
            msg += f' {b} n={bb["n"]} bias {bb["bias"]:+.2f} ρ {bb["spearman"]:.3f} |'
        log(msg)
        c = batch_calibration(yd[ok], p, bd[ok])
        res['calibration'][k] = c
        log('      per-batch intercept calibration: ' + ' | '.join(
            f'k={kk}/batch MAE {v["MAE"][0]:.3f} R2 {v["R2"][0]:.3f}±{v["R2"][1]:.3f} ρ {v["Spearman"][0]:.3f}'
            for kk, v in c.items()))
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), **res), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'batch', 'y'] + list(P))
        for i, u in enumerate(du):
            w.writerow([u, bd[i], yd[i]] + ['' if np.isnan(P[k][i]) else round(float(P[k][i]), 3) for k in P])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
