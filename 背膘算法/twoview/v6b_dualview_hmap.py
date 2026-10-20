# -*- coding: utf-8 -*-
"""
v6b_dualview_hmap.py — 高度图口径的多视角：俯视高度图 + 斜后视裁剪深度图 双分支 ResNet-18（N5）

来源 / 背景
  DECISIONS 2026-10-14：本文收窄为「背膘无接触式测定方法」，单 / 多视角是核心内容之一；本人确认补做与主模型
  同口径的多视角实验（09_研究成果总表 §4 N5）。原 V3（v3b，DECISIONS 2026-10-12（二））两支都是裁剪深度图
  （俯视 0.309、双分支 0.331，+0.022 n.s.），而主模型已换成俯视高度图（v4b，0.378）。斜后视没有高度图
  （v4a 只为俯视设计），因此斜后视分支仍用 v3a 的裁剪深度图。

协议（全部复用已有函数，只换俯视分支的输入）
  - 猪、外层折、内层验证集、种子与 v4b / v3b 完全相同：折取自 runs/framepreds_new_s128/test_preds.csv；
    内层验证集 StratifiedGroupKFold(7, shuffle, seed=42+k) 第 1 份；种子 42 / 43 / 44（np 与 torch 均按种子设定）
  - 俯视分支：v4b.load_hmaps（128 × 256，离地高度 / 1200 mm，帧质控 面积 ≥ 80%、体长 ≥ 85% 本猪中位）
  - 斜后视分支：v3b.load_frames（128 × 128，深度 300–2500 mm 归一化，帧质控 掩膜占画面 0.04–0.25）
  - 模型：v3b.DualNet（两支各自 ResNet-18，特征拼接 → 线性头；训练时视角随机丢弃 p = 0.2）
  - 帧配对：v3b.batches_dual（两相机不同步、不配准；训练时每个俯视帧随机配同猪一个斜后视帧，测试时按帧序循环配对）
  - 训练：v3b.train_fold（Huber δ=1、AdamW 3e-4、15 epoch 余弦、batch 64、两支一起水平翻转 p 0.5、按验证集猪级 MAE 选 epoch）；
    另对俯视高度图加垂直翻转 p 0.5（与 v4b 相同，高度图头尾方向不固定）
  - 猪级预测 = 帧均值；3 种子平均
  - 比较（同 663 头、同折）：双分支 vs 俯视高度图（v4b）；后期平均（俯视高度图与斜后视两个单视角模型的猪级预测取均值）vs 俯视高度图；
    斜后视（v3b sub）vs 俯视高度图；group bootstrap 95% CI + Wilcoxon；子集：全体、丹系头胎、新美系经产；
    另报两视角残差相关
  - 每个种子完成后存档，中断重跑跳过已完成种子；--compare-only 只重算比较

输出：runs/v6_dualview_hmap/{log.txt, result.json, test_preds.csv, frame_preds.csv, _seed_ckpt/}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6b_dualview_hmap.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import pearsonr, wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v3b_view_models import DM_SUB, DualNet, batches_dual, load_frames, predict, train_fold  # noqa: E402
from v4b_hmap_models import DM_HMAP, FOLDS_SRC, INDEX, load_hmaps  # noqa: E402

OUT = os.path.join(HERE, 'runs', 'v6_dualview_hmap')
HMAP_PREDS = os.path.join(HERE, 'runs', 'v4_hmap', 'hmap', 'test_preds.csv')
SUB_PREDS = os.path.join(HERE, 'runs', 'v3_views', 'sub_s128', 'test_preds.csv')
V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')


def read_pred_mean(path, uids):
    with open(path, encoding='utf-8-sig') as f:
        d = {r['uid']: r for r in csv.DictReader(f)}
    return np.array([float(d[u]['pred_mean']) if d[u]['pred_mean'] != '' else np.nan for u in uids])


def compare(uids, groups, y, dual, log):
    """双分支 / 后期平均 / 斜后视 vs 俯视高度图（同猪、同折）。"""
    with open(V1, encoding='utf-8-sig') as f:
        v1 = {r['uid']: r for r in csv.DictReader(f)}
    line = np.array([v1[u]['line'] for u in uids])
    parity = np.array([float(v1[u]['parity']) if v1[u]['parity'] != '' else np.nan for u in uids])
    hmap = read_pred_mean(HMAP_PREDS, uids)
    sub = read_pred_mean(SUB_PREDS, uids)
    late = (hmap + sub) / 2
    models = {'top height map (v4b)': hmap, 'rear-oblique cropped (v3b sub)': sub,
              'late fusion (mean of the two)': late, 'dual-branch (this script)': dual}
    ok = np.all([~np.isnan(p) for p in models.values()], 0)
    subsets = {'all': ok, 'DAN_parity1': ok & (line == 'DAN') & (parity == 1),
               'NAM_multiparous': ok & (line == 'NAM') & (parity >= 2)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in models.items()}
    log('\nmodel                            subset             n    MAE    RMSE   R2     CCC    Spearman')
    for nm in models:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:32s} {s:16s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:.3f}  {r["CCC"]:.3f}  '
                f'{r["Spearman"]:.3f}')
    ref = 'top height map (v4b)'
    comp = {}
    log('\npaired comparisons vs top height map (Δ = model − top height map): group bootstrap 95% CI; Wilcoxon on |error|')
    for nm in models:
        if nm == ref:
            continue
        comp[nm] = {}
        for s, m in subsets.items():
            bs = group_bootstrap(groups[m], y[m], models[nm][m], models[ref][m])
            pw = float(wilcoxon(np.abs(models[nm][m] - y[m]), np.abs(models[ref][m] - y[m])).pvalue)
            comp[nm][s] = dict(**bs, wilcoxon_p=pw)
            log(f'   {nm:32s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')
    rr = {'residual r (top hmap, rear-oblique)': float(pearsonr(hmap[ok] - y[ok], sub[ok] - y[ok])[0]),
          'prediction r (top hmap, rear-oblique)': float(pearsonr(hmap[ok], sub[ok])[0]),
          'residual r (top hmap, dual)': float(pearsonr(hmap[ok] - y[ok], dual[ok] - y[ok])[0])}
    for k, v in rr.items():
        log(f'   {k}: {v:.3f}')
    return dict(table=table, comparisons=comp, correlations=rr, n=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--p-drop', type=float, default=0.2)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    ap.add_argument('--sub-qc', type=float, nargs=2, default=[0.04, 0.25])
    ap.add_argument('--compare-only', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(os.path.join(OUT, '_seed_ckpt'), exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(FOLDS_SRC, encoding='utf-8-sig') as f:
        prow = list(csv.DictReader(f))
    uids = [r['uid'] for r in prow]
    assert uids == sorted(uids)
    groups = np.array([r['group'] for r in prow])
    fold = np.array([int(r['fold']) for r in prow])
    with open(INDEX, encoding='utf-8-sig') as f:
        bf = {r['uid']: float(r['backfat']) for r in csv.DictReader(f)}
    y = np.array([bf[u] for u in uids], np.float32)
    assert np.allclose(y, [float(r['y']) for r in prow])
    log(f'[{time.strftime("%F %T")}] v6b args={vars(a)}')

    if a.compare_only:
        with open(os.path.join(OUT, 'test_preds.csv'), encoding='utf-8-sig') as f:
            d = {r['uid']: r for r in csv.DictReader(f)}
        pm = np.array([float(d[u]['pred_mean']) if d[u]['pred_mean'] != '' else np.nan for u in uids])
        res = compare(uids, groups, y, pm, log)
        with open(os.path.join(OUT, 'result_compare.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=1, default=float)
        return

    XA, own_a, n_all = load_hmaps(uids, DM_HMAP, a.area_rel, a.len_rel)
    XB, own_b = load_frames(DM_SUB, 'uid', uids, 128, qc=tuple(a.sub_qc))
    have = set(own_a.tolist()) & set(own_b.tolist())
    have_mask = np.array([i in have for i in range(len(uids))])
    log(f'   top hmap frames {len(XA)} (of {n_all} ok) {XA.shape[1:]}; rear-oblique frames {len(XB)} {XB.shape[1:]}; '
        f'pigs with both views {have_mask.sum()}/{len(uids)} (load {time.time() - t0:.0f}s)')

    def batch_fn(pigs, bs, shuffle):
        for (xa, xb), ow in batches_dual(XA, XB, own_a, own_b, pigs, bs, shuffle):
            if shuffle and np.random.rand() < 0.5:  # 俯视高度图垂直翻转（v4b 相同）；水平翻转由 train_fold 对两支一起做
                xa = torch.flip(xa, dims=[2])
            yield (xa, xb), ow

    make_model = lambda b: DualNet(b, a.p_drop)
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    preds = {s: np.full(len(uids), np.nan) for s in a.seeds}
    frame_rows = []
    for s in a.seeds:
        ck = os.path.join(OUT, '_seed_ckpt', f'seed{s}.npz')
        if os.path.isfile(ck):
            z = np.load(ck, allow_pickle=True)
            preds[s] = z['pred']
            frame_rows += [tuple(r) for r in z['frames'].tolist()]
            log(f'  seed {s}: loaded checkpoint {ck}')
            continue
        np.random.seed(s)
        torch.manual_seed(s)
        for k in range(5):
            trv, te_all = np.where(fold != k + 1)[0], np.where(fold == k + 1)[0]
            inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
            tr, va, te = [x[have_mask[x]] for x in (trv[tr_i], trv[va_i], te_all)]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
            model = train_fold(make_model, batch_fn, y, tr, va, a, dev, log)
            pt = predict(model, batch_fn, te, dev)
            for i in te:
                preds[s][i] = np.mean(pt[i])
                frame_rows += [(uids[i], groups[i], k + 1, s, j, float(y[i]), round(v, 3)) for j, v in enumerate(pt[i])]
            m = metrics(y[te], preds[s][te])
            log(f'  seed {s} fold {k + 1} TEST MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f}')
            del model
            torch.cuda.empty_cache()
        np.savez(ck, pred=preds[s], frames=np.array([r for r in frame_rows if r[3] == s], dtype=object))
        ok = ~np.isnan(preds[s])
        g = metrics(y[ok], preds[s][ok])
        log(f'== seed {s}: n={ok.sum()} MAE {g["MAE"]:.3f} R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} '
            f'Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')

    pm = np.mean([preds[s] for s in a.seeds], 0)
    ok = ~np.isnan(pm)
    g = metrics(y[ok], pm[ok])
    log(f'== dual (top hmap + rear-oblique) seed-mean: n={ok.sum()} MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} '
        f'R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')
    res = {'args': vars(a), 'n': int(ok.sum()),
           'per_seed': {s: metrics(y[~np.isnan(preds[s])], preds[s][~np.isnan(preds[s])]) for s in a.seeds},
           'seed_mean': g}
    res['compare'] = compare(uids, groups, y, pm, log)
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + [f'pred_s{s}' for s in a.seeds] + ['pred_mean'])
        for i, u in enumerate(uids):
            w.writerow([u, groups[i], fold[i], float(y[i])]
                       + ['' if np.isnan(preds[s][i]) else round(preds[s][i], 3) for s in a.seeds]
                       + ['' if np.isnan(pm[i]) else round(pm[i], 3)])
    with open(os.path.join(OUT, 'frame_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'seed', 'frame_idx', 'y', 'pred'])
        w.writerows(frame_rows)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
