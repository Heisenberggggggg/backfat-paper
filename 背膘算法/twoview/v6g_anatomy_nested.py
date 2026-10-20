# -*- coding: utf-8 -*-
"""
v6g_anatomy_nested.py — 新主模型（解剖引导高度图网络，v6f prof 变体）的嵌套无泄漏图像分数 + 逐帧折外预测

来源 / 背景
  DECISIONS 2026-10-15：本人选方案 B，主模型由高度图 ResNet-18（v4b）换成解剖引导网络（v6f prof）；
  下游的卡尺对比 / 组合（v6a）、重复性与区间（v6c）需用新模型重算。v6f 没有保存逐帧预测和嵌套分数，本脚本补齐。

做法（与 v4h 相同，只换图像模型）
  - 外层：同一套 5 折；每折用 v6f 的 train_eval（prof，种子 42 + 1000·k）在外层训练猪上训练（内层验证集
    StratifiedGroupKFold(7, seed=42+k) 第 1 份选 epoch，与 v6f 相同），取最佳 epoch 模型对测试猪逐帧预测
    → 测试猪图像分数（帧均值）+ 逐帧折外预测（供 ICC / 帧间 SD）
  - 内层：外层训练猪按个体分 5 份（StratifiedGroupKFold(5, shuffle, random_state=100+k)，同 v4h），每份由其余
    4 份训练的网络预测（再留 1/7 选 epoch，StratifiedGroupKFold(7, seed=200+10k+j)，同 v4h）→ 训练猪的折外分数
  - 共 5 + 25 = 30 个网络；每个外层折完成后存档，中断重跑跳过
  - 注意：外层模型是重新训练的（单种子），与 v6f prof 的 seed 42 结果不完全相同（GPU 训练不可逐位复现）；
    下游只用本脚本的分数，保证内部一致

输出：runs/v6_anatomy_nested/{log.txt, preds.csv, inner_scores.csv, frame_preds.csv, _ckpt/}
     preds.csv / inner_scores.csv 列名与 v4h 相同，frame_preds.csv 列名与 v4b 相同（seed = 42）
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6g_anatomy_nested.py
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v4b_hmap_models import DM_HMAP  # noqa: E402
from v6f_anatomy_net import get_profiles, load_hmaps_paths, predict_frames, read_setup, train_eval  # noqa: E402

OUT = os.path.join(HERE, 'runs', 'v6_anatomy_nested')
VARIANT = 'prof'


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
    dev = 'cuda'
    uids, groups, fold, y = read_setup()
    X, owner, paths = load_hmaps_paths(uids, DM_HMAP, a.area_rel, a.len_rel)
    P, V = get_profiles(X, paths, log)
    frames_of = {i: np.where(owner == i)[0] for i in range(len(uids))}
    frame_no = np.zeros(len(owner), int)
    for i, fr in frames_of.items():
        frame_no[fr] = np.arange(len(fr))
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    quiet = lambda s: None
    log(f'[{time.strftime("%F %T")}] v6g variant={VARIANT} args={vars(a)} | pigs {len(uids)}, frames {len(X)}')

    outer_score = np.full(len(uids), np.nan)
    inner_by_fold, frame_rows = {}, []
    for k in sorted(set(fold)):
        ck = os.path.join(OUT, '_ckpt', f'outer{k}.npz')
        te = np.where(fold == k)[0]
        trv = np.where(fold != k)[0]
        if os.path.isfile(ck):
            z = np.load(ck, allow_pickle=True)
            outer_score[te] = z['outer'][te]
            inner_by_fold[k] = z['inner']
            frame_rows += [tuple(r) for r in z['frames'].tolist()]
            log(f'  outer fold {k}: loaded checkpoint')
            continue
        # 外层模型（选 epoch 方式同 v6f：StratifiedGroupKFold(7, seed=42+k0) 第 1 份）
        k0 = k - 1
        tr_i, va_i = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k0)
                          .split(np.zeros(len(trv)), strat[trv], groups[trv]))
        np.random.seed(a.seed + 1000 * k)
        torch.manual_seed(a.seed + 1000 * k)
        hist, model = train_eval(VARIANT, X, P, V, owner, frames_of, y, trv[tr_i], trv[va_i], te, a, dev, quiet,
                                 return_best=True)
        fp = predict_frames(model, X, P, V, owner, frames_of, te, dev)
        del model
        torch.cuda.empty_cache()
        per = {}
        rows_k = []
        for fi, v in fp:
            i = int(owner[fi])
            per.setdefault(i, []).append(v)
            rows_k.append((uids[i], groups[i], k, a.seed, int(frame_no[fi]), float(y[i]), round(v, 3)))
        for i in te:
            outer_score[i] = np.mean(per[i])
        m = metrics(y[te], outer_score[te])
        ei = int(np.argmin([h['val_mae'] for h in hist]))
        log(f'  outer fold {k}: epoch {ei + 1} | test MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} | {(time.time() - t0) / 60:.1f} min')
        # 内层（同 v4h）
        sc = np.full(len(uids), np.nan)
        isk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=100 + k)
        for j, (itr_i, ite_i) in enumerate(isk.split(np.zeros(len(trv)), strat[trv], groups[trv])):
            itr, ite = trv[itr_i], trv[ite_i]
            vs = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=200 + 10 * k + j)
            fit_i, val_i = next(vs.split(np.zeros(len(itr)), strat[itr], groups[itr]))
            np.random.seed(a.seed + 1000 * k + j + 1)
            torch.manual_seed(a.seed + 1000 * k + j + 1)
            h = train_eval(VARIANT, X, P, V, owner, frames_of, y, itr[fit_i], itr[val_i], ite, a, dev, quiet)
            best = h[int(np.argmin([e['val_mae'] for e in h]))]['test_pred']
            for i in ite:
                sc[i] = best[i]
            mi = metrics(y[ite], sc[ite])
            log(f'    outer {k} inner {j + 1}: fit {len(fit_i)} / val {len(val_i)} / pred {len(ite)} | '
                f'MAE {mi["MAE"]:.3f} R2 {mi["R2"]:.3f} | {(time.time() - t0) / 60:.1f} min')
        inner_by_fold[k] = sc
        frame_rows += rows_k
        np.savez(ck, outer=outer_score, inner=sc, frames=np.array(rows_k, dtype=object))
        mi = metrics(y[trv], sc[trv])
        log(f'  outer fold {k}: inner-OOF on {len(trv)} training pigs MAE {mi["MAE"]:.3f} R2 {mi["R2"]:.3f}')

    g = metrics(y, outer_score)
    log(f'== outer image score (single network per fold, seed {a.seed}): MAE {g["MAE"]:.3f} R2 {g["R2"]:.3f} '
        f'| {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y', f'img_raw (outer CNN, seed {a.seed})'])
        for i, u in enumerate(uids):
            w.writerow([u, groups[i], fold[i], float(y[i]), round(float(outer_score[i]), 3)])
    with open(os.path.join(OUT, 'inner_scores.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        ks = sorted(inner_by_fold)
        w.writerow(['uid', 'fold'] + [f'inner_score_outer{k}' for k in ks])
        for i, u in enumerate(uids):
            w.writerow([u, fold[i]] + ['' if np.isnan(inner_by_fold[k][i]) else round(float(inner_by_fold[k][i]), 3)
                                       for k in ks])
    with open(os.path.join(OUT, 'frame_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'seed', 'frame_idx', 'y', 'pred'])
        w.writerows(frame_rows)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
