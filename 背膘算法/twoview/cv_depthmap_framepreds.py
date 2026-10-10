# -*- coding: utf-8 -*-
"""
cv_depthmap_framepreds.py — 与 cv_depthmap_labelfix.py 完全相同的训练，额外保存「逐帧」测试预测（供 A2 重复性分析）

来源 / 背景
  A2（图像表型重复性）需要每头猪每一帧的预测值，用于计算帧级 ICC、Spearman–Brown 所需帧数、
  以及「随机取 k 帧求均值」时误差随 k 的变化。9/28 的 labelfix_new_s128 只保存了猪级均值，也未保存权重，
  因此用同一设置重跑一次。

与旧脚本的区别
  - 直接 import cv_depthmap_labelfix 的 load_data / train_fold / metrics（数据、模型、选模协议完全一致；
    seed=42 → 折划分与旧运行相同；GPU 训练非确定性，猪级数值与旧运行可能有小差异）
  - 只跑新标签（--labels new）
  - 新增输出 frame_preds.csv：uid, group, fold, frame_idx, y, pred（逐帧）
  - 输出目录 runs/framepreds_new_s128/（不覆盖 labelfix_new_s128）

用法
  D:\\Anaconda\\envs\\pytorch\\python.exe cv_depthmap_framepreds.py
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import load_data, train_fold, metrics  # noqa: E402


def predict_frames(model, X, owner, pig_ids, dev, bs=256):
    """返回 (owner_of_each_frame, frame_pred)，只含 pig_ids 的帧，帧顺序与 X 中一致。"""
    model.eval()
    sel = np.where(np.isin(owner, pig_ids))[0]
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(sel), bs):
            xb = torch.from_numpy(X[sel[k:k + bs]].astype(np.float32))[:, None].to(dev)
            out.append(model(xb).float().cpu().numpy())
    return owner[sel], np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    out_dir = os.path.join(HERE, 'runs', f'framepreds_new_s{a.size}')
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    pigs, X, owner, y = load_data('new', a.size)
    groups = np.array([p['group'] for p in pigs])
    log(f'[{time.strftime("%F %T")}] pigs={len(pigs)} groups={len(set(groups))} frames={len(X)} '
        f'(load {time.time() - t0:.0f}s)')

    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    all_p, all_base, fold_of = np.zeros(len(pigs)), np.zeros(len(pigs)), np.zeros(len(pigs), int)
    frame_rows = []
    for k, (trv, te) in enumerate(sgkf.split(np.zeros(len(pigs)), strat, groups)):
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed + k)
        tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
        tr, va = trv[tr_i], trv[va_i]
        log(f'  fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
        model = train_fold(X, owner, y, tr, va, a, dev, log)
        own, fp = predict_frames(model, X, owner, te, dev)
        for i in te:
            pi = fp[own == i]
            all_p[i] = pi.mean()
            for j, v in enumerate(pi):
                frame_rows.append((pigs[i]['uid'], pigs[i]['group'], k + 1, j, float(y[i]), round(float(v), 3)))
        all_base[te], fold_of[te] = y[tr].mean(), k + 1
        m = metrics(y[te], all_p[te])
        log(f'  fold {k + 1} TEST  model MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} CCC {m["CCC"]:.3f}')
        del model
        torch.cuda.empty_cache()

    g, gb = metrics(y, all_p), metrics(y, all_base)
    log(f'== pooled: MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} '
        f'Spearman {g["Spearman"]:.3f} | baseline MAE {gb["MAE"]:.3f} | {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), pooled=g, baseline=gb), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(out_dir, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y', 'pred', 'baseline'])
        for p_, k_, yy, pr, b in zip(pigs, fold_of, y, all_p, all_base):
            w.writerow([p_['uid'], p_['group'], k_, float(yy), round(pr, 3), round(b, 3)])
    with open(os.path.join(out_dir, 'frame_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'frame_idx', 'y', 'pred'])
        w.writerows(frame_rows)
    log(f'saved {len(frame_rows)} frame predictions')


if __name__ == '__main__':
    main()
