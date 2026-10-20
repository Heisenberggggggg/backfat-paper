# -*- coding: utf-8 -*-
"""
cv_depthmap_labelfix.py — 标签纠正前后对比（丰美禾，俯视掩膜深度图，ResNet-18，猪级 5 折）

来源 / 背景
  build_index.py v2 发现：旧脚本对丰美禾用「耳号是目录名子串」全表匹配，未区分 8 月 / 10–11 月两季，
  约 1/3 的猪标签配错（|差| 均值 ≈ 4.4 mm）。本脚本在「完全相同的猪、相同的折、相同的模型」下
  只切换标签来源（--labels old|new），量化标签错配对结果的影响。

与旧脚本（网络/depthmap_cnn/train_depthmap.py）的区别
  - 复用其数据（yolo+FastSAM/depthmap_dataset/*.npy，256×256 掩膜深度 mm）与归一化（300–2500 mm）
  - 划分：StratifiedGroupKFold，按 group_id（同一头猪跨日期重复采集归为一组）→ 防泄漏
  - 选模：每折训练集内再留约 1/7 组做验证集选 epoch，测试折只用一次（E0 统一协议）
  - 损失 Huber(δ=1 mm)；猪级预测 = 该猪全部帧预测均值
  - 每折报「训练集均值」基线；汇总 MAE / RMSE / R² / Lin CCC / Spearman
  - 为适配 RTX 4060 8GB：输入下采样到 --size（默认 128），全部帧以 float16 常驻内存

用法
  python cv_depthmap_labelfix.py --labels new
  python cv_depthmap_labelfix.py --labels old
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import models

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
DM_DIR = os.path.join(HERE, '..', 'yolo+FastSAM', 'depthmap_dataset')
INDEX = INDEX_PATH
NORM_MIN, NORM_MAX = 300.0, 2500.0
BF_MIN, BF_MAX = 7.0, 32.0


def load_data(labels, size):
    idx = {}
    with open(INDEX, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['farm'] == 'FMH':
                idx[r['uid']] = r
    old, frames = {}, {}
    with open(os.path.join(DM_DIR, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            old[r['pig_id']] = float(r['backfat'])
            frames.setdefault(r['pig_id'], []).append(r['frame_path'])
    pigs = []
    for uid in sorted(set(idx) & set(old)):
        new_bf = float(idx[uid]['backfat'])
        if not (BF_MIN <= new_bf <= BF_MAX and BF_MIN <= old[uid] <= BF_MAX):
            continue
        if int(idx[uid]['label_conflict']):
            continue
        pigs.append(dict(uid=uid, group=idx[uid]['group_id'], new=new_bf, old=old[uid],
                         match_level=idx[uid].get('match_level', '')))
    X, owner = [], []
    for i, p in enumerate(pigs):
        for fp in frames[p['uid']]:
            d = np.load(os.path.join(DM_DIR, fp)).astype(np.float32)
            d = np.clip((d - NORM_MIN) / (NORM_MAX - NORM_MIN), 0, 1)
            d = torch.from_numpy(d)[None, None]
            X.append(F.interpolate(d, size=(size, size), mode='area')[0, 0].numpy().astype(np.float16))
            owner.append(i)
    y = np.array([p[labels] for p in pigs], np.float32)
    return pigs, np.stack(X), np.array(owner), y


class Net(nn.Module):
    def __init__(self, init_bias):
        super().__init__()
        r = models.resnet18(weights='IMAGENET1K_V1')
        w = r.conv1.weight.mean(1, keepdim=True)
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        r.conv1.weight = nn.Parameter(w)
        r.fc = nn.Linear(512, 1)
        nn.init.zeros_(r.fc.weight)
        nn.init.constant_(r.fc.bias, init_bias)
        self.r = r

    def forward(self, x):
        return self.r(x).squeeze(1)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    ccc = 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    sp = spearmanr(y, p)[0] if np.std(p) > 0 else float('nan')
    return dict(n=len(y), MAE=mae, RMSE=rmse, R2=r2, CCC=ccc, Spearman=sp)


def predict_pigs(model, X, owner, pig_ids, dev, bs=256):
    model.eval()
    sel = np.isin(owner, pig_ids)
    xs, os_ = X[sel], owner[sel]
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(xs), bs):
            xb = torch.from_numpy(xs[k:k + bs].astype(np.float32))[:, None].to(dev)
            out.append(model(xb).float().cpu().numpy())
    out = np.concatenate(out)
    return np.array([out[os_ == i].mean() for i in pig_ids])


def train_fold(X, owner, y, tr, va, args, dev, log):
    model = Net(float(y[tr].mean())).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    fidx = np.where(np.isin(owner, tr))[0]
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(fidx)
        for k in range(0, len(fidx), args.bs):
            b = fidx[k:k + args.bs]
            xb = torch.from_numpy(X[b].astype(np.float32))[:, None]
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
            xb = xb.to(dev)
            yb = torch.from_numpy(y[owner[b]]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.huber_loss(model(xb).float(), yb, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        pv = predict_pigs(model, X, owner, va, dev)
        vmae = np.mean(np.abs(pv - y[va]))
        if vmae < best:
            best, best_state = vmae, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f'    ep {ep + 1:02d} val MAE {vmae:.3f}{" *" if vmae == best else ""}')
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', choices=['old', 'new'], required=True)
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
    out_dir = os.path.join(RUNS_ROOT, f'labelfix_{a.labels}_s{a.size}')
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    pigs, X, owner, y = load_data(a.labels, a.size)
    groups = np.array([p['group'] for p in pigs])
    y_new = np.array([p['new'] for p in pigs], np.float32)
    log(f'[{time.strftime("%F %T")}] labels={a.labels} pigs={len(pigs)} groups={len(set(groups))} '
        f'frames={len(X)} y mean {y.mean():.2f} sd {y.std():.2f}  (load {time.time() - t0:.0f}s)')

    # 折由「新标签」分层确定，old/new 两次运行完全相同
    strat = np.digitize(y_new, np.quantile(y_new, [0.2, 0.4, 0.6, 0.8]))
    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    all_p, all_base, fold_rows = np.zeros(len(pigs)), np.zeros(len(pigs)), []
    for k, (trv, te) in enumerate(sgkf.split(np.zeros(len(pigs)), strat, groups)):
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed + k)
        tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
        tr, va = trv[tr_i], trv[va_i]
        log(f'  fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
        model = train_fold(X, owner, y, tr, va, a, dev, log)
        p = predict_pigs(model, X, owner, te, dev)
        all_p[te], all_base[te] = p, y[tr].mean()
        m, mb = metrics(y[te], p), metrics(y[te], np.full(len(te), y[tr].mean()))
        fold_rows.append(dict(fold=k + 1, **{f'model_{kk}': v for kk, v in m.items()},
                              **{f'base_{kk}': v for kk, v in mb.items() if kk in ('MAE', 'RMSE')}))
        log(f'  fold {k + 1} TEST  model MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f} CCC {m["CCC"]:.3f} '
            f'| mean-baseline MAE {mb["MAE"]:.3f}')
        del model
        torch.cuda.empty_cache()

    g = metrics(y, all_p)
    gb = metrics(y, all_base)
    log(f'== {a.labels} labels, pooled over folds: MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} R2 {g["R2"]:.3f} '
        f'CCC {g["CCC"]:.3f} Spearman {g["Spearman"]:.3f} | baseline MAE {gb["MAE"]:.3f} RMSE {gb["RMSE"]:.3f} '
        f'| {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), pooled=g, baseline=gb, folds=fold_rows), f, ensure_ascii=False, indent=1,
                  default=float)
    with open(os.path.join(out_dir, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'y_used', 'y_new', 'y_old', 'pred', 'baseline', 'match_level'])
        for p_, pr, b in zip(pigs, all_p, all_base):
            w.writerow([p_['uid'], p_['group'], p_[a.labels], p_['new'], p_['old'], round(pr, 3), round(b, 3),
                        p_['match_level']])


if __name__ == '__main__':
    main()
