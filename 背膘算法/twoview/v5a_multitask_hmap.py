# -*- coding: utf-8 -*-
"""
v5a_multitask_hmap.py — 高度图 ResNet-18 的训练目标改进：辅助监督（体况卡尺 / 体尺）、不平衡回归、方差头
                        （与 v4b 同一套 5 折、同一内层验证集、同种子 → 可逐头配对比较）

来源 / 背景
  DECISIONS 2026-10-13（四）之后本人要求在视觉算法上再加新内容（10-13）：
  - 体况卡尺评分只在训练时可得、部署时不想再人工测 → 作为训练时的辅助目标（multi-task / learning using
    privileged information），预测时只用图像。问题：辅助监督能否提高背膘精度（尤其丹系头胎）？
  - A6：校准斜率 0.37（预测向均值收缩），对育种排序不利 → 不平衡回归（LDS 风格的密度倒数加权）。
  - V6 / v4i：事后拟合的 σ̂ 与 |误差| 几乎不相关 → 网络直接输出方差（异方差头），重新检验复测分流。

模式（--mode）
  cal      背膘 + 卡尺（训练猪内标准化，缺失卡尺不计损失），loss = Huber(bf) + λ·Huber(aux)
  caldims  背膘 + 卡尺 + v4e 体尺 6 列（各列训练猪内标准化）
  lds      只有背膘；每头猪权重 ∝ 1/√密度（训练猪背膘的高斯核密度，带宽 2 mm），归一到均值 1，截断 [0.2, 5]
  nll      背膘 + 对数方差头：loss = Huber(bf) + λ·高斯 NLL（均值 detach，只训练方差头与共享骨干）
  其余与 v4b 完全相同：ResNet-18、128 × 256 输入、高度 / 1200 mm、水平 + 垂直翻转、Huber δ=1、AdamW 3e-4、
  15 epoch 余弦、batch 64、帧质控 0.8 / 0.85；**epoch 仍只按内层验证集背膘猪级 MAE 选**；种子 42 / 43 / 44。

输出：runs/v5_multitask/<mode>/{log.txt, result.json, test_preds.csv, frame_preds.csv, _seed_ckpt/}
  test_preds.csv 另含辅助头的猪级预测（cal / caldims：相机预测的卡尺评分等；nll：猪级对数方差）
  中断后重跑跳过已完成的种子
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5a_multitask_hmap.py --mode cal
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
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import Net, metrics  # noqa: E402
from v4b_hmap_models import DM_HMAP, FOLDS_SRC, INDEX, load_hmaps  # noqa: E402

CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
DIMS_CSV = os.path.join(HERE, 'runs', 'v4_hmap', 'body_dims', 'body_dims.csv')
DIMS = ('back_h_mm', 'length_mm', 'width_mm', 'area_cm2', 'volume_l', 'hw_ratio')


class MTNet(nn.Module):
    """Net（ResNet-18，单通道）的骨干 + 1 + n_out 个输出；第 0 个输出为背膘（初始化为训练均值）。"""

    def __init__(self, init_bias, n_out):
        super().__init__()
        self.r = Net(init_bias).r
        self.r.fc = nn.Linear(512, 1 + n_out)
        nn.init.zeros_(self.r.fc.weight)
        nn.init.zeros_(self.r.fc.bias)
        with torch.no_grad():
            self.r.fc.bias[0] = init_bias

    def forward(self, x):
        return self.r(x)


def predict_all(model, X, owner, pigs, dev, bs=256):
    """返回 {pig: (n_frames, 1 + n_out) 数组}。"""
    model.eval()
    idx = np.where(np.isin(owner, pigs))[0]
    out = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            p = model(torch.from_numpy(X[b].astype(np.float32))[:, None].to(dev)).float().cpu().numpy()
            for o, v in zip(owner[b], p):
                out.setdefault(int(o), []).append(v)
    return {k: np.array(v) for k, v in out.items()}


def lds_weights(y, tr, bw=2.0):
    d = np.array([np.exp(-0.5 * ((y[tr] - v) / bw) ** 2).sum() for v in y])
    w = 1 / np.sqrt(np.maximum(d, 1e-6))
    w = w / w[tr].mean()
    return np.clip(w, 0.2, 5.0).astype(np.float32)


def train_fold(X, owner, y, A, tr, va, a, dev, log):
    """A：(n_pigs, n_aux) 辅助目标（NaN = 缺失），在训练猪内标准化。"""
    n_aux = 0 if A is None else A.shape[1]
    n_out = n_aux + (1 if a.mode == 'nll' else 0)
    model = MTNet(float(y[tr].mean()), n_out).to(dev)
    if a.mode == 'nll':
        with torch.no_grad():
            model.r.fc.bias[1] = float(np.log(np.var(y[tr])))
    mu = sd = None
    if n_aux:
        mu, sd = np.nanmean(A[tr], 0), np.nanstd(A[tr], 0)
        Z = ((A - mu) / np.where(sd > 0, sd, 1)).astype(np.float32)
        M = (~np.isnan(Z)).astype(np.float32)
        Z = np.nan_to_num(Z)
    w_pig = lds_weights(y, tr) if a.mode == 'lds' else np.ones(len(y), np.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler('cuda')
    fidx = np.where(np.isin(owner, tr))[0]
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        np.random.shuffle(fidx)
        for k in range(0, len(fidx), a.bs):
            b = fidx[k:k + a.bs]
            xb = torch.from_numpy(X[b].astype(np.float32))[:, None]
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[2])
            yb = torch.from_numpy(y[owner[b]]).to(dev)
            wb = torch.from_numpy(w_pig[owner[b]]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(xb.to(dev)).float()
            loss = (F.huber_loss(out[:, 0], yb, delta=1.0, reduction='none') * wb).mean()
            if n_aux:
                zb = torch.from_numpy(Z[owner[b]]).to(dev)
                mb = torch.from_numpy(M[owner[b]]).to(dev)
                la = F.huber_loss(out[:, 1:1 + n_aux], zb, delta=1.0, reduction='none') * mb
                loss = loss + a.lam * la.sum() / mb.sum().clamp(min=1)
            if a.mode == 'nll':
                lv = out[:, 1].clamp(-6, 6)
                nll = 0.5 * (lv + (yb - out[:, 0].detach()) ** 2 / torch.exp(lv))
                loss = loss + a.lam * nll.mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        pv = predict_all(model, X, owner, va, dev)
        vmae = np.mean([abs(pv[i][:, 0].mean() - y[i]) for i in va if i in pv])
        if vmae < best:
            best, best_state = vmae, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        log(f'    ep {ep + 1:02d} val MAE {vmae:.3f}{" *" if vmae == best else ""}')
    model.load_state_dict(best_state)
    return model, (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['cal', 'caldims', 'lds', 'nll'])
    ap.add_argument('--lam', type=float, default=0.5, help='辅助损失权重（cal / caldims / nll）')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = 'cuda'
    out_dir = os.path.join(HERE, 'runs', 'v5_multitask', a.tag or a.mode)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'log.txt'), 'a', encoding='utf-8')

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
    log(f'[{time.strftime("%F %T")}] args={vars(a)}')

    aux_names, A = [], None
    if a.mode in ('cal', 'caldims'):
        with open(CAL, encoding='utf-8-sig') as f:
            # 纯整数「4」「5」（int_only）本人 10-11 确认为有效卡尺读数；只排除 implausible（0.8）
            cal = {r['uid']: (float(r['caliper']) if r['caliper_flag'] != 'implausible' and r['caliper'] else np.nan)
                   for r in csv.DictReader(f)}
        cols = [[cal.get(u, np.nan) for u in uids]]
        aux_names = ['caliper']
        if a.mode == 'caldims':
            with open(DIMS_CSV, encoding='utf-8-sig') as f:
                dims = {r['uid']: r for r in csv.DictReader(f)}
            cols += [[float(dims[u][d]) if u in dims and dims[u][d] else np.nan for u in uids] for d in DIMS]
            aux_names += list(DIMS)
        A = np.array(cols, np.float64).T
        log(f'   aux targets {aux_names}; non-missing per column {(~np.isnan(A)).sum(0).tolist()}')

    X, owner, n_all = load_hmaps(uids, DM_HMAP, a.area_rel, a.len_rel)
    have = np.array([i in set(owner.tolist()) for i in range(len(uids))])
    log(f'   frames ok {n_all} → after QC {len(X)} ({len(X) / max(n_all, 1):.1%}); pigs {have.sum()}/{len(uids)}; '
        f'input {X.shape[1:]} (load {time.time() - t0:.0f}s)')

    extra = aux_names if aux_names else (['logvar'] if a.mode == 'nll' else [])
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    preds = {s: np.full(len(uids), np.nan) for s in a.seeds}
    auxp = {s: np.full((len(uids), len(extra)), np.nan) for s in a.seeds}
    frame_rows = []
    ckpt_dir = os.path.join(out_dir, '_seed_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    for s in a.seeds:
        ck = os.path.join(ckpt_dir, f'seed{s}.npz')
        if os.path.isfile(ck):
            z = np.load(ck, allow_pickle=True)
            preds[s], auxp[s] = z['pred'], z['aux']
            frame_rows += [tuple(r) for r in z['frames'].tolist()]
            log(f'  seed {s}: loaded checkpoint {ck}')
            continue
        np.random.seed(s)
        torch.manual_seed(s)
        for k in range(5):
            trv, te = np.where(fold != k + 1)[0], np.where(fold == k + 1)[0]
            inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
            tr, va, te = [x[have[x]] for x in (trv[tr_i], trv[va_i], te)]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
            model, (mu, sd) = train_fold(X, owner, y, A, tr, va, a, dev, log)
            pt = predict_all(model, X, owner, te, dev)
            for i in te:
                preds[s][i] = pt[i][:, 0].mean()
                if aux_names:
                    auxp[s][i] = pt[i][:, 1:].mean(0) * np.where(sd > 0, sd, 1) + mu
                elif a.mode == 'nll':
                    auxp[s][i, 0] = np.log(np.mean(np.exp(np.clip(pt[i][:, 1], -6, 6))))
                frame_rows += [(uids[i], groups[i], k + 1, s, j, float(y[i]), round(float(v[0]), 3))
                               for j, v in enumerate(pt[i])]
            m = metrics(y[te], preds[s][te])
            log(f'  seed {s} fold {k + 1} TEST MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f}')
            del model
            torch.cuda.empty_cache()
        np.savez(ck, pred=preds[s], aux=auxp[s],
                 frames=np.array([r for r in frame_rows if r[3] == s], dtype=object))
        ok = ~np.isnan(preds[s])
        g = metrics(y[ok], preds[s][ok])
        log(f'== seed {s}: n={ok.sum()} MAE {g["MAE"]:.3f} R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} '
            f'Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')

    pm = np.mean([preds[s] for s in a.seeds], 0)
    am = np.mean([auxp[s] for s in a.seeds], 0)
    ok = ~np.isnan(pm)
    slope = float(np.polyfit(y[ok], pm[ok], 1)[0])
    res = {'args': vars(a), 'n': int(ok.sum()), 'aux': extra,
           'per_seed': {s: metrics(y[~np.isnan(preds[s])], preds[s][~np.isnan(preds[s])]) for s in a.seeds},
           'seed_mean': metrics(y[ok], pm[ok]), 'calibration_slope': slope}
    if aux_names:
        cm = ~np.isnan(A[:, 0]) & ok
        res['caliper_pred_vs_manual_r'] = float(np.corrcoef(A[cm, 0], am[cm, 0])[0, 1])
        log(f'   camera-predicted caliper vs manual caliper: r {res["caliper_pred_vs_manual_r"]:.3f} (n={cm.sum()})')
    g = res['seed_mean']
    log(f'== {a.mode} seed-mean: n={ok.sum()} MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} R2 {g["R2"]:.3f} '
        f'CCC {g["CCC"]:.3f} Spearman {g["Spearman"]:.3f} slope {slope:.3f} | {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(out_dir, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + [f'pred_s{s}' for s in a.seeds] + ['pred_mean']
                   + [f'aux_{n}' for n in extra])
        for i, u in enumerate(uids):
            w.writerow([u, groups[i], fold[i], float(y[i])]
                       + ['' if np.isnan(preds[s][i]) else round(preds[s][i], 3) for s in a.seeds]
                       + ['' if np.isnan(pm[i]) else round(pm[i], 3)]
                       + ['' if np.isnan(v) else round(float(v), 4) for v in am[i]])
    with open(os.path.join(out_dir, 'frame_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'seed', 'frame_idx', 'y', 'pred'])
        w.writerows(frame_rows)
    log(f'done -> {out_dir}')


if __name__ == '__main__':
    main()
