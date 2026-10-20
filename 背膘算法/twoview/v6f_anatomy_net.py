# -*- coding: utf-8 -*-
"""
v6f_anatomy_net.py — 解剖引导高度图网络（新模型）：全局 CNN + 沿体长的多横截面「可学习卡尺」分支 + 多帧门控注意力聚合

来源 / 背景
  DECISIONS 2026-10-14（三）：本人决定做新模型作为算法创新点。依据：v6a 中显式横截面几何（虚拟卡尺、分段体宽）在
  高度图 CNN 分数之上仍有 +0.035（p=0.001）→ CNN 没有充分学到背部横截面形状；体况卡尺的原理正是「最后肋骨处背部
  横截面的尖 / 平」。本模型把这一测量原理做进网络：沿体长取 K 条横截面曲线，由可学习的编码器与注意力读出。
  只用 A 场（本人 10-14 确认）。

模型（全部为本脚本自写的 PyTorch 模块；ResNet-18 来自 torchvision，与 v4b 相同的单通道改造）
  ① 全局分支：高度图 128 × 256 → ResNet-18 → 512 维
  ② 横截面分支（variant 含 prof）：每帧按 v5b 的规则（栏杆列剔除、两端列顶高度较低者为头、翻转为头在左）在距头
     f = 0.10…0.95 的 K = 16 个位置各取 ±16 mm 列窗的逐行中位横截面，以脊线为中心截取 ±48 行（±375 mm），
     3 通道：离地高度 / 1200、相对脊线高度 / 300、有值掩膜 → 每条横截面 1D CNN 编码为 64 维 token
     → 加可学习位置编码 → 2 层 Transformer 编码器（4 头）→ 位置注意力池化（权重可解释：网络看重体长哪个位置）→ 64 维
  ③ 多帧聚合（variant 含 mil）：门控注意力 MIL（Ilse 等 2018），以猪为单位训练（每步 8 头猪 × 8 帧，测试用该猪全部帧）；
     不含 mil 时与 v4b 相同：逐帧训练、猪级 = 帧预测均值
  输出层：线性（权重 0 初始化，偏置 = 训练均值）
  变体（预先指定，不再增加）：base（仅 ①，即 v4b 重跑）/ prof（①+②）/ mil（①+③）/ prof_mil（①+②+③，完整模型）

训练与评估（与 v4b 相同：同一套外层 5 折、内层验证集 StratifiedGroupKFold(7, seed=42+k) 第 1 份、种子 42 / 43 / 44、
  Huber δ=1、AdamW 3e-4、15 epoch 余弦、帧质控、高度图水平 + 垂直翻转；横截面另做左右镜像 p=0.5）
  每个 epoch 记录内层验证集与外层测试折的猪级预测，从同一次训练得到两种选模口径：
    - inner（本文协议，主结果）：按内层验证集猪级 MAE 选 epoch
    - testsel（吕年珅稿协议，仅用于同口径对照）：按外层测试折 MAE 选 epoch —— 测试信息参与了选模，结果偏乐观，
      必须如实标注，不能称为独立测试
  指标两种算法都报：合并折外预测（本文）与 5 折各自计算后取平均（吕年珅稿）。

输出：runs/v6_anatomy/{_cache/profiles.npz, <variant>/{log.txt, result.json, test_preds.csv, _seed_ckpt/}}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6f_anatomy_net.py --variant prof_mil
      --smoke（1 种子、1 折、2 epoch，检查能否跑通）；--compare（汇总四个变体）
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
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import Net, metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v4b_hmap_models import DM_HMAP, FOLDS_SRC, H_MAX, INDEX  # noqa: E402
from v5b_virtual_caliper import CELL, col_top  # noqa: E402

OUT = os.path.join(RUNS_ROOT, 'v6_anatomy')
CACHE = os.path.join(OUT, '_cache', 'profiles_v2.npz')   # v2：头尾判定加退路（见 frame_profiles）
V4B = os.path.join(RUNS_ROOT, 'v4_hmap', 'hmap', 'test_preds.csv')
V1 = os.path.join(RUNS_ROOT, 'v1_tabular', 'preds.csv')
F_POS = np.linspace(0.10, 0.95, 16)
HALF = 48                      # 横截面以脊线为中心 ±48 行（±375 mm）
L_PROF = 2 * HALF + 1
VARIANTS = ('base', 'prof', 'mil', 'prof_mil')
BAG_S, BAG_F = 8, 8            # MIL 训练：每步 8 头猪 × 8 帧（= 64 帧，与逐帧训练的 batch 相同）


# ---------------------------------------------------------------- 数据
def load_hmaps_paths(uids, dm_dir, area_rel, len_rel):
    """与 v4b.load_hmaps 完全相同的帧质控，另返回帧路径（用于横截面缓存对齐）。"""
    pos = {u: i for i, u in enumerate(uids)}
    with open(os.path.join(dm_dir, 'metadata.csv'), encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['uid'] in pos and r['ok'] == '1']
    by = {}
    for r in rows:
        by.setdefault(r['uid'], []).append(r)
    keep = []
    for u, rs in by.items():
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        keep += [r for r in rs if float(r['area_cm2']) >= area_rel * ma and float(r['length_mm']) >= len_rel * ml]
    X = np.stack([np.clip(np.load(os.path.join(dm_dir, r['frame_path'])).astype(np.float32) / H_MAX, 0, 1)
                  .astype(np.float16) for r in keep])
    owner = np.array([pos[r['uid']] for r in keep])
    return X, owner, [r['frame_path'] for r in keep]


def frame_profiles(h):
    """h：离地高度（mm），128 × 256。返回 (K, 3, L_PROF) 横截面、(K,) 有效标志；整帧无法定位时全部无效。"""
    P = np.zeros((len(F_POS), 3, L_PROF), np.float32)
    V = np.zeros(len(F_POS), bool)
    m = h > 0
    width = m.sum(0) * CELL
    body = np.where(width >= 80)[0]
    if len(body) < 40:
        return P, V
    c0, c1 = body[0], body[-1]
    top = col_top(h, m)
    inb = np.zeros(h.shape[1], bool)
    inb[c0:c1 + 1] = True
    back = np.nanmedian(top[inb])
    wmed = np.median(width[inb & (width > 0)])
    colmax = np.where(m.any(0), h.max(0), 0)
    bar = inb & ((colmax > back + 120) | (width < 0.6 * wmed))
    bar = bar | np.roll(bar, 1) | np.roll(bar, -1)
    good = inb & ~bar
    L = c1 - c0
    n20 = max(int(0.2 * L), 3)
    with np.errstate(all='ignore'):
        t_lo = np.nanmedian(np.where(good[c0:c0 + n20], top[c0:c0 + n20], np.nan))
        t_hi = np.nanmedian(np.where(good[c1 - n20:c1 + 1], top[c1 - n20:c1 + 1], np.nan))
        if np.isnan(t_lo) or np.isnan(t_hi):
            # 与 v5b 的差别：一端 20% 内全被判为栏杆列（多为头部窄于体宽中位数 60%）时，改用该端全部列判头尾
            t_lo, t_hi = np.nanmedian(top[c0:c0 + n20]), np.nanmedian(top[c1 - n20:c1 + 1])
    if np.isnan(t_lo) or np.isnan(t_hi):
        return P, V
    if t_lo > t_hi:  # 翻转为头在左（同 v5b）
        h, m, good = h[:, ::-1], m[:, ::-1], good[::-1]
        c0, c1 = h.shape[1] - 1 - c1, h.shape[1] - 1 - c0
    for k, f in enumerate(F_POS):
        c = int(round(c0 + f * L))
        win = None
        for off in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
            idx = np.arange(c + off - 2, c + off + 3)
            idx = idx[(idx >= 0) & (idx < h.shape[1])]
            idx = idx[good[idx]]
            if len(idx) >= 2:
                win = idx
                break
        if win is None:
            continue
        with np.errstate(all='ignore'):
            prof = np.nanmedian(np.where(m[:, win], h[:, win], np.nan), axis=1)
        rows = np.where(~np.isnan(prof))[0]
        if len(rows) < 10:
            continue
        mid = (rows[0] + rows[-1]) // 2
        cand = np.arange(max(mid - 5, 0), min(mid + 6, len(prof)))
        cand = cand[~np.isnan(prof[cand])]
        if not len(cand):
            continue
        sp = cand[np.argmax(prof[cand])]
        r = np.arange(sp - HALF, sp + HALF + 1)
        inside = (r >= 0) & (r < len(prof))
        v = np.full(L_PROF, np.nan, np.float32)
        v[inside] = prof[r[inside]]
        pres = ~np.isnan(v)
        P[k, 0] = np.where(pres, v / H_MAX, 0)
        P[k, 1] = np.where(pres, (v - prof[sp]) / 300.0, 0)
        P[k, 2] = pres
        V[k] = True
    return P, V


def get_profiles(X, paths, log):
    if os.path.isfile(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z['paths']) == list(paths):
            log(f'   profiles: loaded cache {CACHE}')
            return z['prof'], z['valid']
        log('   profiles: cache frame list differs → rebuild')
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    t0 = time.time()
    prof = np.zeros((len(X), len(F_POS), 3, L_PROF), np.float16)
    valid = np.zeros((len(X), len(F_POS)), bool)
    for i in range(len(X)):
        p, v = frame_profiles(X[i].astype(np.float32) * H_MAX)
        prof[i], valid[i] = p, v
    np.savez(CACHE, prof=prof, valid=valid, paths=np.array(paths, dtype=object), f_pos=F_POS)
    log(f'   profiles: built {len(X)} frames in {time.time() - t0:.0f}s; valid positions {valid.mean():.1%}; '
        f'frames with no valid position {np.mean(~valid.any(1)):.1%}')
    return prof, valid


# ---------------------------------------------------------------- 模型
class ProfileBranch(nn.Module):
    def __init__(self, K, d=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, d, 3, stride=2, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.pos = nn.Parameter(torch.randn(1, K, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.score = nn.Linear(d, 1)

    def forward(self, p, valid):
        B, K = p.shape[:2]
        t = self.enc(p.flatten(0, 1)).view(B, K, -1) + self.pos
        v = valid.clone()
        v[~v.any(1)] = True                     # 整帧无有效横截面时不屏蔽（避免全屏蔽）
        t = self.tf(t, src_key_padding_mask=~v)
        s = self.score(t).squeeze(-1).float().masked_fill(~v, -1e4)
        w = torch.softmax(s, 1)
        return (w.unsqueeze(-1) * t.float()).sum(1), w


class AnatomyNet(nn.Module):
    def __init__(self, init_bias, use_prof, use_mil, K=len(F_POS)):
        super().__init__()
        r = Net(init_bias).r
        r.fc = nn.Identity()
        self.cnn = r
        self.prof = ProfileBranch(K) if use_prof else None
        dim = 512 + (64 if use_prof else 0)
        self.use_mil = use_mil
        if use_mil:
            self.att_v, self.att_u, self.att_w = nn.Linear(dim, 128), nn.Linear(dim, 128), nn.Linear(128, 1)
        self.head = nn.Linear(dim, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)

    def embed(self, x, p, v):
        z = self.cnn(x).float()
        w = None
        if self.prof is not None:
            zp, w = self.prof(p, v)
            z = torch.cat([z, zp], 1)
        return z, w

    def forward_frames(self, x, p, v):
        z, w = self.embed(x, p, v)
        return self.head(z).squeeze(1), w

    def forward_bags(self, x, p, v, mask):
        S, Fn = mask.shape
        z, w = self.embed(x.flatten(0, 1), p.flatten(0, 1), v.flatten(0, 1))
        z = z.view(S, Fn, -1)
        a = self.att_w(torch.tanh(self.att_v(z)) * torch.sigmoid(self.att_u(z))).squeeze(-1)
        a = torch.softmax(a.float().masked_fill(~mask, -1e4), 1)
        out = self.head((a.unsqueeze(-1) * z).sum(1)).squeeze(1)
        return out, (None if w is None else w.view(S, Fn, -1)), a


# ---------------------------------------------------------------- 训练 / 预测
def to_dev(X, P, V, idx, dev, train):
    x = torch.from_numpy(X[idx].astype(np.float32)).unsqueeze(-3)   # (..., 1, H, W)
    p = torch.from_numpy(P[idx].astype(np.float32))
    v = torch.from_numpy(V[idx])
    if train:
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-1])
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[-2])
        if np.random.rand() < 0.5:
            p = torch.flip(p, dims=[-1])      # 横截面左右镜像
    return x.to(dev), p.to(dev), v.to(dev)


def predict(model, X, P, V, owner, frames_of, pigs, dev):
    """返回 {pig: 预测}、{pig: 位置注意力均值 (K,)}。"""
    model.eval()
    pred, posw = {}, {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        if not model.use_mil:
            idx = np.concatenate([frames_of[i] for i in pigs])
            for k in range(0, len(idx), 256):
                b = idx[k:k + 256]
                o, w = model.forward_frames(*to_dev(X, P, V, b, dev, False))
                o = o.float().cpu().numpy()
                w = None if w is None else w.float().cpu().numpy()
                for j, fi in enumerate(b):
                    pred.setdefault(int(owner[fi]), []).append(o[j])
                    if w is not None:
                        posw.setdefault(int(owner[fi]), []).append(w[j])
            pred = {i: float(np.mean(v)) for i, v in pred.items()}
        else:
            for k in range(0, len(pigs), BAG_S):
                ps = pigs[k:k + BAG_S]
                fm = max(len(frames_of[i]) for i in ps)
                idx = np.stack([np.resize(frames_of[i], fm) for i in ps])
                mask = torch.from_numpy(np.stack([np.arange(fm) < len(frames_of[i]) for i in ps])).to(dev)
                o, w, _ = model.forward_bags(*to_dev(X, P, V, idx, dev, False), mask)
                o = o.float().cpu().numpy()
                for j, i in enumerate(ps):
                    pred[int(i)] = float(o[j])
                    if w is not None:
                        posw[int(i)] = w[j, :len(frames_of[i])].float().mean(0).cpu().numpy()
    posw = {i: np.mean(v, 0) if isinstance(v, list) else v for i, v in posw.items()}
    return pred, posw


def predict_frames(model, X, P, V, owner, frames_of, pigs, dev):
    """逐帧预测（仅非 MIL 变体）：返回 [(帧下标, 预测)]，帧顺序同 frames_of。"""
    model.eval()
    idx = np.concatenate([frames_of[i] for i in pigs])
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(idx), 256):
            b = idx[k:k + 256]
            o, _ = model.forward_frames(*to_dev(X, P, V, b, dev, False))
            out += list(zip(b.tolist(), o.float().cpu().numpy().tolist()))
    return out


def train_eval(variant, X, P, V, owner, frames_of, y, tr, va, te, a, dev, log, return_best=False):
    """训练 15 epoch；每个 epoch 记录验证集 MAE 与测试猪预测、位置注意力。
    return_best=True 时另返回内层验证集 MAE 最小 epoch 的模型（v6g 用），否则只返回 hist。"""
    use_prof, use_mil = 'prof' in variant, 'mil' in variant
    model = AnatomyNet(float(y[tr].mean()), use_prof, use_mil).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler('cuda')
    fidx = np.concatenate([frames_of[i] for i in tr])
    steps = int(np.ceil(len(fidx) / a.bs))
    hist = []
    for ep in range(a.epochs):
        model.train()
        if not use_mil:
            np.random.shuffle(fidx)
        order = np.random.permutation(np.repeat(tr, int(np.ceil(steps * BAG_S / len(tr))) + 1))
        for st in range(steps):
            if not use_mil:
                b = fidx[st * a.bs:(st + 1) * a.bs]
                yb = torch.from_numpy(y[owner[b]]).to(dev)
                with torch.autocast('cuda', dtype=torch.float16):
                    out, _ = model.forward_frames(*to_dev(X, P, V, b, dev, True))
            else:
                ps = order[st * BAG_S:(st + 1) * BAG_S]
                idx = np.stack([np.random.choice(frames_of[i], BAG_F, replace=len(frames_of[i]) < BAG_F) for i in ps])
                mask = torch.ones(len(ps), BAG_F, dtype=torch.bool, device=dev)
                yb = torch.from_numpy(y[ps]).to(dev)
                with torch.autocast('cuda', dtype=torch.float16):
                    out, _, _ = model.forward_bags(*to_dev(X, P, V, idx, dev, True), mask)
            loss = F.huber_loss(out.float(), yb, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        pv, _ = predict(model, X, P, V, owner, frames_of, va, dev)
        pt, wt = predict(model, X, P, V, owner, frames_of, te, dev)
        vmae = float(np.mean([abs(pv[i] - y[i]) for i in va]))
        tmae = float(np.mean([abs(pt[i] - y[i]) for i in te]))
        hist.append(dict(val_mae=vmae, test_mae=tmae, test_pred=pt, posw=wt))
        log(f'    ep {ep + 1:02d} val MAE {vmae:.3f} | test MAE {tmae:.3f}')
        if return_best and vmae < min([h['val_mae'] for h in hist[:-1]], default=np.inf):  # 与 argmin 一致：取首个最小
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
    if return_best:
        model.load_state_dict(best_state)
        return hist, model
    del model
    torch.cuda.empty_cache()
    return hist


# ---------------------------------------------------------------- 评估
def fold_avg(y, p, fold):
    ms = [metrics(y[fold == k], p[fold == k]) for k in sorted(set(fold))]
    return dict(R2=float(np.mean([m['R2'] for m in ms])), MAE=float(np.mean([m['MAE'] for m in ms])),
                R2_per_fold=[float(m['R2']) for m in ms])


def read_setup():
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
    return uids, groups, fold, y


def compare(log):
    uids, groups, fold, y = read_setup()
    with open(V1, encoding='utf-8-sig') as f:
        v1 = {r['uid']: r for r in csv.DictReader(f)}
    line = np.array([v1[u]['line'] for u in uids])
    parity = np.array([float(v1[u]['parity']) if v1[u]['parity'] != '' else np.nan for u in uids])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2)}
    P = {}
    for v in VARIANTS:
        fp = os.path.join(OUT, v, 'test_preds.csv')
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding='utf-8-sig') as f:
            d = {r['uid']: r for r in csv.DictReader(f)}
        P[(v, 'inner')] = np.array([float(d[u]['pred_inner']) for u in uids])
        P[(v, 'testsel')] = np.array([float(d[u]['pred_testsel']) for u in uids])
    with open(V4B, encoding='utf-8-sig') as f:
        d = {r['uid']: float(r['pred_mean']) for r in csv.DictReader(f)}
    P[('v4b', 'inner')] = np.array([d[u] for u in uids])
    log(f'\n[{time.strftime("%F %T")}] == compare (seed-mean predictions)')
    log('variant   protocol  | pooled R2   MAE  | fold-avg R2   MAE  | R2 DAN p1  NAM multi')
    res = {}
    for (v, pr), p in P.items():
        g, fa = metrics(y, p), fold_avg(y, p, fold)
        res[f'{v}|{pr}'] = dict(pooled=g, fold_avg=fa, subsets={s: metrics(y[m], p[m]) for s, m in subsets.items()})
        log(f'{v:9s} {pr:8s}  | {g["R2"]:.3f}  {g["MAE"]:.3f} | {fa["R2"]:.3f}  {fa["MAE"]:.3f}  | '
            f'{res[f"{v}|{pr}"]["subsets"]["DAN_parity1"]["R2"]:.3f}  {res[f"{v}|{pr}"]["subsets"]["NAM_multiparous"]["R2"]:.3f}')
    log('\npaired comparisons, inner protocol (Δ = variant − base): group bootstrap 95% CI; Wilcoxon on |error|')
    res['comparisons'] = {}
    for ref in ('base', 'v4b'):
        if (ref, 'inner') not in P:
            continue
        for v in VARIANTS:
            if v == ref or (v, 'inner') not in P:
                continue
            for s, m in subsets.items():
                pa, pb = P[(v, 'inner')][m], P[(ref, 'inner')][m]
                bs = group_bootstrap(groups[m], y[m], pa, pb)
                pw = float(wilcoxon(np.abs(pa - y[m]), np.abs(pb - y[m])).pvalue)
                res['comparisons'][f'{v} vs {ref} [{s}]'] = dict(**bs, wilcoxon_p=pw)
                log(f'   {v:9s} vs {ref:5s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                    f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')
    with open(os.path.join(OUT, 'compare.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.compare:
        logf = open(os.path.join(OUT, 'compare_log.txt'), 'a', encoding='utf-8')

        def clog(s=''):
            print(s, flush=True)
            logf.write(s + '\n')
            logf.flush()
        compare(clog)
        return
    assert a.variant, '--variant 必填'
    tag = a.variant + ('_smoke' if a.smoke else '')
    od = os.path.join(OUT, tag)
    os.makedirs(os.path.join(od, '_seed_ckpt'), exist_ok=True)
    logf = open(os.path.join(od, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    dev = 'cuda'
    uids, groups, fold, y = read_setup()
    X, owner, paths = load_hmaps_paths(uids, DM_HMAP, a.area_rel, a.len_rel)
    log(f'[{time.strftime("%F %T")}] v6f variant={a.variant} args={vars(a)} | frames {len(X)} (load {time.time() - t0:.0f}s)')
    P, V = get_profiles(X, paths, log)
    frames_of = {i: np.where(owner == i)[0] for i in range(len(uids))}
    assert all(len(v) for v in frames_of.values()), '有猪没有帧'
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    seeds = a.seeds[:1] if a.smoke else a.seeds
    folds = [0] if a.smoke else range(5)
    if a.smoke:
        a.epochs = 2
    n = len(uids)
    store = {}
    for s in seeds:
        ck = os.path.join(od, '_seed_ckpt', f'seed{s}.npz')
        if os.path.isfile(ck):
            z = np.load(ck, allow_pickle=True)
            store[s] = dict(z['store'].item())
            log(f'  seed {s}: loaded checkpoint')
            continue
        np.random.seed(s)
        torch.manual_seed(s)
        ep_pred = np.full((a.epochs, n), np.nan)
        ep_posw = np.full((a.epochs, n, len(F_POS)), np.nan)
        val_mae = np.full((5, a.epochs), np.nan)
        test_mae = np.full((5, a.epochs), np.nan)
        for k in folds:
            trv, te = np.where(fold != k + 1)[0], np.where(fold == k + 1)[0]
            inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
            tr, va = trv[tr_i], trv[va_i]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
            hist = train_eval(a.variant, X, P, V, owner, frames_of, y, tr, va, te, a, dev, log)
            for e, h in enumerate(hist):
                for i in te:
                    ep_pred[e, i] = h['test_pred'][i]
                    if i in h['posw']:
                        ep_posw[e, i] = h['posw'][i]
                val_mae[k, e], test_mae[k, e] = h['val_mae'], h['test_mae']
            ei, et = int(np.argmin(val_mae[k])), int(np.argmin(test_mae[k]))
            log(f'  seed {s} fold {k + 1}: inner-selected ep {ei + 1} test MAE {test_mae[k, ei]:.3f} | '
                f'test-selected ep {et + 1} test MAE {test_mae[k, et]:.3f} | {(time.time() - t0) / 60:.1f} min')
        p_in, p_ts, w_in = np.full(n, np.nan), np.full(n, np.nan), np.full((n, len(F_POS)), np.nan)
        for k in folds:
            te = np.where(fold == k + 1)[0]
            ei, et = int(np.argmin(val_mae[k])), int(np.argmin(test_mae[k]))
            p_in[te], p_ts[te], w_in[te] = ep_pred[ei, te], ep_pred[et, te], ep_posw[ei, te]
        store[s] = dict(p_inner=p_in, p_testsel=p_ts, posw_inner=w_in, val_mae=val_mae, test_mae=test_mae)
        if not a.smoke:
            np.savez(ck, store=np.array(store[s], dtype=object))
        ok = ~np.isnan(p_in)
        g1, g2 = metrics(y[ok], p_in[ok]), metrics(y[ok], p_ts[ok])
        log(f'== seed {s}: inner R2 {g1["R2"]:.3f} MAE {g1["MAE"]:.3f} | testsel R2 {g2["R2"]:.3f} MAE {g2["MAE"]:.3f} '
            f'| {(time.time() - t0) / 60:.1f} min')
    if a.smoke:
        log('smoke test finished')
        return
    pin = np.mean([store[s]['p_inner'] for s in seeds], 0)
    pts = np.mean([store[s]['p_testsel'] for s in seeds], 0)
    posw = np.nanmean([store[s]['posw_inner'] for s in seeds], 0)
    res = {'args': vars(a), 'f_pos': F_POS.tolist()}
    for nm, p in (('inner', pin), ('testsel', pts)):
        res[nm] = dict(pooled=metrics(y, p), fold_avg=fold_avg(y, p, fold),
                       per_seed_pooled=[metrics(y, store[s][f'p_{nm}'])['R2'] for s in seeds])
        log(f'== {a.variant} {nm:8s} seed-mean: pooled R2 {res[nm]["pooled"]["R2"]:.3f} MAE {res[nm]["pooled"]["MAE"]:.3f} '
            f'| fold-avg R2 {res[nm]["fold_avg"]["R2"]:.3f} MAE {res[nm]["fold_avg"]["MAE"]:.3f}')
    if 'prof' in a.variant:
        mw = np.nanmean(posw, 0)
        res['position_attention_mean'] = mw.tolist()
        log('   position attention (mean over test pigs, inner-selected epoch): '
            + ', '.join(f'{f:.2f}:{w:.3f}' for f, w in zip(F_POS, mw)))
    with open(os.path.join(od, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(od, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + [f'pred_inner_s{s}' for s in seeds]
                   + [f'pred_testsel_s{s}' for s in seeds] + ['pred_inner', 'pred_testsel'])
        for i, u in enumerate(uids):
            w.writerow([u, groups[i], fold[i], float(y[i])] + [round(float(store[s]['p_inner'][i]), 3) for s in seeds]
                       + [round(float(store[s]['p_testsel'][i]), 3) for s in seeds]
                       + [round(float(pin[i]), 3), round(float(pts[i]), 3)])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {od}')


if __name__ == '__main__':
    main()
