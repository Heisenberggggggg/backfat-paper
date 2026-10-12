# -*- coding: utf-8 -*-
"""
v3b_view_models.py — V3 第二步：俯视 / 斜后视 / 双视角 ResNet-18（统一协议，同一套 5 折，3 个种子）

来源 / 背景
  06 §5.1 必做第 4 项；05 §3 ⑤（两视角免配准融合，视角随机丢弃；对照胡宝玖 ICP 配准）。
  俯视基线 = cv_depthmap_labelfix.py / cv_depthmap_framepreds.py（R² 0.297）。

协议（与 cv_depthmap_labelfix 完全一致，只换输入）
  - 猪与折：直接读取 runs/framepreds_new_s128/test_preds.csv 的 663 头与 fold 列（外层 5 折）；
    内层验证集按原脚本复现：strat = 背膘 5 分位，StratifiedGroupKFold(7, shuffle, seed=42+k) 取第 1 份
    → 三种视角的训练 / 验证 / 测试猪完全相同（选 epoch 用验证集，测试折只用一次）
  - 标签：index_v2 backfat（新标签）；输入 128×128，深度归一化 300–2500 mm；Huber(δ=1)；AdamW 3e-4；
    15 epoch；水平翻转增广；猪级预测 = 帧预测均值
  - --view master：俯视 depthmap_dataset（重跑，用于同脚本同种子比较）
    --view sub   ：斜后视 depthmap_dataset_sub（v3a_gen_sub_depthmaps.py 生成）
    --view dual  ：双分支 ResNet-18（两视角各自权重），特征拼接 → 线性头；训练时每个俯视帧与同猪随机
                   一个斜后视帧配对（两相机帧不同步，不做配准）；视角随机丢弃 p=0.2（将一支特征置 0 并把
                   另一支放大一倍——仅训练时）；测试时第 i 个俯视帧按帧序配一个斜后视帧，取均值
  - 斜后视帧质控（2026-10-12 目视抽查后、训练前确定）：只保留掩膜面积占整幅画面 0.04–0.25 的帧。
    > 0.25 多为把地面 / 栏杆一起抠入（YOLO 无框时 FastSAM 退回整幅背景掩膜，约 0.98）；< 0.04 多为栏杆碎片。
    保留 15 238 / 16 168 帧（94.2%），每猪 ≥ 5 帧。俯视数据集无该列，保持原样（与 R² 0.297 基线一致）
  - 缺某一视角帧的猪：在该视角的训练 / 测试中剔除（日志报告 n）；折划分始终按 663 头计算

输出：runs/v3_views/{view}_s128/{log.txt, result.json, test_preds.csv, frame_preds.csv}
  test_preds.csv：uid, group, fold, y, pred_s42, pred_s43, pred_s44, pred_mean（未训练的猪留空）
  _seed_ckpt/seed{s}.npz：每个种子完成后的存档（2026-10-12 加：dual 运行曾因系统内存不足在最后一折崩溃）；
  重跑时自动跳过已完成的种子

用法：D:\\Anaconda\\envs\\pytorch\\python.exe v3b_view_models.py --view sub
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
from cv_depthmap_labelfix import DM_DIR, NORM_MAX, NORM_MIN, Net, metrics  # noqa: E402

DM_SUB = os.path.join(HERE, '..', 'yolo+FastSAM', 'depthmap_dataset_sub')
FOLDS_SRC = os.path.join(HERE, 'runs', 'framepreds_new_s128', 'test_preds.csv')
INDEX = os.path.join(HERE, 'index_v2.csv')


def load_frames(dm_dir, uid_col, uids, size, qc=None):
    """返回 X (N,size,size) float16、owner（uids 中的下标）；只含 uids 中的猪。
    qc=(lo, hi)：只保留 lo ≤ mask_frac ≤ hi 的帧（仅 sub 数据集有该列）。"""
    pos = {u: i for i, u in enumerate(uids)}
    X, owner = [], []
    with open(os.path.join(dm_dir, 'metadata.csv'), encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r[uid_col] in pos]
    if qc is not None:
        rows = [r for r in rows if qc[0] <= float(r['mask_frac']) <= qc[1]]
    for r in rows:
        d = np.load(os.path.join(dm_dir, r['frame_path'])).astype(np.float32)
        d = np.clip((d - NORM_MIN) / (NORM_MAX - NORM_MIN), 0, 1)
        d = F.interpolate(torch.from_numpy(d)[None, None], size=(size, size), mode='area')[0, 0]
        X.append(d.numpy().astype(np.float16))
        owner.append(pos[r[uid_col]])
    return np.stack(X), np.array(owner)


class DualNet(nn.Module):
    def __init__(self, init_bias, p_drop=0.2):
        super().__init__()
        self.a, self.b = Net(init_bias).r, Net(init_bias).r
        self.a.fc, self.b.fc = nn.Identity(), nn.Identity()
        self.fc = nn.Linear(1024, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, init_bias)
        self.p = p_drop

    def forward(self, xa, xb):
        fa, fb = self.a(xa), self.b(xb)
        if self.training and self.p > 0:
            r = torch.rand(len(fa), device=fa.device)
            drop_a = (r < self.p / 2).float()[:, None]
            drop_b = ((r >= self.p / 2) & (r < self.p)).float()[:, None]
            fa = fa * (1 - drop_a) * (1 + drop_b)  # 另一支被丢弃时放大一倍，保持尺度
            fb = fb * (1 - drop_b) * (1 + drop_a)
        return self.fc(torch.cat([fa, fb], 1)).squeeze(1)


def batches_single(Xs, own, pigs, bs, shuffle):
    idx = np.where(np.isin(own, pigs))[0]
    if shuffle:
        np.random.shuffle(idx)
    for k in range(0, len(idx), bs):
        b = idx[k:k + bs]
        yield (torch.from_numpy(Xs[b].astype(np.float32))[:, None],), own[b]


def make_pairs(own_a, own_b, pigs, random_pair):
    """俯视帧下标 ia 与同猪斜后视帧下标 ib 配对；random_pair=False 时按帧序循环配对。"""
    by_b = {}
    for j, o in enumerate(own_b):
        by_b.setdefault(int(o), []).append(j)
    ia, ib, cnt = [], [], {}
    for i, o in enumerate(own_a):
        o = int(o)
        if o in pigs and o in by_b:
            lst = by_b[o]
            c = cnt.get(o, 0)
            ia.append(i)
            ib.append(lst[np.random.randint(len(lst))] if random_pair else lst[c % len(lst)])
            cnt[o] = c + 1
    return np.array(ia), np.array(ib)


def batches_dual(XA, XB, own_a, own_b, pigs, bs, shuffle):
    pigs = set(int(p) for p in pigs)
    ia, ib = make_pairs(own_a, own_b, pigs, random_pair=shuffle)
    order = np.random.permutation(len(ia)) if shuffle else np.arange(len(ia))
    for k in range(0, len(order), bs):
        s = order[k:k + bs]
        yield ((torch.from_numpy(XA[ia[s]].astype(np.float32))[:, None],
                torch.from_numpy(XB[ib[s]].astype(np.float32))[:, None]), own_a[ia[s]])


def predict(model, batch_fn, pigs, dev):
    """返回 {pig: [frame preds]}。"""
    model.eval()
    out = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for xs, ow in batch_fn(pigs, 256, False):
            p = model(*[x.to(dev) for x in xs]).float().cpu().numpy()
            for o, v in zip(ow, p):
                out.setdefault(int(o), []).append(float(v))
    return out


def train_fold(make_model, batch_fn, y, tr, va, a, dev, log):
    model = make_model(float(y[tr].mean())).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        for xs, ow in batch_fn(tr, a.bs, True):
            if np.random.rand() < 0.5:
                xs = tuple(torch.flip(x, dims=[3]) for x in xs)
            yb = torch.from_numpy(y[ow]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.huber_loss(model(*[x.to(dev) for x in xs]).float(), yb, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        pv = predict(model, batch_fn, va, dev)
        vmae = np.mean([abs(np.mean(pv[i]) - y[i]) for i in va if i in pv])
        if vmae < best:
            best, best_state = vmae, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f'    ep {ep + 1:02d} val MAE {vmae:.3f}{" *" if vmae == best else ""}')
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--view', required=True, choices=['master', 'sub', 'dual'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--p-drop', type=float, default=0.2)
    ap.add_argument('--sub-qc', type=float, nargs=2, default=[0.04, 0.25],
                    help='斜后视帧质控：掩膜面积占整幅画面比例的保留区间（2026-10-12 目视抽查确定）')
    a = ap.parse_args()
    dev = 'cuda'
    out_dir = os.path.join(HERE, 'runs', 'v3_views', f'{a.view}_s{a.size}')
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
    assert uids == sorted(uids), '折文件猪顺序应与 load_data 一致（按 uid 排序）'
    groups = np.array([r['group'] for r in prow])
    fold = np.array([int(r['fold']) for r in prow])
    with open(INDEX, encoding='utf-8-sig') as f:
        bf = {r['uid']: float(r['backfat']) for r in csv.DictReader(f)}
    y = np.array([bf[u] for u in uids], np.float32)
    assert np.allclose(y, [float(r['y']) for r in prow]), 'index_v2 与折文件标签不一致'

    if a.view in ('master', 'dual'):
        XA, own_a = load_frames(DM_DIR, 'pig_id', uids, a.size)
    if a.view in ('sub', 'dual'):
        XB, own_b = load_frames(DM_SUB, 'uid', uids, a.size, qc=tuple(a.sub_qc))
    if a.view == 'master':
        have = set(own_a.tolist())
        batch_fn = lambda pigs, bs, sh: batches_single(XA, own_a, pigs, bs, sh)
        make_model = lambda b: Net(b)
    elif a.view == 'sub':
        have = set(own_b.tolist())
        batch_fn = lambda pigs, bs, sh: batches_single(XB, own_b, pigs, bs, sh)
        make_model = lambda b: Net(b)
    else:
        have = set(own_a.tolist()) & set(own_b.tolist())
        batch_fn = lambda pigs, bs, sh: batches_dual(XA, XB, own_a, own_b, pigs, bs, sh)
        make_model = lambda b: DualNet(b, a.p_drop)
    have_mask = np.array([i in have for i in range(len(uids))])
    nfr = {}
    if a.view != 'sub':
        nfr['master'] = len(own_a)
    if a.view != 'master':
        nfr['sub'] = len(own_b)
    log(f'[{time.strftime("%F %T")}] view={a.view} args={vars(a)}')
    log(f'   pigs with this view: {have_mask.sum()}/{len(uids)} (missing: '
        f'{[uids[i] for i in np.where(~have_mask)[0]][:10]}); frames {nfr} (load {time.time() - t0:.0f}s)')

    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    preds = {s: np.full(len(uids), np.nan) for s in a.seeds}
    frame_rows = []
    ckpt_dir = os.path.join(out_dir, '_seed_ckpt')  # 每个种子完成后存档，中断后续跑跳过已完成种子
    os.makedirs(ckpt_dir, exist_ok=True)
    for s in a.seeds:
        ck = os.path.join(ckpt_dir, f'seed{s}.npz')
        if os.path.isfile(ck):
            z = np.load(ck, allow_pickle=True)
            preds[s] = z['pred']
            frame_rows += [tuple(r) for r in z['frames'].tolist()]
            log(f'  seed {s}: loaded checkpoint {ck}')
            continue
        np.random.seed(s)
        torch.manual_seed(s)
        for k in range(5):
            te_all = np.where(fold == k + 1)[0]
            trv = np.where(fold != k + 1)[0]
            inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
            tr, va, te = [x[have_mask[x]] for x in (trv[tr_i], trv[va_i], te_all)]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
            model = train_fold(make_model, batch_fn, y, tr, va, a, dev, log)
            pt = predict(model, batch_fn, te, dev)
            for i in te:
                preds[s][i] = np.mean(pt[i])
                for j, v in enumerate(pt[i]):
                    frame_rows.append((uids[i], groups[i], k + 1, s, j, float(y[i]), round(v, 3)))
            m = metrics(y[te], preds[s][te])
            log(f'  seed {s} fold {k + 1} TEST MAE {m["MAE"]:.3f} R2 {m["R2"]:.3f}')
            del model
            torch.cuda.empty_cache()
        ok = ~np.isnan(preds[s])
        np.savez(ck, pred=preds[s], frames=np.array([r for r in frame_rows if r[3] == s], dtype=object))
        g = metrics(y[ok], preds[s][ok])
        log(f'== seed {s}: n={ok.sum()} MAE {g["MAE"]:.3f} R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} '
            f'Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')

    pm = np.mean([preds[s] for s in a.seeds], 0)
    ok = ~np.isnan(pm)
    res = {'args': vars(a), 'n': int(ok.sum()),
           'per_seed': {s: metrics(y[~np.isnan(preds[s])], preds[s][~np.isnan(preds[s])]) for s in a.seeds},
           'seed_mean': metrics(y[ok], pm[ok])}
    g = res['seed_mean']
    log(f'== {a.view} seed-mean ({len(a.seeds)} seeds): n={ok.sum()} MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} '
        f'R2 {g["R2"]:.3f} CCC {g["CCC"]:.3f} Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(out_dir, 'test_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + [f'pred_s{s}' for s in a.seeds] + ['pred_mean'])
        for i, u in enumerate(uids):
            w.writerow([u, groups[i], fold[i], float(y[i])]
                       + ['' if np.isnan(preds[s][i]) else round(preds[s][i], 3) for s in a.seeds]
                       + ['' if np.isnan(pm[i]) else round(pm[i], 3)])
    with open(os.path.join(out_dir, 'frame_preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'seed', 'frame_idx', 'y', 'pred'])
        w.writerows(frame_rows)
    log(f'done -> {out_dir}')


if __name__ == '__main__':
    main()
