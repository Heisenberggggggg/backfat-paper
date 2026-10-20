# -*- coding: utf-8 -*-
"""
v4b_hmap_models.py — 标准化高度图（v4a）上的 ResNet-18：与 V3 俯视同一协议、同一套 5 折、3 个种子

来源 / 背景
  v4a_gen_height_maps.py 生成保留绝对体尺的离地高度图（7.8125 mm/px，2000 × 1000 mm 窗口）。
  对照：V3 俯视（裁剪缩放的相机距离深度图，3 种子平均 R² 0.309，runs/v3_views/master_s128）。

与 v3b_view_models.py 的区别（其余完全相同：Net、Huber δ=1、AdamW 3e-4、15 epoch、余弦、batch 64、
  外层折取自 framepreds_new_s128、内层验证集 StratifiedGroupKFold(7, seed=42+k) 第 1 份、猪级 = 帧均值）
  - 输入：128 × 256 原生分辨率（不缩成正方形）；像素 = 离地高度 / 1200 mm，截断到 [0, 1]，背景 0
  - 增广：水平翻转与垂直翻转各 p = 0.5（高度图头尾方向、左右方向都不固定）
  - 帧质控（2026-10-12 演示数据测试后、训练前确定）：metadata ok = 1，且该帧俯视面积 ≥ 该猪中位面积的 80%、
    体长 ≥ 该猪中位体长的 85%（剔除只分割到部分猪体的帧）

输出：runs/v4_hmap/<tag>/{log.txt, result.json, test_preds.csv, frame_preds.csv, _seed_ckpt/}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4b_hmap_models.py
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
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import Net, metrics  # noqa: E402

DM_HMAP = os.path.join(HERE, '..', 'yolo+FastSAM', 'hmap_dataset')
FOLDS_SRC = os.path.join(RUNS_ROOT, 'framepreds_new_s128', 'test_preds.csv')
INDEX = INDEX_PATH
H_MAX = 1200.0


def load_hmaps(uids, dm_dir, area_rel, len_rel):
    pos = {u: i for i, u in enumerate(uids)}
    with open(os.path.join(dm_dir, 'metadata.csv'), encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['uid'] in pos and r['ok'] == '1']
    by = {}
    for r in rows:
        by.setdefault(r['uid'], []).append(r)
    keep, n_all = [], len(rows)
    for u, rs in by.items():
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        keep += [r for r in rs if float(r['area_cm2']) >= area_rel * ma and float(r['length_mm']) >= len_rel * ml]
    X = np.stack([np.clip(np.load(os.path.join(dm_dir, r['frame_path'])).astype(np.float32) / H_MAX, 0, 1)
                  .astype(np.float16) for r in keep])
    owner = np.array([pos[r['uid']] for r in keep])
    return X, owner, n_all


def predict(model, X, owner, pigs, dev, bs=256):
    model.eval()
    idx = np.where(np.isin(owner, pigs))[0]
    out = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            p = model(torch.from_numpy(X[b].astype(np.float32))[:, None].to(dev)).float().cpu().numpy()
            for o, v in zip(owner[b], p):
                out.setdefault(int(o), []).append(float(v))
    return out


def train_fold(X, owner, y, tr, va, a, dev, log):
    model = Net(float(y[tr].mean())).to(dev)
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
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.huber_loss(model(xb.to(dev)).float(), yb, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        pv = predict(model, X, owner, va, dev)
        vmae = np.mean([abs(np.mean(pv[i]) - y[i]) for i in va if i in pv])
        if vmae < best:
            best, best_state = vmae, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        log(f'    ep {ep + 1:02d} val MAE {vmae:.3f}{" *" if vmae == best else ""}')
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--area-rel', type=float, default=0.8)
    ap.add_argument('--len-rel', type=float, default=0.85)
    ap.add_argument('--dm', default=DM_HMAP)
    ap.add_argument('--tag', default='hmap')
    a = ap.parse_args()
    dev = 'cuda'
    out_dir = os.path.join(RUNS_ROOT, 'v4_hmap', a.tag)
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
    X, owner, n_all = load_hmaps(uids, a.dm, a.area_rel, a.len_rel)
    have = np.array([i in set(owner.tolist()) for i in range(len(uids))])
    log(f'[{time.strftime("%F %T")}] args={vars(a)}')
    log(f'   frames ok {n_all} → after QC {len(X)} ({len(X) / max(n_all, 1):.1%}); pigs {have.sum()}/{len(uids)} '
        f'(missing {[uids[i] for i in np.where(~have)[0]][:10]}); input {X.shape[1:]} (load {time.time() - t0:.0f}s)')

    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    preds = {s: np.full(len(uids), np.nan) for s in a.seeds}
    frame_rows = []
    ckpt_dir = os.path.join(out_dir, '_seed_ckpt')
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
            trv, te = np.where(fold != k + 1)[0], np.where(fold == k + 1)[0]
            inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42 + k)
            tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
            tr, va, te = [x[have[x]] for x in (trv[tr_i], trv[va_i], te)]
            log(f'  seed {s} fold {k + 1}: train {len(tr)} / val {len(va)} / test {len(te)}')
            model = train_fold(X, owner, y, tr, va, a, dev, log)
            pt = predict(model, X, owner, te, dev)
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
    res = {'args': vars(a), 'n': int(ok.sum()),
           'per_seed': {s: metrics(y[~np.isnan(preds[s])], preds[s][~np.isnan(preds[s])]) for s in a.seeds},
           'seed_mean': metrics(y[ok], pm[ok])}
    g = res['seed_mean']
    log(f'== hmap seed-mean: n={ok.sum()} MAE {g["MAE"]:.3f} RMSE {g["RMSE"]:.3f} R2 {g["R2"]:.3f} '
        f'CCC {g["CCC"]:.3f} Spearman {g["Spearman"]:.3f} | {(time.time() - t0) / 60:.1f} min')
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
