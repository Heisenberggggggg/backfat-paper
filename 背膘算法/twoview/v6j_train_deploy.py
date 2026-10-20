# -*- coding: utf-8 -*-
"""
v6j_train_deploy.py — 训练部署用的解剖引导网络权重（全部 663 条记录，固定 epoch），供 Jetson 测速包使用

来源 / 背景
  10 交接 §6 第 4 项、12 进度汇报 §2：Jetson 实测要用新主模型（v6f prof），但 v6f / v6g 只保存了预测、没有保存权重。
  本脚本只产出「部署权重」：测速结果（时延 / 显存 / 功耗）与精度无关；**论文中的所有精度数字仍以 v6f 交叉验证为准**，
  本权重在训练集上的拟合度不报告。（前缀 v7 已被 B 场外测脚本占用，故用 v6j。）
做法（与 v6f prof 完全相同的数据、帧质控、横截面缓存、结构与超参，只是不再留出验证 / 测试折）
  - 训练集 = 全部 663 条记录的全部质控帧；种子 42
  - epoch：v6f prof 5 种子 × 5 折中按内层验证集选出的 epoch 中位数 = 12（4–15）。余弦调度仍按 15 轮设定，
    训练到第 12 轮停止，使部署模型的训练过程与交叉验证中被选中的模型一致
  - 保存 state_dict（加载时 ResNet-18 主干不需要下载 ImageNet 权重）、init_bias、F_POS、H_MAX 等
输出：runs/v6j_deploy/{anatomy_prof_deploy.pth, log.txt}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6j_train_deploy.py
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v6f_anatomy_net import (DM_HMAP, F_POS, H_MAX, HALF, AnatomyNet, get_profiles,  # noqa: E402
                             load_hmaps_paths, read_setup, to_dev)

OUT = os.path.join(HERE, 'runs', 'v6j_deploy')
SEED_CKPT = os.path.join(HERE, 'runs', 'v6_anatomy', 'prof', '_seed_ckpt')


def selected_epochs():
    eps = []
    for s in (42, 43, 44, 45, 46):
        st = np.load(os.path.join(SEED_CKPT, f'seed{s}.npz'), allow_pickle=True)['store'].item()
        eps += (np.argmin(st['val_mae'], 1) + 1).tolist()
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sched-epochs', type=int, default=15)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    eps = selected_epochs()
    n_ep = int(np.median(eps))
    t0 = time.time()
    uids, groups, fold, y = read_setup()
    X, owner, paths = load_hmaps_paths(uids, DM_HMAP, 0.8, 0.85)
    P, V = get_profiles(X, paths, log)
    log(f'[{time.strftime("%F %T")}] v6j deploy weights | records {len(uids)}, frames {len(X)}, seed {a.seed}, '
        f'epochs {n_ep} (median of inner-selected epochs {sorted(eps)}), cosine T_max {a.sched_epochs}')

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    init_bias = float(y.mean())
    model = AnatomyNet(init_bias, use_prof=True, use_mil=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.sched_epochs)
    scaler = torch.amp.GradScaler('cuda')
    fidx = np.arange(len(X))
    steps = int(np.ceil(len(fidx) / a.bs))
    for ep in range(n_ep):
        model.train()
        np.random.shuffle(fidx)
        tot = 0.0
        for st in range(steps):
            b = fidx[st * a.bs:(st + 1) * a.bs]
            yb = torch.from_numpy(y[owner[b]]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out, _ = model.forward_frames(*to_dev(X, P, V, b, dev, True))
            loss = F.huber_loss(out.float(), yb, delta=1.0)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(b)
        sched.step()
        log(f'  ep {ep + 1:02d} train Huber {tot / len(fidx):.3f} | {(time.time() - t0) / 60:.1f} min')

    path = os.path.join(OUT, 'anatomy_prof_deploy.pth')
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, init_bias=init_bias,
                    f_pos=F_POS.tolist(), h_max=H_MAX, half=HALF, epochs=n_ep, seed=a.seed,
                    n_records=len(uids), n_frames=len(X),
                    note='deployment weights for timing only; accuracy is reported from v6f cross-validation'),
               path)
    log(f'saved -> {path} ({os.path.getsize(path) / 1e6:.1f} MB) | {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
