# -*- coding: utf-8 -*-
"""
v6s_backbones.py — 高度图上换骨干网络的对照实验（回答「R² 低是不是 ResNet-18 不好、换新模型会不会更好」）

背景：本人 2026-10-21 要求在实验室补跑（DECISIONS 2026-10-21）。主文的表示 / 视角消融都用 ResNet-18 基线
  （runs_v6o/v4_hmap/hmap，3 种子 R² 0.336），审稿人可能问「为什么用 ResNet-18」。
做法：与 v4b_hmap_models.py 完全同一协议——同一套外层 5 折、同一内层验证集（选 epoch）、同样的帧质控、输入
  （128 × 256 高度图 / 1200 mm）、增广、Huber 损失、AdamW 3e-4、15 epoch 余弦、batch 64、种子 42–44——
  只把网络换成：ResNet-50、EfficientNet-B0、ConvNeXt-Tiny、Swin-T（均为 torchvision 的 ImageNet 预训练权重；
  第一层卷积按通道取平均改为单通道，输出层初始化为训练集均值，与 Net 相同）。不为各结构单独调参（固定配方），
  这一点要在稿件里写明。
实现：把 v4b 模块里的全局 Net 换成对应结构后调用 v4b.main()，所以训练、选模、断点续跑、输出格式与 v4b 完全一致。
输出：<runs>/v4_hmap/bb_<arch>/{log.txt, result.json, test_preds.csv, frame_preds.csv, _seed_ckpt/}
      <runs>/v6s_backbones/{log.txt, result.json}（--compare：各结构 vs ResNet-18，按猪 group bootstrap + Wilcoxon）
权重：默认从 torch hub 缓存读；实验室访问 download.pytorch.org 很慢，可用 --weights-dir 指向预先拷好的 .pth 目录。
用法（在 twoview/ 下，先设 BACKFAT_RUNS=<...>\\runs_v6o 与 BACKFAT_INDEX=<...>\\runs_v6o\\index_v2_v6o.csv）：
  python v6s_backbones.py --arch resnet50 efficientnet_b0 convnext_tiny swin_t --weights-dir D:\\backfat_weights
  python v6s_backbones.py --compare
  冒烟测试：--epochs 1 --seeds 42 --no-pretrained（只检查结构能跑通；须用临时 runs 副本）
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
import torchvision.models as tm

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v6o_paths import RUNS_ROOT  # noqa: E402
import v4b_hmap_models as v4b  # noqa: E402

# 结构 → (构造函数, 权重枚举, 第一层卷积的路径, 输出层的路径)
ARCH = {
    'resnet18': (tm.resnet18, tm.ResNet18_Weights.IMAGENET1K_V1, 'conv1', 'fc'),
    'resnet50': (tm.resnet50, tm.ResNet50_Weights.IMAGENET1K_V2, 'conv1', 'fc'),
    'efficientnet_b0': (tm.efficientnet_b0, tm.EfficientNet_B0_Weights.IMAGENET1K_V1, 'features.0.0', 'classifier.1'),
    'convnext_tiny': (tm.convnext_tiny, tm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 'features.0.0', 'classifier.2'),
    'swin_t': (tm.swin_t, tm.Swin_T_Weights.IMAGENET1K_V1, 'features.0.0', 'head'),
}


def _get(m, path):
    for p in path.split('.'):
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def _set(m, path, new):
    *head, last = path.split('.')
    parent = _get(m, '.'.join(head)) if head else m
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def make_net_class(arch, weights_dir=None, pretrained=True):
    ctor, wenum, first, last = ARCH[arch]

    class BackboneNet(nn.Module):
        def __init__(self, init_bias):
            super().__init__()
            r = ctor(weights=None)
            if pretrained:
                local = os.path.join(weights_dir, os.path.basename(wenum.url)) if weights_dir else None
                sd = (torch.load(local, map_location='cpu', weights_only=True) if local and os.path.isfile(local)
                      else wenum.get_state_dict(progress=False))
                r.load_state_dict(sd)
            c = _get(r, first)
            conv = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=c.bias is not None)
            conv.weight = nn.Parameter(c.weight.detach().mean(1, keepdim=True))     # 3 通道 → 单通道：权重取平均
            if c.bias is not None:
                conv.bias = nn.Parameter(c.bias.detach().clone())
            _set(r, first, conv)
            fc = _get(r, last)
            head = nn.Linear(fc.in_features, 1)
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, init_bias)                                 # 输出层初始化为训练集均值
            _set(r, last, head)
            self.r = r

        def forward(self, x):
            return self.r(x).squeeze(1)

    BackboneNet.__name__ = f'Net_{arch}'
    return BackboneNet


def train(arch, a):
    v4b.Net = make_net_class(arch, a.weights_dir, not a.no_pretrained)
    n_par = sum(p.numel() for p in v4b.Net(17.5).parameters())
    print(f'== {arch}: {n_par / 1e6:.1f} M parameters', flush=True)
    argv = ['v4b', '--tag', f'bb_{arch}', '--epochs', str(a.epochs), '--seeds', *map(str, a.seeds)]
    old = sys.argv
    sys.argv = argv
    try:
        v4b.main()
    finally:
        sys.argv = old
    return n_par


def compare(a):
    from scipy.stats import wilcoxon
    from cv_depthmap_labelfix import metrics
    from v1_tabular_baseline import group_bootstrap

    out = os.path.join(RUNS_ROOT, 'v6s_backbones')
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    def read(tag):
        p = os.path.join(RUNS_ROOT, 'v4_hmap', tag, 'test_preds.csv')
        with open(p, encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))

    ref = read('hmap')                                    # v4b ResNet-18，3 种子（主文表 2 的「Height map, ResNet-18」行）
    uid = [r['uid'] for r in ref]
    y = np.array([float(r['y']) for r in ref])
    grp = np.array([r['group'] for r in ref])
    pref = np.array([float(r['pred_mean']) for r in ref])
    tab = {r['uid']: r for r in csv.DictReader(open(os.path.join(RUNS_ROOT, 'v1_tabular', 'preds.csv'), encoding='utf-8-sig'))}
    line = np.array([tab[u]['line'] for u in uid])
    par = np.array([float(tab[u]['parity']) if tab[u]['parity'] else np.nan for u in uid])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (par == 1),
               'NAM_multiparous': (line == 'NAM') & (par >= 2)}
    log(f'[{time.strftime("%F %T")}] v6s backbone comparison (reference: ResNet-18, runs/v4_hmap/hmap)')
    res = {'reference': {s: metrics(y[m], pref[m]) for s, m in subsets.items()}, 'models': {}}
    g = res['reference']['all']
    log(f'   resnet18 (reference)  R2 {g["R2"]:.3f}  MAE {g["MAE"]:.3f}')
    for arch in [k for k in ARCH if k != 'resnet18']:
        tag = f'bb_{arch}'
        if not os.path.isfile(os.path.join(RUNS_ROOT, 'v4_hmap', tag, 'test_preds.csv')):
            log(f'   {arch}: not run')
            continue
        rows = read(tag)
        assert [r['uid'] for r in rows] == uid and [r['fold'] for r in rows] == [r['fold'] for r in ref], '折或猪不一致'
        p = np.array([float(r['pred_mean']) for r in rows])
        seeds = [c for c in rows[0] if c.startswith('pred_s')]
        per_seed = [metrics(y, np.array([float(r[c]) for r in rows]))['R2'] for c in seeds]
        res['models'][arch] = {'per_seed_R2': per_seed}
        for s, m in subsets.items():
            mm = metrics(y[m], p[m])
            b = group_bootstrap(grp[m], y[m], p[m], pref[m])
            pw = float(wilcoxon(np.abs(p[m] - y[m]), np.abs(pref[m] - y[m])).pvalue)
            res['models'][arch][s] = dict(**mm, **b, wilcoxon_p=pw)
            log(f'   {arch:16s} [{s:15s}] R2 {mm["R2"]:.3f} MAE {mm["MAE"]:.3f} | vs ResNet-18 ΔR² {b["dR2"][1]:+.3f} '
                f'({b["dR2"][0]:+.3f}, {b["dR2"][2]:+.3f})  ΔMAE {b["dMAE"][1]:+.3f} ({b["dMAE"][0]:+.3f}, '
                f'{b["dMAE"][2]:+.3f})  Wilcoxon p={pw:.3g}')
        log(f'   {arch:16s} per-seed R² ' + ', '.join(f'{v:.3f}' for v in per_seed))
    with open(os.path.join(out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', nargs='+', choices=list(ARCH), default=[])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--weights-dir', default=None)
    ap.add_argument('--no-pretrained', action='store_true', help='冒烟测试用：不加载预训练权重')
    ap.add_argument('--compare', action='store_true')
    a = ap.parse_args()
    for arch in a.arch:
        t0 = time.time()
        train(arch, a)
        print(f'== {arch} done in {(time.time() - t0) / 60:.1f} min', flush=True)
    if a.compare:
        compare(a)


if __name__ == '__main__':
    main()
