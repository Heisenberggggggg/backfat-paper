# -*- coding: utf-8 -*-
"""
v4f_extract_dinov3_hmap.py — 冻结 DINOv3 ViT-B/16 提取标准化高度图特征（供 A3 无泄漏堆叠融合）

来源 / 背景
  DECISIONS 2026-10-13：高度图 ResNet-18 场内 R² 0.378 > 俯视深度图 0.309。A3 融合（记录 + 卡尺 + 相机）需要
  无训练的冻结特征才能做无泄漏的嵌套堆叠（a3_foundation_regression.py --vis stack）。本脚本对高度图提取同样的特征，
  输出文件名与 a3_extract_foundation_features.py 一致的格式，直接复用 a3_foundation_regression.py。

与 a3_extract_foundation_features.py 的区别
  - 输入：yolo+FastSAM/hmap_dataset（128 × 256，离地高度 mm）；帧质控同 v4b（ok=1、面积 ≥ 本猪中位 80%、体长 ≥ 85%）
  - 编码 hmap：v = clip(h / 1200 mm, 0, 1)，灰度复制 3 通道，背景 0（保留绝对高度）；缩放到 224 × 448（保长宽比，
    14 × 28 = 392 个 patch），ImageNet 均值 / 方差归一化
  - 特征与原脚本相同：CLS + 猪体区域 patch token 均值（patch 内掩膜占比 > 0.5），猪级 = 帧均值；fp16 推理

输出：runs/a3_foundation/feat_dinov3-vitb16_hmap.npz（丰美禾 663 头，顺序同 runs/v1_tabular/preds.csv）
      runs/a3_foundation/feat_dinov3-vitb16_hmap_dk.npz（DK 159 头，uid 排序）
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4f_extract_dinov3_hmap.py
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v4b_hmap_models import DM_HMAP, H_MAX  # noqa: E402

PIGS = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
INDEX = os.path.join(HERE, 'index_v2.csv')
OUT = os.path.join(HERE, 'runs', 'a3_foundation')
MODEL = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
H_IN, W_IN = 224, 448
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def qc_frames(uids, area_rel=0.8, len_rel=0.85):
    by = {}
    with open(os.path.join(DM_HMAP, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ok'] == '1':
                by.setdefault(r['uid'], []).append(r)
    out = {}
    for u in uids:
        rs = by.get(u, [])
        if not rs:
            out[u] = []
            continue
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        out[u] = [r['frame_path'] for r in rs
                  if float(r['area_cm2']) >= area_rel * ma and float(r['length_mm']) >= len_rel * ml]
    return out


def encode(h):
    m = h > 0
    v = np.clip(h / H_MAX, 0, 1).astype(np.float32)
    img = np.repeat(np.where(m, v, 0)[..., None], 3, axis=2)
    img = cv2.resize(img, (W_IN, H_IN), interpolation=cv2.INTER_LINEAR)
    mk = cv2.resize(m.astype(np.float32), (W_IN, H_IN), interpolation=cv2.INTER_AREA)
    return (img - MEAN) / STD, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bs', type=int, default=48)
    a = ap.parse_args()
    from transformers import AutoModel
    logf = open(os.path.join(OUT, 'extract_log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    dev = 'cuda'
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    patch = model.config.patch_size
    n_patch = (H_IN // patch) * (W_IN // patch)
    log(f'[{time.strftime("%F %T")}] v4f hmap features: {MODEL} patch={patch} input {H_IN}x{W_IN} '
        f'({n_patch} patches) registers={getattr(model.config, "num_register_tokens", 0)}')
    with open(PIGS, encoding='utf-8-sig') as f:
        fmh = [r['uid'] for r in csv.DictReader(f)]
    with open(INDEX, encoding='utf-8-sig') as f:
        dk = sorted(r['uid'] for r in csv.DictReader(f) if r['farm'] == 'DK')
    for tag, uids in (('', fmh), ('_dk', dk)):
        frames = qc_frames(uids)
        feats, owner = [], []
        bx, bm, bo = [], [], []

        def flush():
            x = torch.from_numpy(np.stack(bx)).permute(0, 3, 1, 2).to(dev, torch.float16)
            with torch.no_grad():
                h = model(pixel_values=x).last_hidden_state.float()
            cls, pt = h[:, 0], h[:, -n_patch:]
            assert pt.shape[1] == n_patch
            pm = torch.from_numpy(np.stack(bm)).to(dev)
            pm = torch.nn.functional.avg_pool2d(pm[:, None], patch).flatten(1) > 0.5
            pm[pm.sum(1) == 0] = True
            pmean = (pt * pm[..., None]).sum(1) / pm.sum(1, keepdim=True)
            feats.append(torch.cat([cls, pmean], 1).cpu().numpy().astype(np.float16))
            owner.extend(bo)
            bx.clear(); bm.clear(); bo.clear()

        for i, u in enumerate(uids):
            for fp in frames[u]:
                img, mk = encode(np.load(os.path.join(DM_HMAP, fp)).astype(np.float32))
                bx.append(img); bm.append(mk); bo.append(i)
                if len(bx) == a.bs:
                    flush()
        if bx:
            flush()
        ff = np.concatenate(feats)
        owner = np.array(owner)
        missing = [uids[i] for i in range(len(uids)) if not np.any(owner == i)]
        pig = np.stack([ff[owner == i].astype(np.float32).mean(0) if np.any(owner == i)
                        else np.full(ff.shape[1], np.nan, np.float32) for i in range(len(uids))])
        out = os.path.join(OUT, f'feat_dinov3-vitb16_hmap{tag}.npz')
        np.savez_compressed(out, uids=np.array(uids), owner=owner, frame_feat=ff, pig_feat=pig)
        log(f'  {tag or "fmh"}: pigs {len(uids)} (missing {missing}) frames {len(ff)} dim {ff.shape[1]} -> {out} '
            f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
