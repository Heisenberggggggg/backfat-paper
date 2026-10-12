# -*- coding: utf-8 -*-
"""
a3_extract_foundation_features.py — A3：冻结视觉基础模型（DINOv2 / DINOv3）提取俯视深度图特征

来源 / 背景
  05 §8.2 A3、06 §5.1 第 6 项。视觉 ResNet-18 需要训练且无法与表格变量无泄漏地融合；冻结基础模型
  特征不需训练，可与生产记录、卡尺一起直接进 TabPFN（a3_foundation_regression.py），用于回答
  「记录 + 相机 vs 记录 + 卡尺」「记录 + 卡尺 + 相机是否还有增量」。

输入
  - 猪：runs/v1_tabular/preds.csv 的 663 头（与 ResNet-18 / V1 相同）
  - 帧：yolo+FastSAM/depthmap_dataset（每帧 256×256 float32，相机到猪背距离 mm，背景 0；
        按检测框裁剪后等比缩放 → 绝对尺寸已丢失；边缘有与背景插值产生的伪低值）

深度 → 3 通道编码（--encodings）
  relh    相对高度：有效像素 d ≥ 300 mm；top = 有效深度 2% 分位（背部最高处）；
          v = 1 − clip((d − top) / 400 mm, 0, 1)，灰度复制 3 通道，背景 0
  normal  表面法向：背景填 top + 400 mm，高斯平滑 σ=1.5 px；按名义像素间距 6 mm/px 求梯度
          （绝对尺度已丢失，只能近似，法向仍保留曲率的相对信息）；n = (−gx, −gy, 1)/‖·‖ 映射到 [0,1]，背景 0
  两者都缩放到 224×224，按 ImageNet 均值/方差归一化（与两模型的预处理配置一致）。

特征
  每帧：CLS 向量 + 猪体区域 patch token 的均值（patch 内掩膜占比 > 0.5），拼接 → 2 × hidden；
  DINOv3 的 register token 不参与。猪级 = 该猪全部帧均值。fp16 推理。

输出
  runs/a3_foundation/feat_<model>_<enc>.npz：uids, owner(帧→猪下标), frame_feat(float16), pig_feat(float32)
  （*.npz 不进 git）

用法
  set HF_HUB_DISABLE_XET=1
  D:\\Anaconda\\envs\\pytorch\\python.exe a3_extract_foundation_features.py --model dinov3-vitb16
  D:\\Anaconda\\envs\\pytorch\\python.exe a3_extract_foundation_features.py --model dinov2-base
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
PIGS = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
DM_DIR = os.path.join(HERE, '..', 'yolo+FastSAM', 'depthmap_dataset')
OUT = os.path.join(HERE, 'runs', 'a3_foundation')
MODELS = {'dinov3-vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
          'dinov2-base': 'facebook/dinov2-base'}
MIN_D, REL_RANGE, PX_MM, SIZE = 300.0, 400.0, 6.0, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def encode(d, enc):
    m = d >= MIN_D
    if m.sum() < 50:
        return None, None
    top = np.percentile(d[m], 2)
    if enc == 'relh':
        v = 1.0 - np.clip((d - top) / REL_RANGE, 0, 1)
        img = np.repeat(np.where(m, v, 0)[..., None], 3, axis=2).astype(np.float32)
    elif enc == 'normal':
        z = np.where(m, d, top + REL_RANGE).astype(np.float32)
        z = cv2.GaussianBlur(z, (0, 0), 1.5)
        gy, gx = np.gradient(z / PX_MM)
        n = np.stack([-gx, -gy, np.ones_like(z)], axis=2)
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        img = np.where(m[..., None], (n + 1) / 2, 0).astype(np.float32)
    else:
        raise ValueError(enc)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    mk = cv2.resize(m.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return (img - MEAN) / STD, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS), required=True)
    ap.add_argument('--encodings', default='relh,normal')
    ap.add_argument('--bs', type=int, default=64)
    # 2026-10-11 V7：--pigs dk 提取省农科院 DK 159 头（index_v2 farm=DK；metadata pig_id 形如 nky1_2-1），
    #               输出 feat_<model>_<enc>_dk.npz，不覆盖丰美禾特征
    ap.add_argument('--pigs', choices=['fmh', 'dk'], default='fmh')
    a = ap.parse_args()
    from transformers import AutoModel
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'extract_log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    frames = {}
    with open(os.path.join(DM_DIR, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            frames.setdefault(r['pig_id'], []).append(r['frame_path'])
    if a.pigs == 'fmh':
        with open(PIGS, encoding='utf-8-sig') as f:
            uids = [r['uid'] for r in csv.DictReader(f)]
    else:
        pre = {'nky1': 'dk_nkyfirst_', 'nky2': 'dk_nkysecond_', 'nky3': 'dk_nkythird_'}
        meta2uid = {p: pre[p.split('_')[0]] + p.split('_', 1)[1] for p in frames if p.split('_')[0] in pre}
        with open(os.path.join(HERE, 'index_v2.csv'), encoding='utf-8-sig') as f:
            uids = sorted(r['uid'] for r in csv.DictReader(f) if r['farm'] == 'DK')
        uid2meta = {v: k for k, v in meta2uid.items()}
        frames = {u: frames[uid2meta[u]] for u in uids}
    dev = 'cuda'
    model = AutoModel.from_pretrained(MODELS[a.model], torch_dtype=torch.float16).to(dev).eval()
    cfg = model.config
    patch = cfg.patch_size
    gp = SIZE // patch
    n_patch = gp * gp
    log(f'[{time.strftime("%F %T")}] model={a.model} ({MODELS[a.model]}) hidden={cfg.hidden_size} patch={patch} '
        f'registers={getattr(cfg, "num_register_tokens", 0)} pigs={len(uids)}')

    for enc in a.encodings.split(','):
        feats, owner, skipped = [], [], 0
        batch_x, batch_m, batch_o = [], [], []

        def flush():
            x = torch.from_numpy(np.stack(batch_x)).permute(0, 3, 1, 2).to(dev, torch.float16)
            with torch.no_grad():
                h = model(pixel_values=x).last_hidden_state.float()
            cls = h[:, 0]
            pt = h[:, -n_patch:]                                   # patch token 位于序列末尾（DINOv3 前有 register）
            pm = torch.from_numpy(np.stack(batch_m)).to(dev)
            pm = torch.nn.functional.avg_pool2d(pm[:, None], patch).flatten(1) > 0.5
            pm[pm.sum(1) == 0] = True
            pmean = (pt * pm[..., None]).sum(1) / pm.sum(1, keepdim=True)
            feats.append(torch.cat([cls, pmean], 1).cpu().numpy().astype(np.float16))
            owner.extend(batch_o)
            batch_x.clear(); batch_m.clear(); batch_o.clear()

        for i, u in enumerate(uids):
            for fp in frames[u]:
                img, mk = encode(np.load(os.path.join(DM_DIR, fp)), enc)
                if img is None:
                    skipped += 1
                    continue
                batch_x.append(img); batch_m.append(mk); batch_o.append(i)
                if len(batch_x) == a.bs:
                    flush()
        if batch_x:
            flush()
        ff = np.concatenate(feats)
        owner = np.array(owner)
        pig = np.stack([ff[owner == i].astype(np.float32).mean(0) for i in range(len(uids))])
        out = os.path.join(OUT, f'feat_{a.model}_{enc}' + ('' if a.pigs == 'fmh' else '_dk') + '.npz')
        np.savez_compressed(out, uids=np.array(uids), owner=owner, frame_feat=ff, pig_feat=pig)
        log(f'  enc={enc}: frames={len(ff)} skipped={skipped} dim={ff.shape[1]} '
            f'frames/pig min={np.bincount(owner).min()} -> {out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
