# -*- coding: utf-8 -*-
"""
v6k_make_edge_package.py — 生成 Jetson 测速包（新主模型）：示例猪原始帧 + 部署权重 + 分割模型 + PC 离线参照预测

来源 / 背景
  12 进度汇报 §2「必须」项：Jetson 实测新管线（含 CPU 横截面提取）。测速代码在 edge_deploy/v6k_anatomy_edge/
  （anatomy_edge.py 自包含管线、v6k_edge_profile.py 测速）；本脚本只往该目录放数据与权重，并写离线参照预测。
示例猪（预先设定的规则，非挑结果）：5 个测定日 8.24 / 8.26 / 10.29 / 11.1 / 11.5 各取 uid 排序第一头（A 场 663 条之内）；
  每头拷贝与训练数据相同的 20 帧（generate_depthmap_dataset.get_frame_pairs 均匀抽帧）+ master/intrinsic.json。
离线参照：同一部署权重（runs/v6j_deploy）在离线高度图（hmap_dataset，帧质控同 v4b）上的猪级预测（FP32），
  用于检查 Jetson 上的端到端管线是否算出相同结果；**不是精度**。
输出：edge_deploy/v6k_anatomy_edge/{data/<uid>/{color,depth,intrinsic.json}, weights/*, reference_preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6k_make_edge_package.py [--drive G:H]
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
SEG_DIR = os.path.normpath(os.path.join(HERE, '..', 'yolo+FastSAM'))
PKG = os.path.normpath(os.path.join(HERE, '..', 'edge_deploy', 'v6k_anatomy_edge'))
sys.path.insert(0, HERE)
sys.path.insert(0, SEG_DIR)
sys.path.insert(0, PKG)
import anatomy_edge as AE  # noqa: E402
import generate_depthmap_dataset as G  # noqa: E402
from v6f_anatomy_net import DM_HMAP, load_hmaps_paths  # noqa: E402

DATES = ['8.24', '8.26', '10.29', '11.1', '11.5']
WEIGHTS = os.path.join(HERE, 'runs', 'v6j_deploy', 'anatomy_prof_deploy.pth')
PREDS = os.path.join(HERE, 'runs', 'v6_anatomy', 'prof', 'test_preds.csv')
INDEX = os.path.join(HERE, 'index_v2.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drive', default='G:H', help='原始数据盘符映射（index_v2 记录为 G 盘，副本在 H 盘）')
    a = ap.parse_args()
    src, dst = a.drive.split(':')
    y = {r['uid']: float(r['y']) for r in csv.DictReader(open(PREDS, encoding='utf-8-sig'))}
    idx = {r['uid']: r for r in csv.DictReader(open(INDEX, encoding='utf-8-sig'))}
    uids = [sorted(u for u in y if u.split('_')[1] == d)[0] for d in DATES]
    print('sows:', uids)

    os.makedirs(os.path.join(PKG, 'weights'), exist_ok=True)
    for s, d in ((WEIGHTS, 'anatomy_prof_deploy.pth'), (os.path.join(SEG_DIR, 'yolo11n.pt'), 'yolo11n.pt'),
                 (os.path.join(SEG_DIR, 'FastSAM-s.pt'), 'FastSAM-s.pt')):
        shutil.copy2(s, os.path.join(PKG, 'weights', d))

    for u in uids:
        md = idx[u]['master_dir']
        if md[:2].upper() == f'{src.upper()}:':
            md = f'{dst.upper()}:' + md[2:]
        out = os.path.join(PKG, 'data', u)
        for sub in ('color', 'depth'):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        pairs = G.get_frame_pairs(os.path.join(md, 'color'), os.path.join(md, 'depth'), G.FRAMES_PER_PIG)
        for cp, dp in pairs:
            shutil.copy2(cp, os.path.join(out, 'color', os.path.basename(cp)))
            shutil.copy2(dp, os.path.join(out, 'depth', os.path.basename(dp)))
        shutil.copy2(os.path.join(md, 'intrinsic.json'), os.path.join(out, 'intrinsic.json'))
        print(f'  {u}: {len(pairs)} frames <- {md}')

    # 离线参照（FP32）
    X, owner, _ = load_hmaps_paths(uids, DM_HMAP, AE.AREA_REL, AE.LEN_REL)
    model, _ = AE.build_model(WEIGHTS, 'cuda')
    rows = []
    for i, u in enumerate(uids):
        fr = np.where(owner == i)[0]
        P, V = zip(*[AE.frame_profiles(X[j].astype(np.float32) * AE.H_MAX) for j in fr])
        with torch.no_grad():
            out, _ = model(torch.from_numpy(X[fr].astype(np.float32)).unsqueeze(1).cuda(),
                           torch.from_numpy(np.stack(P)).cuda(), torch.from_numpy(np.stack(V)).cuda())
        rows.append(dict(uid=u, y=y[u], n_qc_offline=len(fr), pred_offline=round(float(out.mean().cpu()), 4)))
        print(f'  reference {u}: {len(fr)} QC frames, pred {rows[-1]["pred_offline"]:.2f} (ultrasound {y[u]:.0f})')
    with open(os.path.join(PKG, 'reference_preds.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('package ->', PKG)


if __name__ == '__main__':
    main()
