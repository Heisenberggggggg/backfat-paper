# -*- coding: utf-8 -*-
"""
为ABC对比实验生成数据（不覆盖原有processed_pointclouds）

输出到: yolo+FastSAM/abc_data/
  depth_{pig_id}.npy   - (N, 224, 224) 掩膜裁剪深度图 (路线A)
  pc_{pig_id}.npy      - (N, 1024, 3)  不归一化点云，仅中心化 (路线B改进)
  meta_{pig_id}.npy    - (N, K)        全局特征 (路线B/C辅助)

全局特征(meta): [bbox_w, bbox_h, bbox_d, centroid_z, n_raw_pts,
                 max_radius, masked_area_ratio, volume]
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
import cv2
import open3d as o3d
import torch
from ultralytics import YOLO, FastSAM
from scipy.spatial import cKDTree
import time
import gc

# ============================================================
# 配置
# ============================================================
BACKFAT_LABELS = {
    '2-1': 16.1, '2-2': 16.0, '2-3': 21.5, '2-5': 14.6,
    '2-8': 15.2, '2-17': 18.8, '2-18': 19.6, '2-22': 17.9,
    '2-25': 13.8, '2-26': 19.8, '2-27': 13.5, '2-28': 12.8,
    '2-29': 17.7, '2-30': 18.1, '2-31': 18.2, '2-32': 18.5,
    '2-33': 13.8, '2-34': 18.8,
    '3-1': 19.0, '3-2': 15.1, '3-3': 17.2, '3-4': 14.1,
    '3-5': 14.5, '3-6': 18.4, '3-7': 13.8, '3-8': 14.1,
    '3-9': 14.5, '3-10': 16.0, '3-11': 16.8, '3-12': 17.2,
    '3-15': 18.8, '3-16': 13.7, '3-17': 10.0, '3-18': 13.9,
    '3-19': 17.6, '3-21': 18.0, '3-22': 20.2, '3-23': 22.0,
    '3-24': 17.1,
}

BASE_DIR = r'G:\耿来硕数据\DK\nkyfirst'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'abc_data')
os.makedirs(OUT_DIR, exist_ok=True)

NKY_INTRINSICS = {
    'fx': 614.84063720703125, 'fy': 614.61328125,
    'cx': 642.5406494140625, 'cy': 364.29312133789062
}

DEPTH_SCALE = 1000.0
MIN_DEPTH = 0.3
MAX_DEPTH = 3.0
NUM_POINTS = 1024
FRAMES_PER_PIG = 30
DEPTH_IMG_SIZE = 224

STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 2.0

SEED = 42
np.random.seed(SEED)


# ============================================================
# 工具函数
# ============================================================
def imread_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def imread_depth_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def get_mask(img, detect_model, segment_model):
    h, w = img.shape[:2]
    det = detect_model.predict(img, conf=0.15, device='cuda', verbose=False)[0]
    if len(det.boxes) == 0:
        m = 0.05
        best_box = np.array([w * m, h * m, w * (1 - m), h * (1 - m)])
    else:
        areas = (det.boxes.xywh[:, 2] * det.boxes.xywh[:, 3]).cpu().numpy()
        best_box = det.boxes.xyxy[np.argmax(areas)].cpu().numpy()
    seg = segment_model.predict(
        img, bboxes=best_box, device='cuda', retina_masks=True, conf=0.4, verbose=False
    )[0]
    if seg.masks is not None:
        mask = seg.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != (720, 1280):
            mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
        return mask
    return None


def depth_outlier_filter(points, k=3.0):
    z = points[:, 2]
    median_z = np.median(z)
    mad = np.median(np.abs(z - median_z))
    if mad < 1e-6:
        return points
    lower = median_z - k * 1.4826 * mad
    upper = median_z + k * 1.4826 * mad
    mask = (z >= lower) & (z <= upper)
    kept = points[mask]
    return kept if len(kept) > 100 else points


def fps_downsample(points, n_target, max_input=5000):
    """最远点采样，先随机预降采样到max_input再FPS"""
    if len(points) <= n_target:
        idx = np.random.choice(len(points), n_target, replace=True)
        return points[idx]

    # 如果点数太多，先随机降采样到 max_input
    if len(points) > max_input:
        idx = np.random.choice(len(points), max_input, replace=False)
        points = points[idx]

    selected = np.zeros(n_target, dtype=int)
    selected[0] = np.random.randint(len(points))
    dists = np.full(len(points), np.inf)
    for i in range(1, n_target):
        p = points[selected[i - 1]]
        d = np.sum((points - p) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return points[selected]


def make_masked_depth_image(depth_img, mask, size=DEPTH_IMG_SIZE):
    """生成掩膜裁剪的深度图 (224x224)，保留原始深度值(米)"""
    # 应用掩膜
    depth_m = depth_img.astype(np.float32) / DEPTH_SCALE
    depth_masked = depth_m * mask

    # 找掩膜的bounding box
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, 0.0
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # 正方形裁剪 (取较大边)
    h, w = y2 - y1, x2 - x1
    side = max(h, w)
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    y1c = max(0, cy - side // 2)
    x1c = max(0, cx - side // 2)
    y2c = min(720, y1c + side)
    x2c = min(1280, x1c + side)

    crop = depth_masked[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, 0.0

    # resize到目标大小
    img_out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    # 计算掩膜面积占比
    mask_crop = mask[y1c:y2c, x1c:x2c]
    area_ratio = np.sum(mask_crop > 0) / (mask_crop.size + 1e-8)

    return img_out, area_ratio


def process_frame(color_path, depth_path, intr, detect_model, segment_model):
    """处理单帧，返回 (depth_img_224, pc_1024, meta_features) 或 None"""
    color = imread_chinese(color_path)
    depth = imread_depth_chinese(depth_path)
    if color is None or depth is None:
        return None

    # 1. FastSAM 分割
    mask = get_mask(color, detect_model, segment_model)
    if mask is None:
        return None

    # 2. 掩膜深度图 (路线A)
    depth_img, area_ratio = make_masked_depth_image(depth, mask)
    if depth_img is None:
        return None

    # 3. 点云生成 (路线B)
    v, u = np.where(mask == 1)
    z = depth[v, u].astype(np.float32) / DEPTH_SCALE
    valid = (z > MIN_DEPTH) & (z < MAX_DEPTH)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - intr['cx']) * z / intr['fx']
    y = (v - intr['cy']) * z / intr['fy']
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    if len(points) < 200:
        return None

    n_raw = len(points)

    # 4. 深度过滤
    points = depth_outlier_filter(points)

    # 5. 统计滤波
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO)
    points = np.asarray(cl.points, dtype=np.float32)

    if len(points) < 200:
        return None

    # 6. FPS下采样到1024
    points = fps_downsample(points, NUM_POINTS)

    # 7. 仅中心化，不缩放！（保留绝对尺寸信息）
    centroid = np.mean(points, axis=0)
    centroid_z = centroid[2]  # 记录原始质心深度
    points_centered = points - centroid

    # 8. 计算全局元数据
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_dims = bbox_max - bbox_min  # (W, H, D) in meters
    max_radius = np.max(np.linalg.norm(points_centered, axis=1))

    meta = np.array([
        bbox_dims[0],     # bbox_w (米)
        bbox_dims[1],     # bbox_h
        bbox_dims[2],     # bbox_d (深度范围)
        centroid_z,       # 质心深度 (离相机距离)
        float(n_raw),     # 原始点数 (FastSAM出来的)
        max_radius,       # 最大半径 (归一化前)
        area_ratio,       # 掩膜面积占比
        np.prod(bbox_dims),  # 体积
    ], dtype=np.float32)

    return depth_img, points_centered, meta


# ============================================================
# 主流程
# ============================================================
def find_pig_dir(pig_id):
    for name in [pig_id, f'c{pig_id}']:
        p = os.path.join(BASE_DIR, name, 'depth_cz_r_d_p', 'master')
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'color')):
            return p
    return None


def main():
    print("=" * 60)
    print("  生成ABC实验数据")
    print(f"  输出目录: {OUT_DIR}")
    print(f"  每头猪采样: {FRAMES_PER_PIG} 帧")
    print("=" * 60)

    # 加载模型
    print("\n加载 YOLO + FastSAM...")
    detect_model = YOLO('yolo11n.pt')
    segment_model = FastSAM('FastSAM-s.pt')
    print("模型加载完成!")

    intr = NKY_INTRINSICS
    total_success = 0
    total_fail = 0

    for pig_idx, (pid, bf) in enumerate(sorted(BACKFAT_LABELS.items())):
        master_dir = find_pig_dir(pid)
        if master_dir is None:
            print(f"\n[{pig_idx+1}/39] {pid}: 目录不存在，跳过")
            continue

        color_dir = os.path.join(master_dir, 'color')
        depth_dir = os.path.join(master_dir, 'depth')

        all_frames = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
        if len(all_frames) > FRAMES_PER_PIG:
            sel = np.random.choice(len(all_frames), FRAMES_PER_PIG, replace=False)
            all_frames = [all_frames[i] for i in sorted(sel)]

        print(f"\n[{pig_idx+1}/39] {pid} (背膘={bf}mm): {len(all_frames)}帧", end="")

        depths_out = []
        pcs_out = []
        metas_out = []

        for fname in all_frames:
            stem = os.path.splitext(fname)[0]
            cpath = os.path.join(color_dir, fname)
            dpath = os.path.join(depth_dir, f"{stem}.png")
            if not os.path.exists(dpath):
                continue

            result = process_frame(cpath, dpath, intr, detect_model, segment_model)
            if result is None:
                total_fail += 1
                continue

            d_img, pc, meta = result
            depths_out.append(d_img)
            pcs_out.append(pc)
            metas_out.append(meta)
            total_success += 1

        if not depths_out:
            print(" - 失败!")
            continue

        # 保存
        np.save(os.path.join(OUT_DIR, f'depth_{pid}.npy'),
                np.array(depths_out, dtype=np.float32))
        np.save(os.path.join(OUT_DIR, f'pc_{pid}.npy'),
                np.array(pcs_out, dtype=np.float32))
        np.save(os.path.join(OUT_DIR, f'meta_{pid}.npy'),
                np.array(metas_out, dtype=np.float32))

        print(f" → {len(depths_out)} 帧成功")

        # 清理显存
        if (pig_idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n{'=' * 60}")
    print(f"  完成! 成功: {total_success}, 失败: {total_fail}")
    print(f"  数据保存在: {OUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
