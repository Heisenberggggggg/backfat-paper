# -*- coding: utf-8 -*-
"""
==========================================================================
generate_pointcloud_v3_rear.py — 臀部视角（sub）掩膜点云批量生成
==========================================================================
背景：
    v2 generate_pointcloud_v2.py 仅处理主视角（master/背部），生成的点云用于
    单视角PointMLP/XGBoost回归背膘厚度，最终 R²≈0.08~0.15，效果有限。

问题分析：
    单一背部视角的点云信息量不足——背膘厚度是皮下脂肪，外表面形状差异微妙。
    臀部视角包含互补信息（臀部丰满度、骨骼突出程度），与体况评分高度相关。

解决思路：
    对所有数据源的臀部视角（sub/sub2/sub10）走同样的流程：
    color + depth → YOLO检测 → FastSAM分割 → 猪体掩膜点云
    最终与背部特征拼接（82维）做双视角早期融合回归。

数据源目录结构（臀部视角）：
    nkyfirst:  G:/耿来硕数据/DK/nkyfirst/{pig}/depth_cz_r_d_p/sub/   (25头)
               G:/耿来硕数据/DK/nkyfirst/{pig}/depth_cz_r_d_p/sub2/  (14头)
    nkysecond: G:/耿来硕数据/DK/nkysecond/{pig}/depth_cz_r_d_p/sub/   (7-X, 11头)
               G:/耿来硕数据/DK/nkysecond/{pig}/depth_cz_r_d_p/sub10/ (6-X, 21头)
    nkythird:  G:/耿来硕数据/DK/nkythird/{kinect}/{pig}/depth_cz_r_d_p/sub/
    丰美禾:    G:/耿来硕数据/丰美禾数据/{date}/{pig}/sub/

已知异常：
    - nkythird/12-242: sub/color 为空（0帧），自动跳过
    - nkythird/13-734: sub 仅147帧（正常180），可正常处理

输出：每头猪的臀部视角目录下新建 masked_pointclouds_v3_rear/ 文件夹

Author: 耿来硕 + Claude
Date: 2026-03-31
==========================================================================
"""

from ultralytics import YOLO, FastSAM
import cv2
import os
import gc
import json
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


# ============================================================
# 全局配置
# ============================================================
DEPTH_SCALE = 1000.0   # 深度值 mm → m
MIN_DEPTH = 0.3        # 最小有效深度 (m)
MAX_DEPTH = 3.0        # 最大有效深度 (m)
FRAMES_PER_PIG = 20    # 每头猪均匀抽取帧数
RESUME = True          # 跳过已生成的文件


# ============================================================
# 支持中文路径的图像读取
# ============================================================
def imread_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imread_depth_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


# ============================================================
# 从 intrinsic.json 读取相机内参
# ============================================================
def load_intrinsic(json_path):
    """
    读取 intrinsic.json，返回 fx, fy, cx, cy, width, height
    intrinsic_matrix 是 3x3 列主序存储：
    [fx, 0, 0, 0, fy, 0, cx, cy, 1]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    mat = data['intrinsic_matrix']
    fx = mat[0]
    fy = mat[4]
    cx = mat[6]
    cy = mat[7]
    width = data['width']
    height = data['height']

    return fx, fy, cx, cy, width, height


# ============================================================
# YOLO + FastSAM 掩膜提取
# ============================================================
def get_mask(detect_model, segment_model, img, height, width):
    """用 YOLO + FastSAM 获取猪体掩膜"""
    det_results = detect_model.predict(img, conf=0.15, device='cuda', verbose=False)[0]

    if len(det_results.boxes) == 0:
        h, w = img.shape[:2]
        margin = 0.05
        best_box = np.array([w * margin, h * margin, w * (1 - margin), h * (1 - margin)])
    else:
        areas = (det_results.boxes.xywh[:, 2] * det_results.boxes.xywh[:, 3]).cpu().numpy()
        max_idx = np.argmax(areas)
        best_box = det_results.boxes.xyxy[max_idx].cpu().numpy()

    seg_results = segment_model.predict(
        img, bboxes=best_box, device='cuda', retina_masks=True, conf=0.4, verbose=False
    )[0]

    if seg_results.masks is not None:
        mask = seg_results.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask
    return None


# ============================================================
# depth + mask → 3D 点云
# ============================================================
def depth_to_pointcloud(depth_img, mask, fx, fy, cx, cy):
    """将掩膜区域的 depth 反投影为 3D 点云（仅xyz，不含颜色）"""
    v, u = np.where(mask == 1)
    z = depth_img[v, u].astype(np.float32) / DEPTH_SCALE

    valid = (z > MIN_DEPTH) & (z < MAX_DEPTH)
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# ============================================================
# 收集所有猪的臀部视角目录信息
# ============================================================
def collect_all_pigs_rear():
    """
    遍历4个数据源，返回列表，每项为 dict:
    {
        'source': 数据源名称,
        'pig_id': 猪耳标,
        'base_dir': 臀部视角目录 (含 color/, depth/, intrinsic.json),
    }
    """
    pigs = []

    # --- nkyfirst ---
    # 25头有 sub，14头有 sub2（2-1也有sub2和pz）
    nky1_root = r"G:\耿来硕数据\DK\nkyfirst"
    if os.path.isdir(nky1_root):
        for pig in sorted(os.listdir(nky1_root)):
            pig_path = os.path.join(nky1_root, pig, "depth_cz_r_d_p")
            # 优先 sub，不存在则 sub2
            for subdir in ["sub", "sub2"]:
                base = os.path.join(pig_path, subdir)
                if os.path.isdir(base) and os.path.isdir(os.path.join(base, "color")):
                    pigs.append({'source': 'nkyfirst', 'pig_id': pig, 'base_dir': base})
                    break

    # --- nkysecond ---
    # 7-X系列用 sub，6-X系列用 sub10
    nky2_root = r"G:\耿来硕数据\DK\nkysecond"
    if os.path.isdir(nky2_root):
        for pig in sorted(os.listdir(nky2_root)):
            pig_path = os.path.join(nky2_root, pig, "depth_cz_r_d_p")
            # 优先 sub10，不存在则 sub
            for subdir in ["sub10", "sub"]:
                base = os.path.join(pig_path, subdir)
                if os.path.isdir(base) and os.path.isdir(os.path.join(base, "color")):
                    pigs.append({'source': 'nkysecond', 'pig_id': pig, 'base_dir': base})
                    break

    # --- nkythird ---
    nky3_root = r"G:\耿来硕数据\DK\nkythird"
    if os.path.isdir(nky3_root):
        for kinect_folder in sorted(os.listdir(nky3_root)):
            kinect_path = os.path.join(nky3_root, kinect_folder)
            if not os.path.isdir(kinect_path):
                continue
            for pig in sorted(os.listdir(kinect_path)):
                base = os.path.join(kinect_path, pig, "depth_cz_r_d_p", "sub")
                if os.path.isdir(base) and os.path.isdir(os.path.join(base, "color")):
                    # 跳过 sub/color 为空的情况（如 12-242）
                    color_dir = os.path.join(base, "color")
                    if len(os.listdir(color_dir)) == 0:
                        print(f"  [!] 跳过空目录: {color_dir}")
                        continue
                    pigs.append({'source': 'nkythird', 'pig_id': f"{kinect_folder}/{pig}", 'base_dir': base})

    # --- 丰美禾 (全部12个日期文件夹) ---
    fmh_base = r"G:\耿来硕数据\丰美禾数据"
    fmh_dates = ["8.24", "8.25", "8.26", "8.27",
                 "10.29", "10.30", "10.31",
                 "11.1", "11.2", "11.3", "11.4", "11.5"]
    for date_folder in fmh_dates:
        fmh_root = os.path.join(fmh_base, date_folder)
        if not os.path.isdir(fmh_root):
            continue
        for pig in sorted(os.listdir(fmh_root)):
            base = os.path.join(fmh_root, pig, "sub")
            if os.path.isdir(base) and os.path.isdir(os.path.join(base, "color")):
                pigs.append({'source': f'丰美禾/{date_folder}', 'pig_id': pig, 'base_dir': base})

    return pigs


# ============================================================
# 均匀抽帧
# ============================================================
def uniform_sample_frames(file_list, n_frames):
    """从 file_list 中均匀抽取 n_frames 个文件"""
    total = len(file_list)
    if total <= n_frames:
        return file_list
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [file_list[i] for i in indices]


# ============================================================
# 处理单头猪
# ============================================================
def process_one_pig(pig_info, detect_model, segment_model):
    """
    处理一头猪（臀部视角）：均匀抽20帧 → YOLO+FastSAM → 掩膜点云 → 保存到 masked_pointclouds_v3_rear/
    返回 (success_count, skip_count, fail_count)
    """
    base_dir = pig_info['base_dir']
    color_dir = os.path.join(base_dir, "color")
    depth_dir = os.path.join(base_dir, "depth")
    intrinsic_path = os.path.join(base_dir, "intrinsic.json")
    output_dir = os.path.join(base_dir, "masked_pointclouds_v3_rear")
    os.makedirs(output_dir, exist_ok=True)

    # 读取相机内参
    if not os.path.exists(intrinsic_path):
        print(f"  [!] intrinsic.json 不存在: {intrinsic_path}")
        return 0, 0, 0

    fx, fy, cx, cy, width, height = load_intrinsic(intrinsic_path)

    # 获取所有帧并均匀抽样
    all_colors = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    if not all_colors:
        # 有些目录可能用 .png
        all_colors = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
    selected = uniform_sample_frames(all_colors, FRAMES_PER_PIG)

    success, skip, fail = 0, 0, 0

    for fname in selected:
        stem = os.path.splitext(fname)[0]
        save_path = os.path.join(output_dir, f"pig_{stem}.ply")

        # 断点续传
        if RESUME and os.path.exists(save_path):
            skip += 1
            continue

        depth_path = os.path.join(depth_dir, f"{stem}.png")
        if not os.path.exists(depth_path):
            fail += 1
            continue

        color_img = imread_chinese(os.path.join(color_dir, fname))
        depth_img = imread_depth_chinese(depth_path)

        if color_img is None or depth_img is None:
            fail += 1
            continue

        # 获取掩膜
        mask = get_mask(detect_model, segment_model, color_img, height, width)
        if mask is None:
            fail += 1
            continue

        # 生成点云
        pcd = depth_to_pointcloud(depth_img, mask, fx, fy, cx, cy)

        if len(pcd.points) < 100:
            fail += 1
            del pcd
            continue

        # 保存
        o3d.io.write_point_cloud(save_path, pcd)
        success += 1

        # 释放内存
        del pcd, color_img, depth_img, mask

    gc.collect()
    torch.cuda.empty_cache()

    return success, skip, fail


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("臀部视角掩膜点云批量生成 v3 (rear/sub)")
    print("用途：双视角早期融合验证 — 与背部特征拼接做回归")
    print("=" * 60)

    # 收集所有猪的臀部视角
    pigs = collect_all_pigs_rear()
    print(f"\n共发现 {len(pigs)} 头猪有臀部视角数据:")
    sources = {}
    for p in pigs:
        sources[p['source']] = sources.get(p['source'], 0) + 1
    for src, cnt in sorted(sources.items()):
        print(f"  {src}: {cnt} 头")

    # 加载模型
    print("\n正在加载 YOLO + FastSAM 模型...")
    detect_model = YOLO('yolo11n.pt')
    segment_model = FastSAM('FastSAM-s.pt')
    print("模型加载完成\n")

    # 逐头处理
    total_success, total_skip, total_fail = 0, 0, 0

    for i, pig_info in enumerate(pigs):
        print(f"[{i+1}/{len(pigs)}] {pig_info['source']}/{pig_info['pig_id']}")

        s, sk, f = process_one_pig(pig_info, detect_model, segment_model)
        total_success += s
        total_skip += sk
        total_fail += f

        if s > 0 or f > 0:
            print(f"  成功: {s}, 跳过: {sk}, 失败: {f}")
        elif sk > 0:
            print(f"  全部已存在，跳过 {sk} 帧")

    print(f"\n{'=' * 60}")
    print(f"全部完成！")
    print(f"  总成功: {total_success}")
    print(f"  总跳过: {total_skip}")
    print(f"  总失败: {total_fail}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
