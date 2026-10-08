# -*- coding: utf-8 -*-
"""
==========================================================================
postprocess_pointcloud_v3_rear.py — 臀部视角点云后处理
==========================================================================
背景：
    generate_pointcloud_v3_rear.py 生成了臀部视角（sub）的掩膜点云。
    本脚本对其进行后处理，与背部 v2 后处理流程基本一致。

与背部后处理(postprocess_pointcloud_v2.py)的区别：
    背部视角：猪被栏杆分成3段（头~13%、背~64%、臀~21%），去掉最小簇（头部）
    臀部视角：相机从后方拍摄，看到的主要是臀部和部分背部，
              栏杆可能仍然切割点云，但不一定有"头部"。
              策略调整：仍然用 HDBSCAN 聚类，但只去掉明显的噪声碎片，
              不强制去掉最小簇（因为臀部视角最小簇可能是有价值的身体部分）。
              如果有 3+ 个大簇且最小簇占比 < 10%，才去掉它。

流程：HDBSCAN聚类 → 去噪声碎片 → 统计滤波 → 体素下采样 → 中心化

输入：masked_pointclouds_v3_rear/ 目录下的 .ply 文件
输出：processed_pointclouds_v3_rear/ 目录下的 .txt 文件

Author: 耿来硕 + Claude
Date: 2026-03-31
==========================================================================
"""

import os
import numpy as np
import open3d as o3d
import hdbscan
from tqdm import tqdm
import gc


# ============================================================
# 超参数
# ============================================================

# HDBSCAN 聚类参数
HDBSCAN_MIN_CLUSTER_SIZE = 500
HDBSCAN_MIN_SAMPLES = 10

# 簇大小阈值：低于此点数的簇视为噪声碎片
NOISE_CLUSTER_THRESHOLD = 5000

# 臀部视角去小簇阈值：最小簇占比低于此值才去掉
REAR_REMOVE_RATIO = 0.10

# 统计滤波参数
STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 2.0

# 体素下采样目标点数
TARGET_POINTS_MIN = 4000
TARGET_POINTS_MAX = 5000

# 断点续处理
RESUME = True


# ============================================================
# 收集所有猪的 masked_pointclouds_v3_rear 目录
# ============================================================
def collect_all_dirs():
    """
    遍历4个数据源，找到所有含 masked_pointclouds_v3_rear 的目录
    返回列表，每项 (source, pig_id, input_dir, output_dir)
    """
    dirs = []

    # --- nkyfirst ---
    nky1_root = r"G:\耿来硕数据\DK\nkyfirst"
    if os.path.isdir(nky1_root):
        for pig in sorted(os.listdir(nky1_root)):
            pig_path = os.path.join(nky1_root, pig, "depth_cz_r_d_p")
            for subdir in ["sub", "sub2"]:
                base = os.path.join(pig_path, subdir)
                inp = os.path.join(base, "masked_pointclouds_v3_rear")
                out = os.path.join(base, "processed_pointclouds_v3_rear")
                if os.path.isdir(inp):
                    dirs.append(('nkyfirst', pig, inp, out))
                    break

    # --- nkysecond ---
    nky2_root = r"G:\耿来硕数据\DK\nkysecond"
    if os.path.isdir(nky2_root):
        for pig in sorted(os.listdir(nky2_root)):
            pig_path = os.path.join(nky2_root, pig, "depth_cz_r_d_p")
            for subdir in ["sub10", "sub"]:
                base = os.path.join(pig_path, subdir)
                inp = os.path.join(base, "masked_pointclouds_v3_rear")
                out = os.path.join(base, "processed_pointclouds_v3_rear")
                if os.path.isdir(inp):
                    dirs.append(('nkysecond', pig, inp, out))
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
                inp = os.path.join(base, "masked_pointclouds_v3_rear")
                out = os.path.join(base, "processed_pointclouds_v3_rear")
                if os.path.isdir(inp):
                    dirs.append(('nkythird', f"{kinect_folder}/{pig}", inp, out))

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
            inp = os.path.join(base, "masked_pointclouds_v3_rear")
            out = os.path.join(base, "processed_pointclouds_v3_rear")
            if os.path.isdir(inp):
                dirs.append((f'丰美禾/{date_folder}', pig, inp, out))

    return dirs


# ============================================================
# HDBSCAN 聚类：臀部视角版本
# ============================================================
def hdbscan_clean_rear(points):
    """
    臀部视角 HDBSCAN 聚类：
    - 去掉噪声碎片（< NOISE_CLUSTER_THRESHOLD 的小簇）
    - 如果有 3+ 个大簇且最小簇占比 < 10%，去掉最小簇
    - 否则保留所有大簇
    """
    if len(points) < HDBSCAN_MIN_CLUSTER_SIZE * 2:
        return points, "点数太少，跳过聚类"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method='eom',
        algorithm='best',
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(points)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return points, "未找到有效簇，跳过"

    # 按点数从大到小排序
    sorted_idx = np.argsort(-counts)
    sorted_labels = unique_labels[sorted_idx]
    sorted_counts = counts[sorted_idx]

    # 过滤掉噪声碎片（< NOISE_CLUSTER_THRESHOLD）
    big_mask = sorted_counts >= NOISE_CLUSTER_THRESHOLD
    big_labels = sorted_labels[big_mask]
    big_counts = sorted_counts[big_mask]

    if len(big_labels) == 0:
        return points, "无大簇，跳过"

    total_big = big_counts.sum()

    if len(big_labels) >= 3:
        # 检查最小的大簇占比是否 < 10%
        smallest_ratio = big_counts[-1] / total_big
        if smallest_ratio < REAR_REMOVE_RATIO:
            keep_labels = big_labels[:-1]
            info = f"去小簇: 移除簇(大小{big_counts[-1]}, 占比{smallest_ratio:.1%}), 保留{len(keep_labels)}个簇"
        else:
            keep_labels = big_labels
            info = f"{len(big_labels)}个大簇(最小占比{smallest_ratio:.1%}>10%)，全部保留"
    else:
        keep_labels = big_labels
        info = f"仅{len(big_labels)}个大簇，全部保留"

    # 构建保留掩膜
    mask = np.isin(labels, keep_labels)
    kept = points[mask]

    return kept, info


# ============================================================
# 统计滤波
# ============================================================
def statistical_filter(pcd):
    """统计离群点去除"""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS,
        std_ratio=STAT_STD_RATIO
    )
    return cl


# ============================================================
# 体素下采样（迭代二分搜索）
# ============================================================
def voxel_downsample_to_target(pcd, target_min=TARGET_POINTS_MIN, target_max=TARGET_POINTS_MAX):
    """迭代调整体素大小，使最终点数落在目标范围内"""
    n = len(pcd.points)
    if n <= target_max:
        return pcd

    pts = np.asarray(pcd.points)
    bbox_vol = np.prod(pts.max(axis=0) - pts.min(axis=0))
    target_n = (target_min + target_max) // 2
    voxel_size = (bbox_vol / target_n) ** (1/3)

    lo, hi = voxel_size * 0.3, voxel_size * 3.0
    best_pcd = pcd
    for _ in range(20):
        mid = (lo + hi) / 2
        downsampled = pcd.voxel_down_sample(mid)
        n_down = len(downsampled.points)
        if target_min <= n_down <= target_max:
            best_pcd = downsampled
            break
        elif n_down > target_max:
            lo = mid
        else:
            hi = mid
        best_pcd = downsampled

    return best_pcd


# ============================================================
# 中心化（不做单位球归一化）
# ============================================================
def center_only(points):
    """只做中心化：减去质心，保留真实尺度"""
    centroid = np.mean(points, axis=0)
    return points - centroid


# ============================================================
# 处理单个点云文件
# ============================================================
def process_one(ply_path, out_path):
    """处理单个点云文件，返回 (True/False, info_str)"""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) < 100:
            return False, f"点数太少({len(pcd.points)})"

        points = np.asarray(pcd.points, dtype=np.float32)
        n_orig = len(points)

        # 1. HDBSCAN 去噪声（臀部视角版本）
        points, hdb_info = hdbscan_clean_rear(points)

        # 2. 统计滤波
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(points)
        pcd_clean = statistical_filter(pcd_clean)

        # 3. 体素下采样
        pcd_down = voxel_downsample_to_target(pcd_clean)

        # 4. 中心化（不做单位球归一化！）
        pts_final = np.asarray(pcd_down.points, dtype=np.float32)
        pts_final = center_only(pts_final)

        # 5. 保存
        np.savetxt(out_path, pts_final, fmt='%.6f')

        return True, f"{n_orig}→{len(pts_final)}点, {hdb_info}"

    except Exception as e:
        return False, f"异常: {e}"


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("臀部视角点云后处理 v3 (rear)")
    print("流程：HDBSCAN去噪 → 统计滤波 → 体素下采样 → 中心化")
    print("=" * 60)

    all_dirs = collect_all_dirs()
    print(f"\n共发现 {len(all_dirs)} 头猪有 masked_pointclouds_v3_rear 数据")

    if len(all_dirs) == 0:
        print("请先运行 generate_pointcloud_v3_rear.py 生成臀部视角掩膜点云")
        return

    total_success, total_skip, total_fail = 0, 0, 0

    for source, pig_id, input_dir, output_dir in tqdm(all_dirs, desc="总进度"):
        os.makedirs(output_dir, exist_ok=True)

        ply_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.ply')])
        if not ply_files:
            continue

        pig_success, pig_skip, pig_fail = 0, 0, 0

        for fname in ply_files:
            ply_path = os.path.join(input_dir, fname)
            out_name = fname.replace('.ply', '.txt')
            out_path = os.path.join(output_dir, out_name)

            if RESUME and os.path.exists(out_path):
                pig_skip += 1
                continue

            ok, info = process_one(ply_path, out_path)
            if ok:
                pig_success += 1
            else:
                pig_fail += 1
                print(f"  [{source}/{pig_id}] {fname}: {info}")

        total_success += pig_success
        total_skip += pig_skip
        total_fail += pig_fail

        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"全部完成！")
    print(f"  总成功: {total_success}")
    print(f"  总跳过: {total_skip}")
    print(f"  总失败: {total_fail}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
