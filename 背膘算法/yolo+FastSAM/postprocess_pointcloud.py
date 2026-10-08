# -*- coding: utf-8 -*-
"""
点云后处理流水线（全自动，无需手动干预）
输入：generate_pointcloud.py 生成的原始掩膜点云 (.ply)
输出：处理后的干净点云 (.txt，格式与师兄数据集完全一致)

流程：
    原始掩膜点云 → HDBSCAN聚类(取最大簇) → 统计滤波去噪 → 体素下采样 → 归一化 → .txt
"""

import os
import sys
import numpy as np
import open3d as o3d
import hdbscan
from tqdm import tqdm
import gc


# ============================================================
# 超参数（一次设定，全自动运行）
# ============================================================
INPUT_DIR   = r"G:\耿来硕数据\DK\nkyfirst\2-1\depth_cz_r_d_p\master\masked_pointclouds"
OUTPUT_DIR  = r"G:\耿来硕数据\DK\nkyfirst\2-1\depth_cz_r_d_p\master\processed_pointclouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HDBSCAN：猪体主体最少点数
HDBSCAN_MIN_CLUSTER_SIZE = 500

# 簇大小阈值：低于此点数的簇视为噪声碎片，予以丢弃
# 猪体被栏杆分成3段，最小段(臀部)约占13% ≈ 26万*0.13 ≈ 34000点
# 噪声碎片通常<2000点，设5000作为安全阈值
NOISE_CLUSTER_THRESHOLD = 5000

# 统计滤波：去除离群点
STAT_NB_NEIGHBORS = 30   # 考察每个点的30个邻居
STAT_STD_RATIO    = 2.0  # 距离均值超过2个标准差则去除（比师兄的1.0更宽松，适合掩膜点云）

# 体素下采样目标点数（迭代调整体素大小，直到点数在目标范围内）
TARGET_POINTS_MIN = 4000
TARGET_POINTS_MAX = 5000

# 是否跳过已处理的文件（断点续处理）
RESUME = True


# ============================================================
# 核心处理函数
# ============================================================

def hdbscan_remove_noise(points: np.ndarray) -> np.ndarray:
    """
    HDBSCAN聚类，保留所有猪体部分（被栏杆分成的多段），只丢弃小碎片噪声。

    猪体被栏杆分成3段（头部、背部、臀部），每段都是有效点云。
    只有点数低于 NOISE_CLUSTER_THRESHOLD 的簇才被视为噪声丢弃。
    """
    if len(points) < HDBSCAN_MIN_CLUSTER_SIZE * 2:
        return points

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=10,
        cluster_selection_method='eom',
        algorithm='best',
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(points)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        print("    [HDBSCAN] 警告：未找到有效簇，跳过聚类步骤")
        return points

    # 保留所有大簇（猪体各部分），只丢弃小碎片和噪声点
    keep_labels = unique_labels[counts >= NOISE_CLUSTER_THRESHOLD]
    discard_labels = unique_labels[counts < NOISE_CLUSTER_THRESHOLD]

    # 构建保留掩膜：大簇的点保留，噪声点(-1)和小簇的点丢弃
    mask = np.isin(labels, keep_labels)
    kept = points[mask]

    n_kept_clusters = len(keep_labels)
    n_discard_clusters = len(discard_labels)
    noise_count = np.sum(labels == -1)
    discard_pts = np.sum(np.isin(labels, discard_labels)) + noise_count

    print(f"    [HDBSCAN] 保留 {n_kept_clusters} 个猪体簇, "
          f"丢弃 {n_discard_clusters} 个碎片簇 + {noise_count} 噪声点 = 去除 {discard_pts} 点")
    for lb, cnt in zip(unique_labels, counts):
        tag = "✅保留" if cnt >= NOISE_CLUSTER_THRESHOLD else "❌丢弃"
        print(f"      簇{lb}: {cnt} 点 ({cnt/len(points)*100:.1f}%) {tag}")

    return kept


def statistical_filter(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """统计离群点去除"""
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS,
        std_ratio=STAT_STD_RATIO
    )
    removed = len(pcd.points) - len(cl.points)
    if removed > 0:
        print(f"    [统计滤波] 去除离群点: {removed} 个")
    return cl


def voxel_downsample_to_target(pcd: o3d.geometry.PointCloud,
                                 target_min: int = TARGET_POINTS_MIN,
                                 target_max: int = TARGET_POINTS_MAX) -> o3d.geometry.PointCloud:
    """
    迭代调整体素大小，使最终点数落在 [target_min, target_max] 范围内。
    全自动，无需手动指定体素大小。
    """
    n = len(pcd.points)
    if n <= target_max:
        if n < target_min:
            print(f"    [体素下采样] 点数({n})已低于目标，跳过下采样")
        return pcd

    # 估计初始体素大小：根据点云包围盒体积和目标点数
    pts = np.asarray(pcd.points)
    bbox_vol = np.prod(pts.max(axis=0) - pts.min(axis=0))
    # 体素数量 ≈ 体积 / 体素大小^3
    target_n = (target_min + target_max) // 2
    voxel_size = (bbox_vol / target_n) ** (1/3)

    # 二分搜索精确调整
    lo, hi = voxel_size * 0.3, voxel_size * 3.0
    best_pcd = pcd
    for _ in range(20):  # 最多迭代20次
        mid = (lo + hi) / 2
        downsampled = pcd.voxel_down_sample(mid)
        n_down = len(downsampled.points)
        if target_min <= n_down <= target_max:
            best_pcd = downsampled
            break
        elif n_down > target_max:
            lo = mid  # 体素太小，点太多 → 增大体素
        else:
            hi = mid  # 体素太大，点太少 → 减小体素
        best_pcd = downsampled

    print(f"    [体素下采样] {n} → {len(best_pcd.points)} 点 (体素={mid:.4f}m)")
    return best_pcd


def normalize(points: np.ndarray) -> np.ndarray:
    """中心化 + 缩放到单位球（与师兄数据集一致）"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dist = np.max(np.linalg.norm(points, axis=1))
    if dist > 0:
        points = points / dist
    return points


def process_one(ply_path: str, out_path: str) -> bool:
    """
    处理单个点云文件。
    返回 True 表示成功，False 表示失败/跳过。
    """
    try:
        # 1. 读取原始点云
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) < 100:
            print(f"    [跳过] 点数太少: {len(pcd.points)}")
            return False

        print(f"    原始点数: {len(pcd.points)}")
        points = np.asarray(pcd.points, dtype=np.float32)

        # 2. HDBSCAN聚类：保留猪体各部分，只去掉小碎片和噪声
        points = hdbscan_remove_noise(points)

        # 3. 统计滤波：去除边缘离群点
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(points)
        pcd_clean = statistical_filter(pcd_clean)

        # 4. 体素下采样：降至4000-5000点
        pcd_down = voxel_downsample_to_target(pcd_clean)

        # 5. 归一化
        pts_final = np.asarray(pcd_down.points, dtype=np.float32)
        pts_final = normalize(pts_final)

        # 6. 保存为 .txt（与师兄数据集格式一致）
        np.savetxt(out_path, pts_final, fmt='%.6f')
        print(f"    ✅ 保存: {os.path.basename(out_path)} ({len(pts_final)} 点)")
        return True

    except Exception as e:
        print(f"    ❌ 处理失败: {e}")
        return False


# ============================================================
# 主流程
# ============================================================

def main():
    ply_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.ply')])
    print(f"共发现 {len(ply_files)} 个PLY文件")

    success, skip, fail = 0, 0, 0

    for fname in tqdm(ply_files, desc="后处理进度"):
        ply_path = os.path.join(INPUT_DIR, fname)
        out_name = fname.replace('.ply', '.txt')
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if RESUME and os.path.exists(out_path):
            skip += 1
            continue

        print(f"\n处理: {fname}")
        ok = process_one(ply_path, out_path)
        if ok:
            success += 1
        else:
            fail += 1

        gc.collect()

    print(f"\n{'='*50}")
    print(f"完成！成功: {success}, 跳过: {skip}, 失败: {fail}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
