# -*- coding: utf-8 -*-
"""
批量处理所有猪只点云 - 阶段一全自动脚本

完整流程 (每帧):
    RGB(.jpg) → YOLO+FastSAM → 掩膜
    掩膜 + Depth(.png) + 相机内参 → 3D点云
    → HDBSCAN去噪 → 统计滤波 → 体素下采样(4000-5000点) → 归一化 → .txt

支持:
    - 5个批次不同目录结构自动适配
    - 自动读取每头猪的 intrinsic.json 相机内参
    - 断点续传 (已有 .txt 的帧自动跳过)
    - GPU显存自动清理
"""

import os
import sys
import json
import gc
import time
import numpy as np
import cv2
import open3d as o3d
import hdbscan
import torch
from ultralytics import YOLO, FastSAM
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# 数据源配置
# ============================================================
# nky 批次默认内参 (省农科院 Azure Kinect, SN: 000341693512)
NKY_INTRINSICS = {
    'fx': 614.84063720703125, 'fy': 614.61328125,
    'cx': 642.5406494140625,  'cy': 364.29312133789062
}

# 丰美禾默认内参 (另一台 Azure Kinect, SN: 000493521612)
FMH_INTRINSICS = {
    'fx': 608.392333984375,   'fy': 608.1031494140625,
    'cx': 641.02337646484375, 'cy': 363.46273803710938
}

BATCH_CONFIGS = [
    {
        'name': 'nkyfirst',
        'root': r'G:\耿来硕数据\DK\nkyfirst',
        'structure': 'nky_flat',
        'default_intrinsics': NKY_INTRINSICS,
    },
    {
        'name': 'nkysecond',
        'root': r'G:\耿来硕数据\DK\nkysecond',
        'structure': 'nky_flat',
        'default_intrinsics': NKY_INTRINSICS,
    },
    {
        'name': 'nkythird',
        'root': r'G:\耿来硕数据\DK\nkythird',
        'structure': 'nky_nested',
        'default_intrinsics': NKY_INTRINSICS,
    },
    {
        'name': 'nky3.18',
        'root': r'G:\耿来硕数据\DK\nky3.18',
        'structure': 'nky_flat',
        'default_intrinsics': NKY_INTRINSICS,
    },
    {
        'name': '丰美禾',
        'root': r'G:\耿来硕数据\丰美禾数据',
        'structure': 'fmh',
        'default_intrinsics': FMH_INTRINSICS,
    },
]


# ============================================================
# 后处理超参数
# ============================================================
HDBSCAN_MIN_CLUSTER_SIZE = 500
NOISE_CLUSTER_THRESHOLD = 5000
STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 2.0
TARGET_POINTS_MIN = 4000
TARGET_POINTS_MAX = 5000

# 深度图参数
DEPTH_SCALE = 1000.0
MIN_DEPTH = 0.3
MAX_DEPTH = 3.0


# ============================================================
# 工具函数
# ============================================================

def imread_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imread_depth_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def read_intrinsics(json_path):
    """从 intrinsic.json 读取相机内参"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        m = data['intrinsic_matrix']
        # column-major 3x3: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        return {'fx': m[0], 'fy': m[4], 'cx': m[6], 'cy': m[7]}
    except Exception:
        return None


# ============================================================
# 扫描所有猪只目录
# ============================================================

def find_master_dir_nky(pig_path):
    """在 nky 批次猪目录下查找 master 目录"""
    dczrdp = os.path.join(pig_path, 'depth_cz_r_d_p')
    if not os.path.isdir(dczrdp):
        return None
    for mname in ['master', 'master10']:
        candidate = os.path.join(dczrdp, mname)
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, 'color')):
            return candidate
    return None


def scan_all_pigs():
    """
    扫描所有批次，返回列表:
    [(batch_name, pig_id, master_dir, intrinsics), ...]
    """
    all_pigs = []

    for cfg in BATCH_CONFIGS:
        root = cfg['root']
        if not os.path.isdir(root):
            print(f"  [跳过] 目录不存在: {root}")
            continue

        structure = cfg['structure']
        default_intr = cfg['default_intrinsics']
        batch_name = cfg['name']
        count_before = len(all_pigs)

        if structure == 'nky_flat':
            # {root}/{pig}/depth_cz_r_d_p/{master}/
            for pig in sorted(os.listdir(root)):
                pig_path = os.path.join(root, pig)
                if not os.path.isdir(pig_path):
                    continue
                master_dir = find_master_dir_nky(pig_path)
                if master_dir is None:
                    continue
                intr_path = os.path.join(master_dir, 'intrinsic.json')
                intr = read_intrinsics(intr_path) or default_intr
                all_pigs.append((batch_name, pig, master_dir, intr))

        elif structure == 'nky_nested':
            # {root}/{date_dir}/{pig}/depth_cz_r_d_p/{master}/
            for date_dir in sorted(os.listdir(root)):
                date_path = os.path.join(root, date_dir)
                if not os.path.isdir(date_path):
                    continue
                for pig in sorted(os.listdir(date_path)):
                    pig_path = os.path.join(date_path, pig)
                    if not os.path.isdir(pig_path):
                        continue
                    master_dir = find_master_dir_nky(pig_path)
                    if master_dir is None:
                        continue
                    intr_path = os.path.join(master_dir, 'intrinsic.json')
                    intr = read_intrinsics(intr_path) or default_intr
                    all_pigs.append((batch_name, f"{date_dir}/{pig}", master_dir, intr))

        elif structure == 'fmh':
            # {root}/{date}/{pig}/master/
            for date_dir in sorted(os.listdir(root)):
                date_path = os.path.join(root, date_dir)
                if not os.path.isdir(date_path):
                    continue
                for pig in sorted(os.listdir(date_path)):
                    pig_path = os.path.join(date_path, pig)
                    if not os.path.isdir(pig_path):
                        continue
                    master_dir = os.path.join(pig_path, 'master')
                    if not os.path.isdir(master_dir):
                        continue
                    if not os.path.isdir(os.path.join(master_dir, 'color')):
                        continue
                    intr_path = os.path.join(master_dir, 'intrinsic.json')
                    intr = read_intrinsics(intr_path) or default_intr
                    all_pigs.append((batch_name, f"{date_dir}/{pig}", master_dir, intr))

        count_batch = len(all_pigs) - count_before
        print(f"  [{batch_name}] {count_batch} 头猪")

    return all_pigs


# ============================================================
# FastSAM 掩膜生成
# ============================================================

def get_mask(img, detect_model, segment_model):
    """用 YOLO + FastSAM 获取猪体掩膜"""
    h, w = img.shape[:2]

    det_results = detect_model.predict(img, conf=0.15, device='cuda', verbose=False)[0]

    if len(det_results.boxes) == 0:
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
        # 确保 mask 和深度图分辨率一致
        depth_h, depth_w = 720, 1280
        if mask.shape[:2] != (depth_h, depth_w):
            mask = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
        return mask
    return None


# ============================================================
# 点云生成
# ============================================================

def depth_to_pointcloud(depth_img, mask, intr):
    """将掩膜区域的 depth 反投影为 3D 点云"""
    v, u = np.where(mask == 1)
    z = depth_img[v, u].astype(np.float32) / DEPTH_SCALE

    valid = (z > MIN_DEPTH) & (z < MAX_DEPTH)
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - intr['cx']) * z / intr['fx']
    y = (v - intr['cy']) * z / intr['fy']

    points = np.stack([x, y, z], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# ============================================================
# 后处理函数
# ============================================================

# HDBSCAN 预下采样：先粗下采样到此点数再聚类，大幅加速
HDBSCAN_PRE_DOWNSAMPLE = 50000


def hdbscan_remove_noise(points):
    """
    HDBSCAN聚类: 保留所有猪体大簇, 只丢弃小碎片噪声。
    先粗下采样到 ~50000 点再聚类 (300k→50k, HDBSCAN 速度提升 ~6倍)
    """
    if len(points) < HDBSCAN_MIN_CLUSTER_SIZE * 2:
        return points

    # 随机子采样以加速 HDBSCAN (聚类只需识别簇，不需要全分辨率)
    if len(points) > HDBSCAN_PRE_DOWNSAMPLE:
        indices = np.random.choice(len(points), HDBSCAN_PRE_DOWNSAMPLE, replace=False)
        indices.sort()
        points_for_cluster = points[indices]
    else:
        points_for_cluster = points

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(HDBSCAN_MIN_CLUSTER_SIZE, len(points_for_cluster) // 5),
        min_samples=10,
        cluster_selection_method='eom',
        algorithm='best',
        core_dist_n_jobs=1,  # 单线程，避免Windows多进程+CUDA死锁
    )
    labels = clusterer.fit_predict(points_for_cluster)

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return points

    # 噪声阈值按下采样比例缩放
    ratio = len(points_for_cluster) / len(points) if len(points) > HDBSCAN_PRE_DOWNSAMPLE else 1.0
    scaled_threshold = max(int(NOISE_CLUSTER_THRESHOLD * ratio), 500)
    keep_labels = unique_labels[counts >= scaled_threshold]

    if len(points) > HDBSCAN_PRE_DOWNSAMPLE:
        # 将下采样点的聚类结果映射回原始点云 (最近邻)
        from scipy.spatial import cKDTree
        tree = cKDTree(points_for_cluster)
        _, indices = tree.query(points, k=1)
        original_labels = labels[indices]
        mask = np.isin(original_labels, keep_labels)
    else:
        mask = np.isin(labels, keep_labels)

    kept = points[mask]
    return kept if len(kept) > 100 else points


def statistical_filter(pcd):
    """统计离群点去除"""
    cl, _ = pcd.remove_statistical_outlier(
        nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO
    )
    return cl


def voxel_downsample_to_target(pcd):
    """迭代体素下采样到目标点数范围"""
    n = len(pcd.points)
    if n <= TARGET_POINTS_MAX:
        return pcd

    pts = np.asarray(pcd.points)
    bbox_range = pts.max(axis=0) - pts.min(axis=0)
    bbox_vol = np.prod(np.maximum(bbox_range, 1e-6))
    target_n = (TARGET_POINTS_MIN + TARGET_POINTS_MAX) // 2
    voxel_size = (bbox_vol / target_n) ** (1 / 3)

    lo, hi = voxel_size * 0.3, voxel_size * 3.0
    best_pcd = pcd
    mid = voxel_size
    for _ in range(20):
        mid = (lo + hi) / 2
        downsampled = pcd.voxel_down_sample(mid)
        n_down = len(downsampled.points)
        if TARGET_POINTS_MIN <= n_down <= TARGET_POINTS_MAX:
            return downsampled
        elif n_down > TARGET_POINTS_MAX:
            lo = mid
        else:
            hi = mid
        best_pcd = downsampled

    return best_pcd


def depth_outlier_filter(points, k=3.0):
    """
    深度异常值过滤：去除栏杆等遮挡物的点。

    原理：相机从上方拍摄，猪体表面深度(Z)集中在一个范围内，
    而栏杆在猪体前方(更靠近相机)，Z值明显偏小。
    用中位数 + MAD 方法去除深度异常偏小的点。
    """
    z = points[:, 2]  # 深度轴 (相机坐标系中 Z = 深度)
    median_z = np.median(z)
    mad = np.median(np.abs(z - median_z))

    if mad < 1e-6:
        return points

    # 去除深度明显偏小(栏杆/遮挡物更靠近相机)和偏大的点
    lower = median_z - k * 1.4826 * mad  # 1.4826使MAD等效于标准差
    upper = median_z + k * 1.4826 * mad
    mask = (z >= lower) & (z <= upper)
    kept = points[mask]

    return kept if len(kept) > 100 else points


def normalize(points):
    """中心化 + 缩放到单位球"""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dist = np.max(np.linalg.norm(points, axis=1))
    if dist > 0:
        points = points / dist
    return points


# ============================================================
# 处理单帧 (完整流水线, 主线程直接执行)
# ============================================================

def process_single_frame(color_path, depth_path, txt_out_path,
                         detect_model, segment_model, intr):
    """
    单帧完整处理: RGB→掩膜→点云→后处理→.txt
    返回 (True, 点数) 或 (False, 错误信息)
    """
    try:
        # 1. 读取图像
        color_img = imread_chinese(color_path)
        depth_img = imread_depth_chinese(depth_path)
        if color_img is None or depth_img is None:
            return False, "图片读取失败"

        # 2. YOLO + FastSAM 分割
        mask = get_mask(color_img, detect_model, segment_model)
        if mask is None:
            return False, "分割失败"

        # 3. 掩膜 + 深度图 → 3D点云
        pcd = depth_to_pointcloud(depth_img, mask, intr)
        n_raw = len(pcd.points)
        if n_raw < 100:
            return False, f"有效点太少({n_raw})"

        # 4. HDBSCAN 去噪 (内部先粗下采样再聚类)
        points = np.asarray(pcd.points, dtype=np.float32)
        points = hdbscan_remove_noise(points)

        # 5. 深度过滤：去除栏杆等遮挡物 (Z值异常偏小的点)
        points = depth_outlier_filter(points)

        # 6. 统计滤波
        pcd_clean = o3d.geometry.PointCloud()
        pcd_clean.points = o3d.utility.Vector3dVector(points)
        pcd_clean = statistical_filter(pcd_clean)

        # 7. 体素下采样
        pcd_down = voxel_downsample_to_target(pcd_clean)

        # 8. 归一化
        pts_final = np.asarray(pcd_down.points, dtype=np.float32)
        if len(pts_final) < 50:
            return False, f"后处理后点数过少({len(pts_final)})"
        pts_final = normalize(pts_final)

        # 9. 保存 .txt
        np.savetxt(txt_out_path, pts_final, fmt='%.6f')
        return True, len(pts_final)

    except Exception as e:
        return False, str(e)

    finally:
        gc.collect()


# ============================================================
# 处理单头猪
# ============================================================

def process_one_pig(batch_name, pig_id, master_dir, intr,
                    detect_model, segment_model, pig_idx, total_pigs):
    """
    处理一头猪的所有帧。
    返回 (success_count, skip_count, fail_count)
    """
    color_dir = os.path.join(master_dir, 'color')
    depth_dir = os.path.join(master_dir, 'depth')

    if not os.path.isdir(color_dir) or not os.path.isdir(depth_dir):
        return 0, 0, 0

    out_dir = os.path.join(master_dir, 'processed_pointclouds')
    os.makedirs(out_dir, exist_ok=True)

    # 获取所有帧
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    if not color_files:
        return 0, 0, 0

    # 筛选待处理帧 (断点续传)
    frames_todo = []
    skip = 0
    for fname in color_files:
        stem = os.path.splitext(fname)[0]
        txt_path = os.path.join(out_dir, f"pig_{stem}.txt")
        if os.path.exists(txt_path):
            skip += 1
        else:
            depth_path = os.path.join(depth_dir, f"{stem}.png")
            if os.path.exists(depth_path):
                frames_todo.append((fname, stem, depth_path, txt_path))

    if not frames_todo:
        return 0, skip, 0

    print(f"\n[{pig_idx + 1}/{total_pigs}] {batch_name}/{pig_id}: "
          f"{len(frames_todo)} 帧待处理, {skip} 帧已跳过")

    success = 0
    fail = 0

    for fname, stem, depth_path, txt_path in tqdm(
        frames_todo, desc=f"  {pig_id[-20:]}", leave=False
    ):
        color_path = os.path.join(color_dir, fname)
        ok, result = process_single_frame(
            color_path, depth_path, txt_path,
            detect_model, segment_model, intr
        )
        if ok:
            success += 1
        else:
            fail += 1

        # 每20帧清理一次GPU显存
        if (success + fail) % 20 == 0:
            torch.cuda.empty_cache()

    print(f"  → 成功 {success}, 失败 {fail}")
    torch.cuda.empty_cache()

    return success, skip, fail


# ============================================================
# 主流程
# ============================================================

def main():
    start_time = time.time()

    print("=" * 60)
    print(" 批量点云处理脚本 - 阶段一")
    print(" RGB → FastSAM掩膜 → 点云 → 后处理 → .txt")
    print("=" * 60)

    # 1. 扫描所有猪只
    print("\n正在扫描所有数据目录...")
    all_pigs = scan_all_pigs()
    print(f"\n共扫描到 {len(all_pigs)} 头猪")

    if not all_pigs:
        print("未找到任何猪只数据，请检查路径配置")
        return

    # 2. 加载模型 (全局只加载一次)
    print("\n正在加载 YOLO + FastSAM 模型...")
    detect_model = YOLO('yolo11n.pt')
    segment_model = FastSAM('FastSAM-s.pt')
    print("模型加载完成!\n")

    # 3. 逐头猪处理
    total_success = 0
    total_skip = 0
    total_fail = 0
    total_pigs = len(all_pigs)

    for pig_idx, (batch_name, pig_id, master_dir, intr) in enumerate(all_pigs):
        s, sk, f = process_one_pig(
            batch_name, pig_id, master_dir, intr,
            detect_model, segment_model,
            pig_idx, total_pigs
        )
        total_success += s
        total_skip += sk
        total_fail += f

    # 4. 最终统计
    elapsed = time.time() - start_time
    hours = elapsed / 3600

    print(f"\n{'=' * 60}")
    print(f" 全部处理完成!")
    print(f" 成功: {total_success}")
    print(f" 跳过: {total_skip}")
    print(f" 失败: {total_fail}")
    print(f" 耗时: {hours:.1f} 小时")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
