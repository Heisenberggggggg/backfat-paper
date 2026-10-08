"""
点云统计特征提取脚本
==================
背景：PointMLP-Elite 深度学习模型在189头猪的小样本上严重过拟合（训练R²=0.69，验证R²≈0，
测试R²=-0.22），模型无法从原始xyz坐标中学到与背膘厚度相关的泛化特征。

解决思路：不让模型自己从原始坐标学特征，而是人工计算点云的统计特征（高度分布、宽度、曲率、
体积等约30维），将"点云→背膘"的问题转化为"30个数字→背膘"的结构化回归问题，
然后用XGBoost等传统ML方法训练（小样本下远优于深度学习）。

本脚本属于"方案三：传统ML+手工特征"的第一步（特征提取），在后处理之后、模型训练之前执行。
复用 train.py 中的 build_dataset_registry() 获取数据路径和标签，确保数据一致。

输入：各猪目录下的 .txt 点云文件（每行 x y z，约4000-5000点）
输出：traditional_ml/features_per_frame.csv（每帧一行）
      traditional_ml/features_per_pig.csv（每猪一行，各帧均值）

Author: 耿来硕 + Claude
Date: 2026-03-23
"""

import numpy as np
import os
import sys
import json
import time
import csv
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA

# 将当前目录加入 path，以便 import train.py 中的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_dataset_registry


# ==============================================================================
# 特征提取函数
# ==============================================================================

def extract_features_from_pointcloud(pc):
    """
    从单个点云（N×3 numpy数组）提取统计特征向量。

    返回：(feature_dict, feature_names)
        feature_dict: {特征名: 值}
    """
    features = {}

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    n_points = len(pc)

    # ------------------------------------------------------------------
    # 1. 各轴基础统计量
    # ------------------------------------------------------------------
    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_range'] = np.ptp(axis_data)  # max - min
        features[f'{axis_name}_skew'] = float(skew(axis_data))
        features[f'{axis_name}_kurtosis'] = float(kurtosis(axis_data))

    # ------------------------------------------------------------------
    # 2. 四分位数（重点关注Z轴/高度，与背膘最相关）
    # ------------------------------------------------------------------
    for axis_name, axis_data in [('z', z)]:
        q25, q50, q75 = np.percentile(axis_data, [25, 50, 75])
        features[f'{axis_name}_q25'] = q25
        features[f'{axis_name}_q50'] = q50
        features[f'{axis_name}_q75'] = q75
        features[f'{axis_name}_iqr'] = q75 - q25

    # ------------------------------------------------------------------
    # 3. 表面平坦度 / 曲度：点到最佳拟合平面的距离
    # ------------------------------------------------------------------
    try:
        centroid = np.mean(pc, axis=0)
        pc_centered = pc - centroid
        # SVD 拟合最佳平面
        _, s, vh = np.linalg.svd(pc_centered, full_matrices=False)
        normal = vh[2]  # 最小奇异值对应的方向 = 平面法向量
        # 各点到平面的有符号距离
        dists_to_plane = pc_centered @ normal
        features['plane_residual_mean'] = np.mean(np.abs(dists_to_plane))
        features['plane_residual_std'] = np.std(dists_to_plane)
        features['plane_residual_max'] = np.max(np.abs(dists_to_plane))
        # 奇异值比例反映形状扁平程度
        s_sum = np.sum(s) + 1e-8
        features['svd_ratio_1'] = s[0] / s_sum  # 最大方向占比
        features['svd_ratio_2'] = s[1] / s_sum
        features['svd_ratio_3'] = s[2] / s_sum  # 越小=越平坦
    except Exception:
        features['plane_residual_mean'] = 0.0
        features['plane_residual_std'] = 0.0
        features['plane_residual_max'] = 0.0
        features['svd_ratio_1'] = 0.33
        features['svd_ratio_2'] = 0.33
        features['svd_ratio_3'] = 0.33

    # ------------------------------------------------------------------
    # 4. PCA 主成分方差比例（体型/姿态信息）
    # ------------------------------------------------------------------
    try:
        pca = PCA(n_components=3)
        pca.fit(pc)
        features['pca_var_ratio_1'] = pca.explained_variance_ratio_[0]
        features['pca_var_ratio_2'] = pca.explained_variance_ratio_[1]
        features['pca_var_ratio_3'] = pca.explained_variance_ratio_[2]
    except Exception:
        features['pca_var_ratio_1'] = 0.33
        features['pca_var_ratio_2'] = 0.33
        features['pca_var_ratio_3'] = 0.33

    # ------------------------------------------------------------------
    # 5. 凸包特征（体积、表面积、密度）
    # ------------------------------------------------------------------
    try:
        hull = ConvexHull(pc)
        features['convex_hull_volume'] = hull.volume
        features['convex_hull_area'] = hull.area
        features['point_density'] = n_points / (hull.volume + 1e-8)
    except Exception:
        # 点云退化（共面或点太少）
        features['convex_hull_volume'] = 0.0
        features['convex_hull_area'] = 0.0
        features['point_density'] = 0.0

    # ------------------------------------------------------------------
    # 6. 协方差特征（轴间相关性）
    # ------------------------------------------------------------------
    cov_matrix = np.cov(pc.T)
    features['cov_xy'] = cov_matrix[0, 1]
    features['cov_xz'] = cov_matrix[0, 2]
    features['cov_yz'] = cov_matrix[1, 2]

    # ------------------------------------------------------------------
    # 7. 点数（可能反映点云质量/猪体大小）
    # ------------------------------------------------------------------
    features['n_points'] = n_points

    return features


def get_feature_names():
    """返回固定顺序的特征名列表（和 extract_features_from_pointcloud 的 key 顺序一致）。"""
    names = []
    for axis in ['x', 'y', 'z']:
        names += [f'{axis}_mean', f'{axis}_std', f'{axis}_min', f'{axis}_max',
                  f'{axis}_range', f'{axis}_skew', f'{axis}_kurtosis']
    names += ['z_q25', 'z_q50', 'z_q75', 'z_iqr']
    names += ['plane_residual_mean', 'plane_residual_std', 'plane_residual_max',
              'svd_ratio_1', 'svd_ratio_2', 'svd_ratio_3']
    names += ['pca_var_ratio_1', 'pca_var_ratio_2', 'pca_var_ratio_3']
    names += ['convex_hull_volume', 'convex_hull_area', 'point_density']
    names += ['cov_xy', 'cov_xz', 'cov_yz']
    names += ['n_points']
    return names


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'traditional_ml')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'extract_features_log.txt')

    def log(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("开始提取点云统计特征")
    log("=" * 60)

    # 1. 构建数据注册表（复用 train.py 的函数）
    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    log(f"数据源: DK={dk_base}, FMH={fmh_base}")
    registry = build_dataset_registry(dk_base, fmh_base)
    log(f"匹配到 {len(registry)} 头猪")

    # 2. 读取已有的数据划分（确保和深度学习版一致）
    split_path = os.path.join(script_dir, 'checkpoints', 'split_info.json')
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split_info = json.load(f)
        log(f"使用已有的数据划分: train={len(split_info['train_pigs'])}头, "
            f"val={len(split_info['val_pigs'])}头, test={len(split_info['test_pigs'])}头")
    else:
        log("警告: 未找到 split_info.json，将无法标记 train/val/test 归属")
        split_info = None

    # 构建 pig_id -> split 映射
    pig_split_map = {}
    if split_info:
        for pid in split_info.get('train_pigs', []):
            pig_split_map[pid] = 'train'
        for pid in split_info.get('val_pigs', []):
            pig_split_map[pid] = 'val'
        for pid in split_info.get('test_pigs', []):
            pig_split_map[pid] = 'test'

    # 3. 提取特征
    feature_names = get_feature_names()
    frame_header = ['pig_id', 'split', 'backfat', 'file_path'] + feature_names

    frame_csv_path = os.path.join(output_dir, 'features_per_frame.csv')
    pig_csv_path = os.path.join(output_dir, 'features_per_pig.csv')

    # 按猪聚合
    pig_features = {}  # {pig_id: {'backfat': float, 'split': str, 'features': [list of feature_dicts]}}

    total_files = sum(info[2] for info in registry.values())
    log(f"共需处理 {total_files} 个点云文件")

    processed = 0
    errors = 0
    start_time = time.time()

    with open(frame_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(frame_header)

        for pig_idx, (pig_id, (data_dir, backfat, n_frames)) in enumerate(sorted(registry.items())):
            split_label = pig_split_map.get(pig_id, 'unknown')
            pig_features[pig_id] = {
                'backfat': backfat,
                'split': split_label,
                'features': []
            }

            # 遍历该猪的所有帧
            txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
            for fname in txt_files:
                fpath = os.path.join(data_dir, fname)
                try:
                    pc = np.loadtxt(fpath, dtype=np.float32)
                    if pc.ndim != 2 or pc.shape[1] < 3:
                        errors += 1
                        continue
                    pc = pc[:, :3]

                    feat = extract_features_from_pointcloud(pc)
                    feat_values = [feat[name] for name in feature_names]

                    writer.writerow([pig_id, split_label, backfat, fpath] + feat_values)
                    pig_features[pig_id]['features'].append(feat)

                except Exception as e:
                    errors += 1
                    if errors <= 10:
                        log(f"  错误: {fpath}: {e}")

                processed += 1
                if processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    speed = processed / elapsed
                    eta = (total_files - processed) / speed
                    log(f"  进度: {processed}/{total_files} ({100*processed/total_files:.1f}%), "
                        f"速度: {speed:.0f} 文件/秒, 预计剩余: {eta:.0f}秒")

            if (pig_idx + 1) % 20 == 0:
                log(f"  已完成 {pig_idx+1}/{len(registry)} 头猪")

    elapsed_total = time.time() - start_time
    log(f"帧级特征提取完成: {processed} 帧, {errors} 个错误, 耗时 {elapsed_total:.1f}秒")
    log(f"保存到: {frame_csv_path}")

    # 4. 计算猪级别的平均特征
    pig_header = ['pig_id', 'split', 'backfat', 'n_frames'] + feature_names

    with open(pig_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(pig_header)

        for pig_id in sorted(pig_features.keys()):
            info = pig_features[pig_id]
            feat_list = info['features']
            if not feat_list:
                log(f"  警告: {pig_id} 无有效帧，跳过")
                continue

            # 各特征取均值
            avg_feat = {}
            for name in feature_names:
                values = [f[name] for f in feat_list]
                avg_feat[name] = np.mean(values)

            feat_values = [avg_feat[name] for name in feature_names]
            writer.writerow([pig_id, info['split'], info['backfat'], len(feat_list)] + feat_values)

    log(f"猪级特征保存到: {pig_csv_path}")
    log(f"共 {len([p for p in pig_features if pig_features[p]['features']])} 头猪有有效特征")

    # 5. 打印特征统计概览
    log("\n特征维度: {}".format(len(feature_names)))
    log("特征列表:")
    for i, name in enumerate(feature_names):
        log(f"  [{i+1:2d}] {name}")

    log("\n" + "=" * 60)
    log("特征提取完成！")
    log(f"下一步: 运行 train_traditional.py 进行模型训练")
    log("=" * 60)


if __name__ == '__main__':
    main()
