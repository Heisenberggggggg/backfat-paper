# -*- coding: utf-8 -*-
"""
快速可行性验证：点云统计特征 vs 背膘值 相关性分析
用已处理完的5头猪做初步检验，判断技术路线是否可行
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 数据配置
# ============================================================
# B超背膘值 (mm) - 来自第一次数据7.29.docx
BACKFAT_LABELS = {
    '2-1':  16.1,
    '2-2':  16.0,
    '2-18': 19.6,
    '2-22': 17.9,
    '2-25': 13.8,
}

BASE_DIR = r'G:\耿来硕数据\DK\nkyfirst'


def get_processed_dir(pig_id):
    return os.path.join(BASE_DIR, pig_id, 'depth_cz_r_d_p', 'master', 'processed_pointclouds')


def load_all_pointclouds(pig_id):
    """加载一头猪所有帧的点云"""
    pcd_dir = get_processed_dir(pig_id)
    txt_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.txt')])

    all_points = []
    for f in txt_files:
        pts = np.loadtxt(os.path.join(pcd_dir, f))
        if len(pts) > 100:
            all_points.append(pts)
    return all_points


def extract_features(points):
    """
    从单帧归一化点云中提取统计特征
    注意：点云已归一化到单位球，绝对尺度信息已丢失
    """
    features = {}

    # 基本统计
    features['n_points'] = len(points)
    features['x_std'] = np.std(points[:, 0])
    features['y_std'] = np.std(points[:, 1])
    features['z_std'] = np.std(points[:, 2])

    # 各轴范围
    features['x_range'] = np.ptp(points[:, 0])
    features['y_range'] = np.ptp(points[:, 1])
    features['z_range'] = np.ptp(points[:, 2])

    # 形状比率 (扁平度、拉伸度)
    ranges = sorted([features['x_range'], features['y_range'], features['z_range']])
    features['flatness'] = ranges[0] / (ranges[2] + 1e-8)  # 最小/最大轴比
    features['elongation'] = ranges[1] / (ranges[2] + 1e-8)  # 中间/最大轴比

    # PCA 特征值比率 (形状描述符)
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]  # 降序
    total_var = np.sum(eigenvalues)
    features['pca_ratio_1'] = eigenvalues[0] / (total_var + 1e-8)  # 第一主成分占比
    features['pca_ratio_2'] = eigenvalues[1] / (total_var + 1e-8)  # 第二主成分占比
    features['pca_ratio_3'] = eigenvalues[2] / (total_var + 1e-8)  # 第三主成分占比
    features['linearity'] = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-8)
    features['planarity'] = (eigenvalues[1] - eigenvalues[2]) / (eigenvalues[0] + 1e-8)
    features['sphericity'] = eigenvalues[2] / (eigenvalues[0] + 1e-8)

    # Z轴分布 (归一化后Z轴仍保留相对深度信息)
    features['z_mean'] = np.mean(points[:, 2])
    features['z_skew'] = stats.skew(points[:, 2])
    features['z_kurtosis'] = stats.kurtosis(points[:, 2])

    # 表面粗糙度估计 (用最近邻距离的方差近似)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=6)  # 包含自身
    nn_dists = dists[:, 1:]  # 去掉自身
    features['roughness_mean'] = np.mean(nn_dists)
    features['roughness_std'] = np.std(nn_dists)

    # 点密度不均匀性
    features['density_cv'] = np.std(nn_dists.mean(axis=1)) / (np.mean(nn_dists.mean(axis=1)) + 1e-8)

    return features


# ============================================================
# 主分析
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  快速可行性验证：点云特征 vs 背膘值 相关性分析")
    print("=" * 70)

    # 1. 加载所有猪的点云并提取特征
    pig_features = {}

    for pig_id, backfat in BACKFAT_LABELS.items():
        print(f"\n处理 {pig_id} (背膘={backfat}mm)...")
        point_clouds = load_all_pointclouds(pig_id)
        print(f"  加载了 {len(point_clouds)} 帧")

        # 对每帧提取特征
        frame_features = []
        for pts in point_clouds:
            feat = extract_features(pts)
            frame_features.append(feat)

        # 所有帧取平均
        avg_features = {}
        for key in frame_features[0].keys():
            values = [f[key] for f in frame_features]
            avg_features[key] = np.mean(values)
            avg_features[f'{key}_cv'] = np.std(values) / (np.mean(values) + 1e-8)  # 帧间变异系数

        pig_features[pig_id] = avg_features
        print(f"  提取了 {len(avg_features)} 个特征")

    # 2. 相关性分析
    print(f"\n{'=' * 70}")
    print(f"  相关性分析结果 (N={len(BACKFAT_LABELS)})")
    print(f"{'=' * 70}")

    backfat_values = np.array([BACKFAT_LABELS[pid] for pid in BACKFAT_LABELS])
    feature_names = sorted(pig_features[list(pig_features.keys())[0]].keys())

    # 计算每个特征与背膘值的相关系数
    results = []
    for feat_name in feature_names:
        feat_values = np.array([pig_features[pid][feat_name] for pid in BACKFAT_LABELS])

        # Pearson 相关系数
        if np.std(feat_values) < 1e-10:
            continue
        r, p = stats.pearsonr(feat_values, backfat_values)
        # Spearman 秩相关
        rho, p_spearman = stats.spearmanr(feat_values, backfat_values)

        results.append((feat_name, r, p, rho, p_spearman, feat_values))

    # 按|r|排序，显示前20个
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'特征名':<30} {'Pearson r':>10} {'p值':>10} {'Spearman ρ':>12} {'p值':>10}")
    print("-" * 80)
    for feat_name, r, p, rho, p_sp, _ in results[:25]:
        marker = " ***" if abs(r) > 0.7 else (" **" if abs(r) > 0.5 else "")
        print(f"{feat_name:<30} {r:>10.4f} {p:>10.4f} {rho:>12.4f} {p_sp:>10.4f}{marker}")

    # 3. 详细展示最强相关特征
    print(f"\n{'=' * 70}")
    print(f"  Top 5 最强相关特征详情")
    print(f"{'=' * 70}")

    for feat_name, r, p, rho, p_sp, feat_values in results[:5]:
        print(f"\n{feat_name} (r={r:.4f}, p={p:.4f}):")
        for pid in BACKFAT_LABELS:
            bf = BACKFAT_LABELS[pid]
            fv = pig_features[pid][feat_name]
            print(f"  {pid}: 背膘={bf:>5.1f}mm, 特征={fv:.6f}")

    # 4. 总结
    print(f"\n{'=' * 70}")
    print(f"  结论")
    print(f"{'=' * 70}")

    strong = [r for r in results if abs(r[1]) > 0.7]
    moderate = [r for r in results if 0.5 < abs(r[1]) <= 0.7]
    weak = [r for r in results if 0.3 < abs(r[1]) <= 0.5]

    print(f"  强相关 (|r|>0.7): {len(strong)} 个特征")
    print(f"  中等相关 (0.5<|r|≤0.7): {len(moderate)} 个特征")
    print(f"  弱相关 (0.3<|r|≤0.5): {len(weak)} 个特征")
    print(f"  无相关 (|r|≤0.3): {len(results) - len(strong) - len(moderate) - len(weak)} 个特征")

    if len(strong) > 0:
        print(f"\n  ✅ 存在强相关特征，技术路线可能可行！")
        print(f"     但注意：N={len(BACKFAT_LABELS)}很小，结论需谨慎。")
    elif len(moderate) > 0:
        print(f"\n  ⚠️  存在中等相关特征，技术路线有一定可能性。")
        print(f"     需要更多数据验证。")
    else:
        print(f"\n  ❌ 未发现有意义的相关性，需重新审视技术路线。")

    print(f"\n  ⚠️  重要提醒:")
    print(f"     - 仅5头猪，样本量极小，相关性可能是偶然的")
    print(f"     - 点云已归一化，丢失了绝对尺寸信息")
    print(f"     - 深度学习模型可能学到更复杂的非线性特征")
    print(f"     - 这只是最粗略的检验，不能完全否定技术路线")
