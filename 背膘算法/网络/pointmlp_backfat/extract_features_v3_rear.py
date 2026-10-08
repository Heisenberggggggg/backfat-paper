"""
臀部视角点云统计特征提取脚本 v3
================================
背景：
    extract_features_v2.py 从背部视角（master）的 processed_pointclouds_v2/ 提取 41 维统计特征，
    用于单视角 XGBoost/RF/Lasso 回归，最终 R²≈0.08~0.15。

问题分析：
    单一背部视角的信息量不足以准确预测皮下背膘厚度。

解决思路：
    从臀部视角（sub）的 processed_pointclouds_v3_rear/ 同样提取 41 维统计特征，
    后续与背部特征拼接成 82 维向量做双视角早期融合回归（train_traditional_v3_dualview.py）。

与 extract_features_v2.py 的区别：
    - 扫描目录: processed_pointclouds_v2 → processed_pointclouds_v3_rear
    - 搜索路径: master/ → sub/sub2/sub10/
    - 输出目录: traditional_ml_v2 → traditional_ml_v3
    - 输出文件: features_rear_per_frame.csv, features_rear_per_pig.csv

Author: 耿来硕 + Claude
Date: 2026-03-31
"""

import numpy as np
import os
import re
import time
import csv
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA


# ==============================================================================
# 标签加载函数（与 extract_features_v2.py 完全一致）
# ==============================================================================

def load_labels_from_docx_nky1(docx_path):
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 9:
                ear_tag = cells[1]
                bf_str = cells[8]
                if ear_tag and bf_str and '-' in ear_tag:
                    try:
                        labels[ear_tag] = float(bf_str)
                    except ValueError:
                        pass
    return labels


def load_labels_from_docx_nky2(docx_path):
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 13:
                ear_tag = cells[1]
                bf_str = cells[11] if cells[11] else (cells[12] if len(cells) > 12 else '')
                if ear_tag and bf_str and '-' in ear_tag:
                    try:
                        labels[ear_tag] = float(bf_str)
                    except ValueError:
                        pass
    return labels


def load_labels_from_docx_nky3(docx_path):
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    dong_num = None
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            header_text = ' '.join(cells)
            for d in [10, 11, 12, 13]:
                if f'第{d}栋' in header_text or f'{d}栋' in header_text:
                    dong_num = d
                    break
            if dong_num and len(cells) >= 6:
                ear_short = cells[1]
                bf_str = cells[5]
                if ear_short and bf_str and cells[0].isdigit():
                    try:
                        bf = float(bf_str)
                        full_id = f'{dong_num}-{ear_short}'
                        labels[full_id] = bf
                    except ValueError:
                        pass
    return labels


def load_labels_from_xlsx_fmh(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Sheet1']
    entries = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        cells = [str(c).strip() if c else '' for c in row[:3]]
        seq_str, tag, bf_str = cells
        if seq_str and tag and bf_str:
            try:
                entries.append((int(seq_str), tag, float(bf_str)))
            except ValueError:
                pass
    return entries


def load_labels_from_xlsx_fmh_nov(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    entries = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        cells = [str(c).strip() if c else '' for c in row[:3]]
        seq_str, tag, bf_str = cells
        if seq_str and tag and bf_str:
            try:
                entries.append((int(seq_str), tag, float(bf_str)))
            except ValueError:
                pass
    return entries


# ==============================================================================
# 数据注册表构建 v3（臀部视角）
# ==============================================================================

def build_dataset_registry_v3_rear(dk_base, fmh_base):
    """扫描所有数据目录的 processed_pointclouds_v3_rear/，匹配标签。
    返回 {pig_uid: (data_dir, backfat, n_frames)}。
    """
    registry = {}

    # === nkyfirst ===
    labels1 = load_labels_from_docx_nky1(os.path.join(dk_base, '第一次数据7.29.docx'))
    nky1_dir = os.path.join(dk_base, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        for pig_id in os.listdir(nky1_dir):
            if pig_id in labels1:
                # 尝试 sub, sub2
                for subdir in ['sub', 'sub2']:
                    pp = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', subdir, 'processed_pointclouds_v3_rear')
                    if os.path.isdir(pp):
                        files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                        if files:
                            registry[f'nky1_{pig_id}'] = (pp, labels1[pig_id], len(files))
                            break

    # === nkysecond ===
    labels2 = load_labels_from_docx_nky2(os.path.join(dk_base, '第二次数据9.6.docx'))
    nky2_dir = os.path.join(dk_base, 'nkysecond')
    if os.path.isdir(nky2_dir):
        for pig_id in os.listdir(nky2_dir):
            if pig_id in labels2:
                # 6-X 用 sub10, 7-X 用 sub
                if pig_id.startswith('6-'):
                    sub_name = 'sub10'
                elif pig_id.startswith('7-'):
                    sub_name = 'sub'
                else:
                    sub_name = None
                    for s in ['sub10', 'sub']:
                        pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', s, 'processed_pointclouds_v3_rear')
                        if os.path.isdir(pp):
                            sub_name = s
                            break
                    if sub_name is None:
                        continue

                pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', sub_name, 'processed_pointclouds_v3_rear')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky2_{pig_id}'] = (pp, labels2[pig_id], len(files))

    # === nkythird ===
    labels3 = load_labels_from_docx_nky3(os.path.join(dk_base, '第三次数据12.9.docx'))
    nky3_dir = os.path.join(dk_base, 'nkythird')
    if os.path.isdir(nky3_dir):
        for kinect_dir in os.listdir(nky3_dir):
            kinect_path = os.path.join(nky3_dir, kinect_dir)
            if not os.path.isdir(kinect_path):
                continue
            for pig_dir in os.listdir(kinect_path):
                pig_path = os.path.join(kinect_path, pig_dir)
                if not os.path.isdir(pig_path):
                    continue
                if pig_dir.startswith('w') or pig_dir.startswith('wc'):
                    continue
                m = re.match(r'^[cwk]*(\d+)-(.+)$', pig_dir)
                if not m:
                    continue
                dong_num = m.group(1)
                suffix = m.group(2)
                label_key = f'{dong_num}-{suffix}'
                if label_key not in labels3:
                    continue
                pp = os.path.join(pig_path, 'depth_cz_r_d_p', 'sub', 'processed_pointclouds_v3_rear')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky3_{pig_dir}'] = (pp, labels3[label_key], len(files))

    # === 丰美禾 ALL 12 dates ===
    fmh_dates = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                 '11.1', '11.2', '11.3', '11.4', '11.5']

    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    fmh_entries = []
    if os.path.isfile(xlsx_path):
        fmh_entries = load_labels_from_xlsx_fmh(xlsx_path)

    xlsx_nov_path = os.path.join(fmh_base, '11.10处理背膘数据.xlsx')
    fmh_nov_entries = []
    if os.path.isfile(xlsx_nov_path):
        fmh_nov_entries = load_labels_from_xlsx_fmh_nov(xlsx_nov_path)

    all_fmh_entries = fmh_entries + fmh_nov_entries

    for date_str in fmh_dates:
        date_dir = os.path.join(fmh_base, date_str)
        if not os.path.isdir(date_dir):
            continue
        for pig_dirname in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dirname)
            if not os.path.isdir(pig_path):
                continue
            pp = os.path.join(pig_path, 'sub', 'processed_pointclouds_v3_rear')
            if not os.path.isdir(pp):
                continue
            files = [f for f in os.listdir(pp) if f.endswith('.txt')]
            if not files:
                continue
            for seq, tag, bf in all_fmh_entries:
                tag_clean = tag.strip()
                if tag_clean and tag_clean in pig_dirname:
                    uid = f'fmh_{date_str}_{pig_dirname}'
                    registry[uid] = (pp, bf, len(files))
                    break

    return registry


# ==============================================================================
# 特征提取函数（与 extract_features_v2.py 完全一致，41维特征）
# ==============================================================================

def extract_features_from_pointcloud(pc):
    features = {}
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    n_points = len(pc)

    for axis_name, axis_data in [('x', x), ('y', y), ('z', z)]:
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_range'] = np.ptp(axis_data)
        features[f'{axis_name}_skew'] = float(skew(axis_data))
        features[f'{axis_name}_kurtosis'] = float(kurtosis(axis_data))

    for axis_name, axis_data in [('z', z)]:
        q25, q50, q75 = np.percentile(axis_data, [25, 50, 75])
        features[f'{axis_name}_q25'] = q25
        features[f'{axis_name}_q50'] = q50
        features[f'{axis_name}_q75'] = q75
        features[f'{axis_name}_iqr'] = q75 - q25

    try:
        centroid = np.mean(pc, axis=0)
        pc_centered = pc - centroid
        _, s, vh = np.linalg.svd(pc_centered, full_matrices=False)
        normal = vh[2]
        dists_to_plane = pc_centered @ normal
        features['plane_residual_mean'] = np.mean(np.abs(dists_to_plane))
        features['plane_residual_std'] = np.std(dists_to_plane)
        features['plane_residual_max'] = np.max(np.abs(dists_to_plane))
        s_sum = np.sum(s) + 1e-8
        features['svd_ratio_1'] = s[0] / s_sum
        features['svd_ratio_2'] = s[1] / s_sum
        features['svd_ratio_3'] = s[2] / s_sum
    except Exception:
        features['plane_residual_mean'] = 0.0
        features['plane_residual_std'] = 0.0
        features['plane_residual_max'] = 0.0
        features['svd_ratio_1'] = 0.33
        features['svd_ratio_2'] = 0.33
        features['svd_ratio_3'] = 0.33

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

    try:
        hull = ConvexHull(pc)
        features['convex_hull_volume'] = hull.volume
        features['convex_hull_area'] = hull.area
        features['point_density'] = n_points / (hull.volume + 1e-8)
    except Exception:
        features['convex_hull_volume'] = 0.0
        features['convex_hull_area'] = 0.0
        features['point_density'] = 0.0

    cov_matrix = np.cov(pc.T)
    features['cov_xy'] = cov_matrix[0, 1]
    features['cov_xz'] = cov_matrix[0, 2]
    features['cov_yz'] = cov_matrix[1, 2]

    features['n_points'] = n_points

    return features


def get_feature_names():
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
    output_dir = os.path.join(script_dir, 'traditional_ml_v3')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'extract_features_rear_log.txt')

    def log(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("开始提取臀部视角点云统计特征 (v3 rear)")
    log("=" * 60)

    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    log(f"数据源: DK={dk_base}, FMH={fmh_base}")
    log(f"扫描目录: processed_pointclouds_v3_rear/")
    registry = build_dataset_registry_v3_rear(dk_base, fmh_base)
    log(f"匹配到 {len(registry)} 头猪")

    feature_names = get_feature_names()
    frame_header = ['pig_id', 'split', 'backfat', 'file_path'] + feature_names

    frame_csv_path = os.path.join(output_dir, 'features_rear_per_frame.csv')
    pig_csv_path = os.path.join(output_dir, 'features_rear_per_pig.csv')

    pig_features = {}
    total_files = sum(info[2] for info in registry.values())
    log(f"共需处理 {total_files} 个点云文件")

    processed = 0
    errors = 0
    start_time = time.time()

    with open(frame_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(frame_header)

        for pig_idx, (pig_id, (data_dir, backfat, n_frames)) in enumerate(sorted(registry.items())):
            split_label = 'unknown'
            pig_features[pig_id] = {
                'backfat': backfat,
                'split': split_label,
                'features': []
            }

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

    # 计算猪级别的平均特征
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

            avg_feat = {}
            for name in feature_names:
                values = [f[name] for f in feat_list]
                avg_feat[name] = np.mean(values)

            feat_values = [avg_feat[name] for name in feature_names]
            writer.writerow([pig_id, info['split'], info['backfat'], len(feat_list)] + feat_values)

    n_valid = len([p for p in pig_features if pig_features[p]['features']])
    log(f"猪级特征保存到: {pig_csv_path}")
    log(f"共 {n_valid} 头猪有有效特征")

    # 按数据源统计
    source_counts = {'nky1': 0, 'nky2': 0, 'nky3': 0, 'fmh': 0}
    for pid in registry:
        for prefix in source_counts:
            if pid.startswith(prefix):
                source_counts[prefix] += 1
                break
    log("\n各数据源匹配数量:")
    log(f"  nkyfirst:  {source_counts['nky1']} 头")
    log(f"  nkysecond: {source_counts['nky2']} 头")
    log(f"  nkythird:  {source_counts['nky3']} 头")
    log(f"  丰美禾:    {source_counts['fmh']} 头")

    log("\n" + "=" * 60)
    log("臀部视角特征提取完成！(v3 rear)")
    log(f"下一步: 运行 train_traditional_v3_dualview.py 进行双视角融合训练")
    log("=" * 60)


if __name__ == '__main__':
    main()
