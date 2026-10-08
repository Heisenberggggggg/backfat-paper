"""
点云统计特征提取脚本 v2
======================
v1 (extract_features.py) 使用 processed_pointclouds/ 目录下 189 头猪的点云，
该版本点云经过 unit-sphere 归一化（丢失了真实尺度信息），数据仅来自 nkyfirst + nkysecond
+ nkythird + 丰美禾 10.29。

v2 (本脚本) 使用 processed_pointclouds_v2/ 目录下的点云，该版本点云经过：
  - 去头处理（仅保留背部+臀部约85%的点）
  - center-only 归一化（保留真实尺度）
  - 每头猪最多 20 帧
数据源扩展到：
  - nkyfirst (39头)
  - nkysecond (21头)
  - nkythird (122头，6个kinect子目录)
  - 丰美禾 ALL 12个日期 (8.24~11.5)
标签来源扩展：增加 11.10处理背膘数据.xlsx 覆盖丰美禾 11.1~11.5 日期

改动总结：
  - 扫描路径: processed_pointclouds -> processed_pointclouds_v2
  - 标签来源: 4 -> 5 个文件（新增 11.10 xlsx）
  - 丰美禾数据: 仅 10.29 -> 12 个日期
  - 输出目录: traditional_ml -> traditional_ml_v2
  - 不依赖 split_info.json，所有猪标记为 'unknown'，由训练脚本决定划分

预期效果：匹配到 916+ 头猪（vs v1 的 189 头），特征质量更好（保留真实尺度）

Author: 耿来硕 + Claude
Date: 2026-03-25
"""

import numpy as np
import os
import re
import sys
import time
import csv
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA


# ==============================================================================
# 标签加载函数（从 train.py 复制，避免依赖）
# ==============================================================================

def load_labels_from_docx_nky1(docx_path):
    """读取第一次数据 docx，返回 {ear_tag: backfat}。"""
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
    """读取第二次数据 docx，返回 {ear_tag: backfat}。"""
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 13:
                ear_tag = cells[1]
                # B超背膘可能在第11或12列
                bf_str = cells[11] if cells[11] else (cells[12] if len(cells) > 12 else '')
                if ear_tag and bf_str and '-' in ear_tag:
                    try:
                        labels[ear_tag] = float(bf_str)
                    except ValueError:
                        pass
    return labels


def load_labels_from_docx_nky3(docx_path):
    """读取第三次数据 docx（按栋分表），返回 {dong-short_id: backfat}。"""
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
    """读取丰美禾背膘记录表，返回 [(seq, tag, backfat), ...]。"""
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
    """读取丰美禾 11.10处理背膘数据.xlsx（11.1~11.5 日期的额外标签）。
    返回 [(seq, tag, backfat), ...]，格式与 load_labels_from_xlsx_fmh 一致。
    """
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
# 数据注册表构建 v2
# ==============================================================================

def build_dataset_registry_v2(dk_base, fmh_base):
    """扫描所有数据目录的 processed_pointclouds_v2/，匹配标签。
    返回 {pig_uid: (data_dir, backfat, n_frames)}。
    """
    registry = {}

    # === nkyfirst ===
    # 路径: nkyfirst/{pig}/depth_cz_r_d_p/master/processed_pointclouds_v2/
    labels1 = load_labels_from_docx_nky1(os.path.join(dk_base, '第一次数据7.29.docx'))
    nky1_dir = os.path.join(dk_base, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        for pig_id in os.listdir(nky1_dir):
            if pig_id in labels1:
                pp = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky1_{pig_id}'] = (pp, labels1[pig_id], len(files))

    # === nkysecond ===
    # 路径: nkysecond/{pig}/depth_cz_r_d_p/{master10 or master}/processed_pointclouds_v2/
    # 6-X 猪用 master10, 7-X 猪用 master
    labels2 = load_labels_from_docx_nky2(os.path.join(dk_base, '第二次数据9.6.docx'))
    nky2_dir = os.path.join(dk_base, 'nkysecond')
    if os.path.isdir(nky2_dir):
        for pig_id in os.listdir(nky2_dir):
            if pig_id in labels2:
                if pig_id.startswith('6-'):
                    master_name = 'master10'
                elif pig_id.startswith('7-'):
                    master_name = 'master'
                else:
                    # 尝试两个
                    master_name = None
                    for m in ['master10', 'master']:
                        pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', m, 'processed_pointclouds_v2')
                        if os.path.isdir(pp):
                            master_name = m
                            break
                    if master_name is None:
                        continue

                pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', master_name, 'processed_pointclouds_v2')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky2_{pig_id}'] = (pp, labels2[pig_id], len(files))

    # === nkythird ===
    # 路径: nkythird/{kinect}/{pig}/depth_cz_r_d_p/master/processed_pointclouds_v2/
    # pig 目录名如 "13-255", "c13-277", "w10-143"
    # 用 re.match(r'^[cwk]*\d+-(.+)$', pig_dir) 提取后缀匹配标签
    # 跳过 "w" 和 "wc" 前缀目录（无标签的未标记猪）
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

                # 跳过 w 和 wc 前缀（无标签）
                if pig_dir.startswith('w') or pig_dir.startswith('wc'):
                    continue

                # 提取后缀匹配标签
                m = re.match(r'^[cwk]*(\d+)-(.+)$', pig_dir)
                if not m:
                    continue
                dong_num = m.group(1)
                suffix = m.group(2)
                # 构造 labels3 的 key: "dong_num-suffix"
                label_key = f'{dong_num}-{suffix}'

                if label_key not in labels3:
                    continue

                pp = os.path.join(pig_path, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky3_{pig_dir}'] = (pp, labels3[label_key], len(files))

    # === 丰美禾 ALL 12 dates ===
    # 路径: 丰美禾数据/{date}/{pig}/master/processed_pointclouds_v2/
    fmh_dates = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                 '11.1', '11.2', '11.3', '11.4', '11.5']

    # 加载主标签文件
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    fmh_entries = []
    if os.path.isfile(xlsx_path):
        fmh_entries = load_labels_from_xlsx_fmh(xlsx_path)

    # 加载 11.1~11.5 额外标签文件
    xlsx_nov_path = os.path.join(fmh_base, '11.10处理背膘数据.xlsx')
    fmh_nov_entries = []
    if os.path.isfile(xlsx_nov_path):
        fmh_nov_entries = load_labels_from_xlsx_fmh_nov(xlsx_nov_path)

    # 合并标签：(seq, tag, backfat)  列表
    all_fmh_entries = fmh_entries + fmh_nov_entries

    for date_str in fmh_dates:
        date_dir = os.path.join(fmh_base, date_str)
        if not os.path.isdir(date_dir):
            continue

        for pig_dirname in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dirname)
            if not os.path.isdir(pig_path):
                continue

            pp = os.path.join(pig_path, 'master', 'processed_pointclouds_v2')
            if not os.path.isdir(pp):
                continue

            files = [f for f in os.listdir(pp) if f.endswith('.txt')]
            if not files:
                continue

            # 匹配标签：检查 ear_tag 是否出现在目录名中
            matched = False
            for seq, tag, bf in all_fmh_entries:
                tag_clean = tag.strip()
                if tag_clean and tag_clean in pig_dirname:
                    uid = f'fmh_{date_str}_{pig_dirname}'
                    registry[uid] = (pp, bf, len(files))
                    matched = True
                    break

    return registry


# ==============================================================================
# 特征提取函数（与 extract_features.py 完全一致，41维特征）
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
    output_dir = os.path.join(script_dir, 'traditional_ml_v2')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'extract_features_log.txt')

    def log(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("开始提取点云统计特征 (v2)")
    log("=" * 60)

    # 1. 构建数据注册表 v2
    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    log(f"数据源: DK={dk_base}, FMH={fmh_base}")
    log(f"扫描目录: processed_pointclouds_v2/")
    registry = build_dataset_registry_v2(dk_base, fmh_base)
    log(f"匹配到 {len(registry)} 头猪")

    # v2 不依赖 split_info.json，所有猪标记为 'unknown'
    log("注意: v2 不使用 split_info.json，所有猪标记为 'unknown' split")

    # 2. 提取特征
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
            split_label = 'unknown'
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

    # 3. 计算猪级别的平均特征
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

    # 4. 打印特征统计概览
    log("\n特征维度: {}".format(len(feature_names)))
    log("特征列表:")
    for i, name in enumerate(feature_names):
        log(f"  [{i+1:2d}] {name}")

    # 5. 按数据源统计
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
    log("特征提取完成！(v2)")
    log(f"下一步: 运行 train_traditional_v2.py 进行模型训练")
    log("=" * 60)


if __name__ == '__main__':
    main()
