"""
点云投影图数据集生成脚本 — 3D点云 → 多视角2D投影 → CNN回归
================================================================
将处理后的3D点云（processed_pointclouds_v2/）投影为3通道2D图像，
用于 CNN 背膘回归。

投影方案（3通道，类似RGB）：
  Channel 0 — 鸟瞰图 (BEV):  XY平面投影，像素值 = Z坐标（高度信息）
  Channel 1 — 正视图 (Front): XZ平面投影，像素值 = Y坐标（宽度/体宽）
  Channel 2 — 侧视图 (Side):  YZ平面投影，像素值 = X坐标（长度/体长）

论文叙事：基于3D点云的多视角投影特征 + CNN回归，
        类似 PointPillars / BEV 系列的投影变换思路。

输入：processed_pointclouds_v2/ 目录下的 .txt 点云文件
输出：projection_dataset/ 目录下的 .npy 文件 (3, 256, 256) + metadata.csv

不覆盖任何已有数据。

Author: 耿来硕 + Claude
Date: 2026-04-02
"""

import os
import re
import csv
import time
import numpy as np
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

DK_BASE = r'G:\耿来硕数据\DK'
FMH_BASE = r'G:\耿来硕数据\丰美禾数据'
OUTPUT_DIR = r'F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\projection_dataset'

IMG_SIZE = 256       # 投影图分辨率
PADDING = 0.05       # 坐标范围两侧留 5% padding


# ============================================================
# 标签加载（与其他脚本一致）
# ============================================================

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
                        labels[f'{dong_num}-{ear_short}'] = bf
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


# ============================================================
# 数据注册表
# ============================================================

def build_pig_registry():
    """扫描所有 processed_pointclouds_v2/ 目录，匹配标签。"""
    registry = {}
    BACKFAT_MIN, BACKFAT_MAX = 5.0, 40.0

    # === nkyfirst ===
    labels1 = load_labels_from_docx_nky1(os.path.join(DK_BASE, '第一次数据7.29.docx'))
    nky1_dir = os.path.join(DK_BASE, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        for pig_id in os.listdir(nky1_dir):
            if pig_id not in labels1:
                continue
            bf = labels1[pig_id]
            if not (BACKFAT_MIN <= bf <= BACKFAT_MAX):
                continue
            pp = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
            if os.path.isdir(pp):
                files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                if files:
                    registry[f'nky1_{pig_id}'] = {'pc_dir': pp, 'backfat': bf, 'source': 'nkyfirst'}

    # === nkysecond ===
    labels2 = load_labels_from_docx_nky2(os.path.join(DK_BASE, '第二次数据9.6.docx'))
    nky2_dir = os.path.join(DK_BASE, 'nkysecond')
    if os.path.isdir(nky2_dir):
        for pig_id in os.listdir(nky2_dir):
            if pig_id not in labels2:
                continue
            bf = labels2[pig_id]
            if not (BACKFAT_MIN <= bf <= BACKFAT_MAX):
                continue
            if pig_id.startswith('6-'):
                master = 'master10'
            elif pig_id.startswith('7-'):
                master = 'master'
            else:
                master = None
                for m in ['master10', 'master']:
                    pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', m, 'processed_pointclouds_v2')
                    if os.path.isdir(pp):
                        master = m
                        break
                if master is None:
                    continue
            pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', master, 'processed_pointclouds_v2')
            if os.path.isdir(pp):
                files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                if files:
                    registry[f'nky2_{pig_id}'] = {'pc_dir': pp, 'backfat': bf, 'source': 'nkysecond'}

    # === nkythird ===
    labels3 = load_labels_from_docx_nky3(os.path.join(DK_BASE, '第三次数据12.9.docx'))
    nky3_dir = os.path.join(DK_BASE, 'nkythird')
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
                label_key = f'{m.group(1)}-{m.group(2)}'
                if label_key not in labels3:
                    continue
                bf = labels3[label_key]
                if not (BACKFAT_MIN <= bf <= BACKFAT_MAX):
                    continue
                pp = os.path.join(pig_path, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
                if os.path.isdir(pp):
                    files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                    if files:
                        registry[f'nky3_{pig_dir}'] = {'pc_dir': pp, 'backfat': bf, 'source': 'nkythird'}

    # === 丰美禾 ===
    fmh_dates = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                 '11.1', '11.2', '11.3', '11.4', '11.5']
    xlsx_path = os.path.join(FMH_BASE, '背膘记录表.xlsx')
    fmh_entries = load_labels_from_xlsx_fmh(xlsx_path) if os.path.isfile(xlsx_path) else []
    xlsx_nov_path = os.path.join(FMH_BASE, '11.10处理背膘数据.xlsx')
    fmh_nov_entries = load_labels_from_xlsx_fmh_nov(xlsx_nov_path) if os.path.isfile(xlsx_nov_path) else []
    all_fmh_entries = fmh_entries + fmh_nov_entries

    for date_str in fmh_dates:
        date_dir = os.path.join(FMH_BASE, date_str)
        if not os.path.isdir(date_dir):
            continue
        for pig_dirname in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dirname)
            if not os.path.isdir(pig_path):
                continue
            matched_bf = None
            for seq, tag, bf in all_fmh_entries:
                tag_clean = tag.strip()
                if tag_clean and tag_clean in pig_dirname:
                    matched_bf = bf
                    break
            if matched_bf is None or not (BACKFAT_MIN <= matched_bf <= BACKFAT_MAX):
                continue
            pp = os.path.join(pig_path, 'master', 'processed_pointclouds_v2')
            if os.path.isdir(pp):
                files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                if files:
                    registry[f'fmh_{date_str}_{pig_dirname}'] = {
                        'pc_dir': pp, 'backfat': matched_bf, 'source': f'fmh_{date_str}'
                    }

    return registry


# ============================================================
# 点云 → 三视角投影
# ============================================================

def pointcloud_to_projection(pc, img_size=256, padding=0.05):
    """
    将点云投影为3通道图像：BEV(XY→Z), Front(XZ→Y), Side(YZ→X)。

    Args:
        pc: (N, 3) float32, center-only归一化后的点云
        img_size: 输出图像尺寸
        padding: 坐标范围两侧留白比例

    Returns:
        projection: (3, img_size, img_size) float32, 各通道值在[0,1]
    """
    if len(pc) < 10:
        return None

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    # 统一坐标范围（加 padding）
    def coord_range(vals):
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin
        if span < 1e-6:
            span = 1.0
        pad = span * padding
        return vmin - pad, vmax + pad

    x_min, x_max = coord_range(x)
    y_min, y_max = coord_range(y)
    z_min, z_max = coord_range(z)

    def project_to_grid(axis1, axis2, values, a1_min, a1_max, a2_min, a2_max):
        """将点投影到2D网格，每个格子取 values 的均值。"""
        grid = np.zeros((img_size, img_size), dtype=np.float32)
        count = np.zeros((img_size, img_size), dtype=np.float32)

        # 量化到网格坐标
        col = ((axis1 - a1_min) / (a1_max - a1_min) * (img_size - 1)).astype(np.int32)
        row = ((axis2 - a2_min) / (a2_max - a2_min) * (img_size - 1)).astype(np.int32)

        # 裁剪到合法范围
        col = np.clip(col, 0, img_size - 1)
        row = np.clip(row, 0, img_size - 1)

        # 累加（用均值，比最大值更平滑）
        for i in range(len(values)):
            r, c = row[i], col[i]
            grid[r, c] += values[i]
            count[r, c] += 1

        # 均值
        valid = count > 0
        grid[valid] /= count[valid]

        # 归一化到 [0, 1]
        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 1e-6:
            grid[valid] = (grid[valid] - v_min) / (v_max - v_min)
        else:
            grid[valid] = 0.5

        return grid

    # Channel 0: 鸟瞰图 (BEV) — XY 平面，值 = Z（高度）
    ch0 = project_to_grid(x, y, z, x_min, x_max, y_min, y_max)

    # Channel 1: 正视图 (Front) — XZ 平面，值 = Y（宽度/深度）
    ch1 = project_to_grid(x, z, y, x_min, x_max, z_min, z_max)

    # Channel 2: 侧视图 (Side) — YZ 平面，值 = X（长度方向）
    ch2 = project_to_grid(y, z, x, y_min, y_max, z_min, z_max)

    projection = np.stack([ch0, ch1, ch2], axis=0)  # (3, H, W)
    return projection


def pointcloud_to_projection_fast(pc, img_size=256, padding=0.05):
    """
    向量化版本的投影函数（用 np.add.at 代替 Python 循环）。
    """
    if len(pc) < 10:
        return None

    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

    def coord_range(vals):
        vmin, vmax = vals.min(), vals.max()
        span = max(vmax - vmin, 1e-6)
        pad = span * padding
        return vmin - pad, vmax + pad

    x_min, x_max = coord_range(x)
    y_min, y_max = coord_range(y)
    z_min, z_max = coord_range(z)

    def project_fast(axis1, axis2, values, a1_min, a1_max, a2_min, a2_max):
        grid = np.zeros((img_size, img_size), dtype=np.float64)
        count = np.zeros((img_size, img_size), dtype=np.float64)

        col = np.clip(((axis1 - a1_min) / (a1_max - a1_min) * (img_size - 1)).astype(np.int32),
                       0, img_size - 1)
        row = np.clip(((axis2 - a2_min) / (a2_max - a2_min) * (img_size - 1)).astype(np.int32),
                       0, img_size - 1)

        np.add.at(grid, (row, col), values.astype(np.float64))
        np.add.at(count, (row, col), 1.0)

        valid = count > 0
        grid[valid] /= count[valid]

        v_min, v_max = values.min(), values.max()
        if v_max - v_min > 1e-6:
            grid[valid] = (grid[valid] - v_min) / (v_max - v_min)
        else:
            grid[valid] = 0.5

        return grid.astype(np.float32)

    ch0 = project_fast(x, y, z, x_min, x_max, y_min, y_max)
    ch1 = project_fast(x, z, y, x_min, x_max, z_min, z_max)
    ch2 = project_fast(y, z, x, y_min, y_max, z_min, z_max)

    return np.stack([ch0, ch1, ch2], axis=0)


# ============================================================
# 主处理流程
# ============================================================

def process_one_pig(pig_uid, info, output_dir):
    """处理一头猪的所有帧。返回 [(frame_path, backfat), ...]。"""
    pc_dir = info['pc_dir']
    backfat = info['backfat']

    files = sorted([f for f in os.listdir(pc_dir) if f.endswith('.txt')])
    if not files:
        return []

    pig_out_dir = os.path.join(output_dir, pig_uid)
    os.makedirs(pig_out_dir, exist_ok=True)

    saved = []
    for idx, fname in enumerate(files):
        pc_path = os.path.join(pc_dir, fname)
        try:
            pc = np.loadtxt(pc_path, dtype=np.float32)
        except Exception:
            continue

        if pc.ndim != 2 or pc.shape[1] < 3:
            continue
        pc = pc[:, :3]

        proj = pointcloud_to_projection_fast(pc, IMG_SIZE, PADDING)
        if proj is None:
            continue

        frame_name = f'frame_{idx:03d}.npy'
        save_path = os.path.join(pig_out_dir, frame_name)
        np.save(save_path, proj)
        saved.append((os.path.join(pig_uid, frame_name), backfat))

    return saved


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("3D Point Cloud -> Multi-view Projection Dataset")
    print("BEV(XY->Z) + Front(XZ->Y) + Side(YZ->X) = 3 channels")
    print("=" * 60)

    # 1. 构建注册表
    print("\n[1/2] Building pig registry (processed_pointclouds_v2)...")
    registry = build_pig_registry()
    print(f"  Found {len(registry)} pigs")

    source_counts = defaultdict(int)
    for info in registry.values():
        source_counts[info['source']] += 1
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt}")

    # 2. 处理
    print(f"\n[2/2] Generating projection images ({IMG_SIZE}x{IMG_SIZE}, 3ch)...")

    all_records = []
    success_count = 0
    fail_count = 0

    pig_items = sorted(registry.items())
    for i, (pig_uid, info) in enumerate(pig_items, 1):
        if i % 100 == 0 or i <= 3:
            elapsed = time.time() - start_time
            eta = elapsed / i * (len(pig_items) - i)
            print(f"  [{i}/{len(pig_items)}] {pig_uid} "
                  f"(elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)")

        saved = process_one_pig(pig_uid, info, OUTPUT_DIR)
        if saved:
            success_count += 1
            for frame_path, bf in saved:
                all_records.append((frame_path, pig_uid, bf, info['source']))
        else:
            fail_count += 1

    # 3. 保存 metadata
    csv_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_path', 'pig_id', 'backfat', 'source'])
        for rec in all_records:
            writer.writerow(rec)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Success: {success_count} pigs")
    print(f"  Failed:  {fail_count} pigs")
    print(f"  Frames:  {len(all_records)}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Time:    {total_time:.1f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
