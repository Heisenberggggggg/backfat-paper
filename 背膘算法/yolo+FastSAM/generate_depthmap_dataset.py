"""
深度图数据集生成脚本 — 用于 CNN 背膘回归
==========================================
从 Azure Kinect 原始数据中生成裁剪后的深度图数据集。

流程：
  1. 遍历所有4个数据源（nkyfirst/second/third + 丰美禾 12个日期）
  2. 对每帧：加载 RGB → YOLO 检测 → FastSAM 分割 → 得到 mask
  3. 加载对应 16-bit 深度图 (.png)，用 mask 裁出猪体区域
  4. 裁到 bounding box → resize 到 256×256 → 背景填 0
  5. 保存为 .npy（float32, 单位 mm）到 depthmap_dataset/ 目录
  6. 生成 metadata.csv（pig_id, frame_path, backfat, source）

数据源:
  - G:\耿来硕数据\DK\nkyfirst    (39头)
  - G:\耿来硕数据\DK\nkysecond   (21头)
  - G:\耿来硕数据\DK\nkythird    (122头)
  - G:\耿来硕数据\丰美禾数据\*   (12个日期, 共 ~756头)

不覆盖任何已有数据，输出到全新的 depthmap_dataset/ 目录。

Author: 耿来硕 + Claude
Date: 2026-04-01
"""

import os
import sys
import re
import cv2
import csv
import json
import time
import numpy as np
from pathlib import Path

# ============================================================
# 配置
# ============================================================

DK_BASE = r'G:\耿来硕数据\DK'
FMH_BASE = r'G:\耿来硕数据\丰美禾数据'
OUTPUT_DIR = r'F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\depthmap_dataset'

YOLO_MODEL_PATH = 'yolo11n.pt'       # 通用 YOLO11n（与其他脚本一致）
FASTSAM_MODEL_PATH = 'FastSAM-s.pt'  # FastSAM-s 分割模型

TARGET_SIZE = 256          # 输出深度图尺寸 (256×256)
FRAMES_PER_PIG = 20        # 每头猪均匀采样帧数
YOLO_CONF = 0.15           # YOLO 检测置信度（低阈值保证召回）
FASTSAM_CONF = 0.4         # FastSAM 分割置信度
DEPTH_SCALE = 1000.0       # 深度图原始值(mm)→米的除数
MIN_DEPTH_MM = 300.0       # 最小有效深度 (mm)
MAX_DEPTH_MM = 3000.0      # 最大有效深度 (mm)
MARGIN_RATIO = 0.05        # fallback 时的边缘比例


# ============================================================
# 标签加载（复用现有逻辑）
# ============================================================

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
    """读取丰美禾 11.10处理背膘数据.xlsx。"""
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
# 图像 I/O（支持中文路径）
# ============================================================

def imread_chinese(path, flags=cv2.IMREAD_COLOR):
    """读取中文路径图片。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def imread_depth_chinese(path):
    """读取16-bit深度图（中文路径）。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


# ============================================================
# YOLO + FastSAM 检测/分割
# ============================================================

def load_models():
    """加载 YOLO 和 FastSAM 模型。"""
    from ultralytics import YOLO
    print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"Loading FastSAM model: {FASTSAM_MODEL_PATH}")
    fastsam_model = YOLO(FASTSAM_MODEL_PATH)
    return yolo_model, fastsam_model


def get_pig_mask(color_img, yolo_model, fastsam_model):
    """
    对一帧 RGB 图像做 YOLO 检测 + FastSAM 分割，返回二值 mask。
    返回: mask (H, W) uint8, 1=猪体 0=背景; 如果失败返回 None
    """
    h, w = color_img.shape[:2]

    # --- YOLO 检测 ---
    results = yolo_model(color_img, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        # fallback: 用整张图减去边缘
        margin_x = int(w * MARGIN_RATIO)
        margin_y = int(h * MARGIN_RATIO)
        x1, y1, x2, y2 = margin_x, margin_y, w - margin_x, h - margin_y
    else:
        # 取面积最大的检测框
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        best_idx = areas.argmax().item()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # --- FastSAM 分割 ---
    try:
        sam_results = fastsam_model(color_img, retina_masks=True, conf=FASTSAM_CONF, verbose=False)
        if sam_results[0].masks is None or len(sam_results[0].masks) == 0:
            return None

        masks = sam_results[0].masks.data.cpu().numpy()  # (N, H', W')

        # 找与 YOLO bbox 重叠最大的 mask
        best_mask = None
        best_overlap = 0
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            # 计算与 bbox 的重叠
            roi = m[y1:y2, x1:x2]
            overlap = roi.sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_mask = m

        if best_mask is None:
            return None

        return best_mask.astype(np.uint8)

    except Exception as e:
        print(f"  FastSAM error: {e}")
        return None


# ============================================================
# 深度图裁剪与处理
# ============================================================

def crop_masked_depthmap(depth_img, mask, target_size=256):
    """
    用 mask 裁出深度图的猪体区域，resize 到 target_size × target_size。

    Args:
        depth_img: (H, W) uint16, 原始深度值 (mm)
        mask: (H, W) uint8, 1=猪体
        target_size: 输出尺寸

    Returns:
        cropped: (target_size, target_size) float32, 单位 mm, 背景=0
    """
    # 应用 mask
    masked = depth_img.astype(np.float32) * mask.astype(np.float32)

    # 过滤无效深度值
    valid = (masked > MIN_DEPTH_MM) & (masked < MAX_DEPTH_MM)
    masked[~valid] = 0

    # 找 bounding box
    ys, xs = np.where(masked > 0)
    if len(ys) < 100:  # 有效像素太少
        return None

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    # 裁出 ROI
    roi = masked[y1:y2, x1:x2]

    # 保持宽高比 resize 到 target_size
    rh, rw = roi.shape
    scale = target_size / max(rh, rw)
    new_h, new_w = int(rh * scale), int(rw * scale)

    # 用 INTER_AREA（缩小时抗锯齿）或 INTER_LINEAR（放大时平滑）
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(roi, (new_w, new_h), interpolation=interp)

    # 放到 target_size × target_size 画布中央
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


# ============================================================
# 数据注册表构建
# ============================================================

def build_pig_registry():
    """
    扫描所有数据源，匹配标签，返回:
    {pig_uid: {
        'color_dir': str,
        'depth_dir': str,
        'backfat': float,
        'source': str
    }}
    """
    registry = {}
    BACKFAT_MIN, BACKFAT_MAX = 5.0, 40.0  # 过滤异常值

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
            base = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', 'master')
            color_dir = os.path.join(base, 'color')
            depth_dir = os.path.join(base, 'depth')
            if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
                registry[f'nky1_{pig_id}'] = {
                    'color_dir': color_dir, 'depth_dir': depth_dir,
                    'backfat': bf, 'source': 'nkyfirst'
                }

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
            # 6-X 用 master10, 7-X 用 master
            if pig_id.startswith('6-'):
                master = 'master10'
            elif pig_id.startswith('7-'):
                master = 'master'
            else:
                master = None
                for m in ['master10', 'master']:
                    if os.path.isdir(os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', m, 'depth')):
                        master = m
                        break
                if master is None:
                    continue
            base = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', master)
            color_dir = os.path.join(base, 'color')
            depth_dir = os.path.join(base, 'depth')
            if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
                registry[f'nky2_{pig_id}'] = {
                    'color_dir': color_dir, 'depth_dir': depth_dir,
                    'backfat': bf, 'source': 'nkysecond'
                }

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
                dong_num = m.group(1)
                suffix = m.group(2)
                label_key = f'{dong_num}-{suffix}'
                if label_key not in labels3:
                    continue
                bf = labels3[label_key]
                if not (BACKFAT_MIN <= bf <= BACKFAT_MAX):
                    continue
                base = os.path.join(pig_path, 'depth_cz_r_d_p', 'master')
                color_dir = os.path.join(base, 'color')
                depth_dir = os.path.join(base, 'depth')
                if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
                    registry[f'nky3_{pig_dir}'] = {
                        'color_dir': color_dir, 'depth_dir': depth_dir,
                        'backfat': bf, 'source': 'nkythird'
                    }

    # === 丰美禾 ALL 12 dates ===
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
            # 匹配标签
            matched_bf = None
            for seq, tag, bf in all_fmh_entries:
                tag_clean = tag.strip()
                if tag_clean and tag_clean in pig_dirname:
                    matched_bf = bf
                    break
            if matched_bf is None:
                continue
            if not (BACKFAT_MIN <= matched_bf <= BACKFAT_MAX):
                continue

            base = os.path.join(pig_path, 'master')
            color_dir = os.path.join(base, 'color')
            depth_dir = os.path.join(base, 'depth')
            if os.path.isdir(color_dir) and os.path.isdir(depth_dir):
                registry[f'fmh_{date_str}_{pig_dirname}'] = {
                    'color_dir': color_dir, 'depth_dir': depth_dir,
                    'backfat': matched_bf, 'source': f'fmh_{date_str}'
                }

    return registry


# ============================================================
# 主处理流程
# ============================================================

def get_frame_pairs(color_dir, depth_dir, n_frames=20):
    """
    获取 color/depth 的对齐帧列表，均匀采样 n_frames 帧。
    返回 [(color_path, depth_path), ...]
    """
    # 深度图文件列表
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    if not depth_files:
        return []

    # 颜色图文件列表（.jpg 或 .png）
    color_files = sorted([f for f in os.listdir(color_dir)
                          if f.endswith('.jpg') or f.endswith('.png')])
    if not color_files:
        return []

    # 按文件名匹配（去掉后缀）
    color_stems = {os.path.splitext(f)[0]: f for f in color_files}
    depth_stems = {os.path.splitext(f)[0]: f for f in depth_files}

    # 找公共帧
    common_stems = sorted(set(color_stems.keys()) & set(depth_stems.keys()))
    if not common_stems:
        return []

    # 均匀采样
    total = len(common_stems)
    if total <= n_frames:
        selected = common_stems
    else:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        selected = [common_stems[i] for i in indices]

    pairs = []
    for stem in selected:
        cp = os.path.join(color_dir, color_stems[stem])
        dp = os.path.join(depth_dir, depth_stems[stem])
        pairs.append((cp, dp))

    return pairs


def process_one_pig(pig_uid, info, yolo_model, fastsam_model, output_dir):
    """
    处理一头猪：生成裁剪后的深度图。
    返回: [(frame_filename, backfat), ...] 成功保存的帧
    """
    color_dir = info['color_dir']
    depth_dir = info['depth_dir']
    backfat = info['backfat']

    # 获取帧对
    pairs = get_frame_pairs(color_dir, depth_dir, FRAMES_PER_PIG)
    if not pairs:
        return []

    # 猪的输出目录
    pig_out_dir = os.path.join(output_dir, pig_uid)
    os.makedirs(pig_out_dir, exist_ok=True)

    saved = []
    for idx, (color_path, depth_path) in enumerate(pairs):
        # 读取 RGB
        color_img = imread_chinese(color_path)
        if color_img is None:
            continue

        # 获取 mask
        mask = get_pig_mask(color_img, yolo_model, fastsam_model)
        if mask is None:
            continue

        # 读取深度图
        depth_img = imread_depth_chinese(depth_path)
        if depth_img is None:
            continue

        # 确保 mask 尺寸匹配
        if mask.shape != depth_img.shape:
            mask = cv2.resize(mask, (depth_img.shape[1], depth_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # 裁剪深度图
        cropped = crop_masked_depthmap(depth_img, mask, TARGET_SIZE)
        if cropped is None:
            continue

        # 保存
        frame_name = f'frame_{idx:03d}.npy'
        save_path = os.path.join(pig_out_dir, frame_name)
        np.save(save_path, cropped)
        saved.append((os.path.join(pig_uid, frame_name), backfat))

    return saved


def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("深度图数据集生成 — CNN 背膘回归")
    print("=" * 60)

    # 1. 构建注册表
    print("\n[1/3] 构建猪数据注册表...")
    registry = build_pig_registry()
    print(f"  共找到 {len(registry)} 头猪（已过滤异常背膘值）")

    # 统计各数据源
    source_counts = {}
    for info in registry.values():
        src = info['source']
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt} 头")

    # 2. 加载模型
    print("\n[2/3] 加载 YOLO + FastSAM 模型...")
    yolo_model, fastsam_model = load_models()
    print("  模型加载完成")

    # 3. 处理每头猪
    print(f"\n[3/3] 开始处理 {len(registry)} 头猪（每头 {FRAMES_PER_PIG} 帧）...")

    all_records = []  # [(frame_path, pig_uid, backfat, source), ...]
    success_count = 0
    fail_count = 0

    pig_items = sorted(registry.items())
    for i, (pig_uid, info) in enumerate(pig_items, 1):
        if i % 50 == 0 or i <= 3:
            elapsed = time.time() - start_time
            eta = elapsed / i * (len(pig_items) - i) if i > 0 else 0
            print(f"  [{i}/{len(pig_items)}] {pig_uid} "
                  f"(elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)")

        saved = process_one_pig(pig_uid, info, yolo_model, fastsam_model, OUTPUT_DIR)
        if saved:
            success_count += 1
            for frame_path, bf in saved:
                all_records.append((frame_path, pig_uid, bf, info['source']))
        else:
            fail_count += 1

    # 4. 保存 metadata.csv
    csv_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_path', 'pig_id', 'backfat', 'source'])
        for rec in all_records:
            writer.writerow(rec)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"  成功处理: {success_count} 头猪")
    print(f"  失败跳过: {fail_count} 头猪")
    print(f"  总帧数:   {len(all_records)}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  元数据:   {csv_path}")
    print(f"  总耗时:   {total_time/60:.1f} 分钟")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
