#!/usr/bin/env python3
"""
分析背膘数据分布 + 体况-背膘一致性检查
v1 - 2026-04-08
目的：诊断数据不均匀和标签噪声问题
"""
import sys, os, re, importlib

# 复用 train_v4.py 中的数据加载函数
sys.path.insert(0, os.path.dirname(__file__))
import train_v4 as tv4

import numpy as np

def main():
    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    print("=" * 70)
    print("背膘数据分布分析")
    print("=" * 70)

    # 1. 加载所有注册表
    registry = tv4.build_dataset_registry_v2(dk_base, fmh_base)
    print(f"\n总猪数: {len(registry)}")

    # 提取背膘值
    backfats = []
    pig_info = []  # (uid, backfat, n_frames, source)
    for uid, (data_dir, bf, nf) in registry.items():
        backfats.append(bf)
        # 确定来源
        if uid.startswith('nky1_'):
            src = 'nkyfirst'
        elif uid.startswith('nky2_'):
            src = 'nkysecond'
        elif uid.startswith('nky3_'):
            src = 'nkythird'
        elif uid.startswith('fmh_'):
            src = 'fengmeihe'
        else:
            src = 'unknown'
        pig_info.append((uid, bf, nf, src))

    backfats = np.array(backfats)

    # 2. 基本统计
    print(f"\n--- 基本统计 ---")
    print(f"  Min:    {backfats.min():.1f} mm")
    print(f"  Max:    {backfats.max():.1f} mm")
    print(f"  Mean:   {backfats.mean():.2f} mm")
    print(f"  Median: {np.median(backfats):.2f} mm")
    print(f"  Std:    {backfats.std():.2f} mm")
    print(f"  Q1:     {np.percentile(backfats, 25):.1f} mm")
    print(f"  Q3:     {np.percentile(backfats, 75):.1f} mm")

    # 偏度和峰度
    from scipy import stats as sp_stats
    skew = sp_stats.skew(backfats)
    kurt = sp_stats.kurtosis(backfats)
    print(f"  Skewness: {skew:.3f}  (>0=右偏, 0=对称)")
    print(f"  Kurtosis: {kurt:.3f}  (>0=尖峰, 0=正态)")

    # 3. 直方图 (每1mm一个bin)
    print(f"\n--- 直方图 (1mm bins) ---")
    bin_min = int(np.floor(backfats.min()))
    bin_max = int(np.ceil(backfats.max())) + 1
    bins = np.arange(bin_min, bin_max + 1, 1)
    counts, edges = np.histogram(backfats, bins=bins)

    max_count = counts.max()
    bar_width = 40  # 最大柱宽

    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar = '█' * int(c / max_count * bar_width) if max_count > 0 else ''
        pct = c / len(backfats) * 100
        print(f"  [{lo:5.0f}-{hi:5.0f}) mm: {c:4d} ({pct:5.1f}%) {bar}")

    # 4. 按来源分布
    print(f"\n--- 按数据来源分布 ---")
    sources = {}
    for uid, bf, nf, src in pig_info:
        if src not in sources:
            sources[src] = []
        sources[src].append(bf)

    for src in sorted(sources.keys()):
        vals = np.array(sources[src])
        print(f"\n  {src}: {len(vals)} pigs")
        print(f"    Range: {vals.min():.1f} - {vals.max():.1f} mm")
        print(f"    Mean:  {vals.mean():.1f} ± {vals.std():.1f} mm")

    # 5. 累积分布：看哪些区间数据稀疏
    print(f"\n--- 数据稀疏区间分析 ---")
    # 每2mm一个区间
    bins_2mm = np.arange(bin_min, bin_max + 1, 2)
    counts_2mm, edges_2mm = np.histogram(backfats, bins=bins_2mm)
    mean_count = counts_2mm.mean()

    print(f"  2mm区间平均样本数: {mean_count:.1f}")
    print(f"  稀疏区间 (< 平均的50%):")
    for i, c in enumerate(counts_2mm):
        if c < mean_count * 0.5:
            lo = edges_2mm[i]
            hi = edges_2mm[i + 1]
            print(f"    [{lo:.0f}-{hi:.0f}) mm: {c} 头猪 (仅 {c/len(backfats)*100:.1f}%)")

    print(f"\n  密集区间 (> 平均的150%):")
    for i, c in enumerate(counts_2mm):
        if c > mean_count * 1.5:
            lo = edges_2mm[i]
            hi = edges_2mm[i + 1]
            print(f"    [{lo:.0f}-{hi:.0f}) mm: {c} 头猪 ({c/len(backfats)*100:.1f}%)")

    # 6. 不均匀性量化
    # Gini系数 / 变异系数
    counts_nonzero = counts[counts > 0]
    cv = counts_nonzero.std() / counts_nonzero.mean()
    print(f"\n--- 不均匀性指标 ---")
    print(f"  1mm bin counts 变异系数 (CV): {cv:.3f}  (>0.5=严重不均匀)")

    # 最多 vs 最少的倍数关系
    if counts_nonzero.min() > 0:
        ratio = counts_nonzero.max() / counts_nonzero.min()
        print(f"  最多/最少 bin 倍数: {ratio:.1f}x")

    # 7. 每头猪的帧数分布
    frames = [nf for _, _, nf, _ in pig_info]
    frames = np.array(frames)
    print(f"\n--- 每头猪帧数分布 ---")
    print(f"  Min:    {frames.min()}")
    print(f"  Max:    {frames.max()}")
    print(f"  Mean:   {frames.mean():.1f}")
    print(f"  Median: {np.median(frames):.0f}")
    print(f"  Total frames: {frames.sum()}")

    # 8. 按背膘区间统计帧数（帧级别的不均匀）
    print(f"\n--- 帧级别不均匀性 ---")
    bins_3mm = np.arange(bin_min, bin_max + 1, 3)
    for i in range(len(bins_3mm) - 1):
        lo, hi = bins_3mm[i], bins_3mm[i + 1]
        mask = (backfats >= lo) & (backfats < hi)
        n_pigs = mask.sum()
        n_frames_bin = sum(nf for uid, bf, nf, src in pig_info if lo <= bf < hi)
        if n_pigs > 0:
            print(f"  [{lo:5.0f}-{hi:5.0f}) mm: {n_pigs:4d} pigs, {n_frames_bin:6d} frames "
                  f"(pig {n_pigs/len(backfats)*100:.1f}%, frame {n_frames_bin/frames.sum()*100:.1f}%)")

    # 9. 体况-背膘匹配分析（如果有体况数据）
    print(f"\n--- 体况-背膘匹配分析 ---")
    print("  尝试从Excel加载体况评分...")
    try:
        analyze_bcs_backfat_consistency(dk_base, fmh_base, pig_info)
    except Exception as e:
        print(f"  体况数据加载失败: {e}")
        print("  需要手动检查Word/Excel中是否有体况评分列")

    # 10. 潜在异常值检测
    print(f"\n--- 潜在异常值 (IQR法) ---")
    q1 = np.percentile(backfats, 25)
    q3 = np.percentile(backfats, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print(f"  IQR: {iqr:.1f}, 下界: {lower:.1f}, 上界: {upper:.1f}")

    outliers = [(uid, bf, src) for uid, bf, nf, src in pig_info if bf < lower or bf > upper]
    print(f"  异常值数量: {len(outliers)}")
    for uid, bf, src in sorted(outliers, key=lambda x: x[1]):
        print(f"    {uid}: {bf:.1f}mm ({src})")

    print("\n" + "=" * 70)
    print("分析完成")


def analyze_bcs_backfat_consistency(dk_base, fmh_base, pig_info):
    """尝试从原始文件中读取体况评分，与背膘做交叉检查"""
    import openpyxl

    # 检查丰美禾Excel是否有体况列
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    if not os.path.isfile(xlsx_path):
        print("  找不到背膘记录表.xlsx")
        return

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)

    print(f"\n  Sheet names: {wb.sheetnames}")
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"\n  Sheet '{sheet_name}': {ws.max_row} rows x {ws.max_column} cols")
        # 打印前3行看列头
        for row_idx in range(1, min(4, ws.max_row + 1)):
            cells = []
            for col_idx in range(1, min(ws.max_column + 1, 15)):
                val = ws.cell(row=row_idx, column=col_idx).value
                if val is not None:
                    cells.append(f"C{col_idx}={val}")
            if cells:
                print(f"    Row {row_idx}: {', '.join(cells)}")

    wb.close()

    # 检查 Word docs 的表格列数
    print(f"\n  检查Word文档中的列...")
    try:
        from docx import Document
        for docx_name in ['第一次数据7.29.docx', '第二次数据9.6.docx', '第三次数据12.9.docx']:
            docx_path = os.path.join(dk_base, docx_name)
            if os.path.isfile(docx_path):
                doc = Document(docx_path)
                for i, table in enumerate(doc.tables):
                    # 打印表头
                    if len(table.rows) > 0:
                        header = [cell.text.strip() for cell in table.rows[0].cells]
                        print(f"    {docx_name} Table {i}: {len(table.columns)} cols, {len(table.rows)} rows")
                        print(f"      Header: {header}")
                        # 打印第一行数据
                        if len(table.rows) > 1:
                            data1 = [cell.text.strip() for cell in table.rows[1].cells]
                            print(f"      Row 1:  {data1}")
    except ImportError:
        print("  python-docx 未安装，跳过Word文件检查")


if __name__ == '__main__':
    main()
