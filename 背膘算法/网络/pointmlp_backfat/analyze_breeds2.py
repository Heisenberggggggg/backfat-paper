#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析耳标号中的品种/类别信息
直接扫描目录名提取耳标，避免中文路径docx加载问题
"""
import os, re, sys
import numpy as np
from collections import defaultdict, Counter

def extract_breed_prefix(tag):
    """从耳标/目录名中提取字母前缀"""
    # 去掉数字前缀（如"55LY0923"中提取"LY"）
    # 丰美禾目录名格式: 数字+字母+数字, 如 55LY0923, 618LY0923, 687T204
    m = re.match(r'^\d+([A-Za-z]+)', tag)
    if m:
        return m.group(1).upper()
    # 纯字母开头
    m = re.match(r'^([A-Za-z]+)', tag)
    if m:
        return m.group(1).upper()
    return None

def load_fmh_labels(fmh_base):
    """加载丰美禾标签"""
    import openpyxl
    entries = []

    # 主标签文件
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    if os.path.isfile(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb['Sheet1']
        current_date = None
        for row in ws.iter_rows(min_row=1, values_only=True):
            vals = list(row)
            # 检测日期行
            if vals[0] and isinstance(vals[0], str):
                for date_str in ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                                 '11.1', '11.2', '11.3', '11.4', '11.5']:
                    if date_str in str(vals[0]) or (vals[5] and date_str in str(vals[5])):
                        current_date = date_str
                        break
            # 数据行: 序号(int), 耳标(str), 背膘值(float)
            if vals[0] and vals[1] and vals[2]:
                try:
                    seq = int(vals[0]) if isinstance(vals[0], (int, float)) else None
                    tag = str(vals[1]).strip()
                    bf = float(vals[2])
                    if seq and tag and 3 < bf < 250:
                        entries.append((tag, bf, current_date or ''))
                except (ValueError, TypeError):
                    pass
        wb.close()

    # 11月补充标签
    xlsx_nov = os.path.join(fmh_base, '11.10处理背膘数据.xlsx')
    if os.path.isfile(xlsx_nov):
        wb = openpyxl.load_workbook(xlsx_nov, data_only=True)
        ws = wb.active
        current_date = None
        for row in ws.iter_rows(min_row=1, values_only=True):
            vals = list(row)
            if vals[0] and isinstance(vals[0], str):
                for date_str in ['11.1', '11.2', '11.3', '11.4', '11.5']:
                    if date_str in str(vals[0]):
                        current_date = date_str
                        break
            if vals[0] and vals[1] and vals[2]:
                try:
                    seq = int(vals[0]) if isinstance(vals[0], (int, float)) else None
                    tag = str(vals[1]).strip()
                    bf = float(vals[2])
                    if seq and tag and 3 < bf < 250:
                        entries.append((tag, bf, current_date or ''))
                except (ValueError, TypeError):
                    pass
        wb.close()

    return entries

def main():
    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    print("=" * 70)
    print("耳标品种分类分析")
    print("=" * 70)

    # =========================================
    # 丰美禾: 扫描目录名 + 匹配标签
    # =========================================
    fmh_dates = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                 '11.1', '11.2', '11.3', '11.4', '11.5']

    # 收集所有目录名
    fmh_pigs = []  # (date, dirname, backfat_or_None)
    for date_str in fmh_dates:
        date_dir = os.path.join(fmh_base, date_str)
        if not os.path.isdir(date_dir):
            continue
        for pig_dir in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dir)
            if not os.path.isdir(pig_path):
                continue
            # 检查是否有点云数据
            pp = os.path.join(pig_path, 'master', 'processed_pointclouds_v2')
            if os.path.isdir(pp) and any(f.endswith('.txt') for f in os.listdir(pp)):
                fmh_pigs.append((date_str, pig_dir, None))

    # 加载标签
    fmh_labels = load_fmh_labels(fmh_base)
    label_map = {}  # tag -> backfat (last occurrence wins)
    for tag, bf, date in fmh_labels:
        label_map[tag] = bf

    # 匹配标签到目录
    matched_pigs = []
    for date_str, dirname, _ in fmh_pigs:
        bf = None
        for tag, bf_val in label_map.items():
            if tag.strip() in dirname:
                bf = bf_val
                break
        matched_pigs.append((date_str, dirname, bf))

    print(f"\n{'='*70}")
    print("丰美禾数据耳标分析")
    print(f"{'='*70}")
    print(f"\n  总目录数: {len(matched_pigs)}, 有标签: {sum(1 for _,_,bf in matched_pigs if bf)}")

    # 按品种前缀分类
    prefix_groups = defaultdict(list)  # prefix -> [(date, dirname, bf)]
    no_prefix = []

    for date_str, dirname, bf in matched_pigs:
        prefix = extract_breed_prefix(dirname)
        if prefix:
            prefix_groups[prefix].append((date_str, dirname, bf))
        else:
            no_prefix.append((date_str, dirname, bf))

    print(f"\n  {'品种前缀':<10} {'猪只数':>6} {'有标签':>6} {'占比':>8} {'背膘范围':>14} {'均值±标准差':>16}")
    print(f"  {'-'*66}")

    for prefix, items in sorted(prefix_groups.items(), key=lambda x: -len(x[1])):
        bfs = [bf for _, _, bf in items if bf is not None and 5 < bf < 40]
        n_labeled = len(bfs)
        pct = len(items) / len(matched_pigs) * 100
        if bfs:
            bfs_arr = np.array(bfs)
            print(f"  {prefix:<10} {len(items):>6} {n_labeled:>6} {pct:>7.1f}%  "
                  f"{bfs_arr.min():>5.1f}-{bfs_arr.max():<5.1f}mm  {bfs_arr.mean():.1f}±{bfs_arr.std():.1f}mm")
        else:
            print(f"  {prefix:<10} {len(items):>6} {n_labeled:>6} {pct:>7.1f}%  {'N/A':>14} {'N/A':>16}")

    if no_prefix:
        print(f"\n  无字母前缀: {len(no_prefix)} 头")
        for date, dn, bf in sorted(no_prefix)[:10]:
            bf_str = f"{bf:.1f}mm" if bf else "无标签"
            print(f"    {date}/{dn}: {bf_str}")

    # 耳标样例
    print(f"\n  --- 各品种耳标样例 ---")
    for prefix, items in sorted(prefix_groups.items(), key=lambda x: -len(x[1])):
        samples = sorted(set(dn for _, dn, _ in items))[:6]
        extra = f" ... (共{len(items)}个)" if len(items) > 6 else ""
        print(f"  {prefix}: {', '.join(samples)}{extra}")

    # =========================================
    # 省农科院数据
    # =========================================
    print(f"\n{'='*70}")
    print("省农科院(DK)数据耳标分析")
    print(f"{'='*70}")

    # nkyfirst: 直接目录名就是耳标 (2-1, 2-2, 3-1, ...)
    nky1_dir = os.path.join(dk_base, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        pigs = [d for d in os.listdir(nky1_dir) if os.path.isdir(os.path.join(nky1_dir, d))]
        dong_groups = defaultdict(list)
        for p in pigs:
            m = re.match(r'^[c]*(\d+)-', p)
            if m:
                dong_groups[m.group(1)].append(p)
        print(f"\n  --- nkyfirst ({len(pigs)}头) ---")
        for dong, ps in sorted(dong_groups.items()):
            print(f"    栋{dong}: {len(ps)}头, 例: {', '.join(sorted(ps)[:6])}")

    # nkysecond
    nky2_dir = os.path.join(dk_base, 'nkysecond')
    if os.path.isdir(nky2_dir):
        pigs = [d for d in os.listdir(nky2_dir) if os.path.isdir(os.path.join(nky2_dir, d))]
        dong_groups = defaultdict(list)
        for p in pigs:
            m = re.match(r'^(\d+)-', p)
            if m:
                dong_groups[m.group(1)].append(p)
        print(f"\n  --- nkysecond ({len(pigs)}头) ---")
        for dong, ps in sorted(dong_groups.items()):
            print(f"    栋{dong}: {len(ps)}头, 例: {', '.join(sorted(ps)[:6])}")

    # nkythird
    nky3_dir = os.path.join(dk_base, 'nkythird')
    if os.path.isdir(nky3_dir):
        print(f"\n  --- nkythird ---")
        all_pigs_3 = []
        for kinect_dir in sorted(os.listdir(nky3_dir)):
            kp = os.path.join(nky3_dir, kinect_dir)
            if not os.path.isdir(kp):
                continue
            pigs = [d for d in os.listdir(kp) if os.path.isdir(os.path.join(kp, d))]
            all_pigs_3.extend(pigs)

        # 分析前缀模式
        prefix_3 = defaultdict(list)
        for p in all_pigs_3:
            if p.startswith('w') or p.startswith('wc'):
                prefix_3['w/wc(无标签)'].append(p)
            elif p.startswith('c'):
                prefix_3['c(差异)'].append(p)
            elif p.startswith('k'):
                prefix_3['k(待确认)'].append(p)
            else:
                # 纯数字-开头
                m = re.match(r'^(\d+)-', p)
                if m:
                    prefix_3[f'栋{m.group(1)}'].append(p)
                else:
                    prefix_3['其他'].append(p)

        print(f"    总目录数: {len(all_pigs_3)}")
        for pre, ps in sorted(prefix_3.items(), key=lambda x: -len(x[1])):
            # 提取栋号
            dong_set = set()
            for p in ps:
                m = re.match(r'^[cwk]*(\d+)-', p)
                if m:
                    dong_set.add(m.group(1))
            dong_str = f" (栋号: {','.join(sorted(dong_set))})" if dong_set else ""
            print(f"    {pre}: {len(ps)}头{dong_str}, 例: {', '.join(sorted(ps)[:4])}")

        # nkythird中耳标是否有品种信息？
        print(f"\n    耳标样例（去前缀后）:")
        numeric_tags = []
        for p in all_pigs_3:
            m = re.match(r'^[cwk]*\d+-(.+)$', p)
            if m:
                numeric_tags.append(m.group(1))
        tag_types = Counter()
        for t in numeric_tags:
            if re.match(r'^\d+$', t):
                tag_types['纯数字'] += 1
            elif re.match(r'^\d+[A-Za-z]$', t):
                tag_types['数字+字母后缀'] += 1
            else:
                tag_types[f'其他({t[:8]})'] += 1
        for tt, cnt in tag_types.most_common():
            print(f"      {tt}: {cnt}个")

    # =========================================
    # 全局汇总
    # =========================================
    print(f"\n{'='*70}")
    print("全局品种汇总（可用于分组训练）")
    print(f"{'='*70}")

    print(f"\n  丰美禾数据按品种分组:")
    for prefix, items in sorted(prefix_groups.items(), key=lambda x: -len(x[1])):
        bfs = [bf for _, _, bf in items if bf is not None and 5 < bf < 40]
        if len(bfs) >= 5:
            bfs_arr = np.array(bfs)
            print(f"    {prefix}: {len(items)}头 (有效标签{len(bfs)}), "
                  f"背膘{bfs_arr.mean():.1f}±{bfs_arr.std():.1f}mm")

    print(f"\n  省农科院数据:")
    print(f"    nkyfirst: 39头 (栋2+栋3), 背膘10-22mm")
    print(f"    nkysecond: 32头 (栋6+栋7), 背膘9.7-22mm")
    print(f"    nkythird: ~92头 (栋10-13), 背膘7.1-22mm")

    # 跨日期重复耳标检测
    print(f"\n{'='*70}")
    print("丰美禾跨日期重复耳标检测（同一头猪多次采集）")
    print(f"{'='*70}")

    by_dirname = defaultdict(list)
    for date_str, dirname, bf in matched_pigs:
        by_dirname[dirname].append((date_str, bf))

    repeats = {dn: dates for dn, dates in by_dirname.items() if len(dates) > 1}
    print(f"\n  唯一目录名: {len(by_dirname)}")
    print(f"  出现多次的目录名: {len(repeats)}")

    if repeats:
        # 按品种分组统计重复
        repeat_by_prefix = defaultdict(int)
        for dn in repeats:
            p = extract_breed_prefix(dn) or '其他'
            repeat_by_prefix[p] += 1
        print(f"\n  重复耳标按品种分布:")
        for p, cnt in sorted(repeat_by_prefix.items(), key=lambda x: -x[1]):
            print(f"    {p}: {cnt}头有重复采集")

        print(f"\n  重复耳标详情 (前15个):")
        for dn, dates in sorted(repeats.items(), key=lambda x: -len(x[1]))[:15]:
            dates_str = ", ".join(f"{d}({bf:.0f}mm)" if bf else f"{d}(无标签)"
                                 for d, bf in sorted(dates))
            bfs_valid = [bf for _, bf in dates if bf and 5 < bf < 40]
            if len(bfs_valid) >= 2:
                bf_range = max(bfs_valid) - min(bfs_valid)
                flag = f" *** 变化{bf_range:.0f}mm!" if bf_range > 3 else ""
            else:
                flag = ""
            print(f"    {dn}: {len(dates)}次 -> {dates_str}{flag}")

    print(f"\n{'='*70}")
    print("分析完成")

if __name__ == '__main__':
    main()
