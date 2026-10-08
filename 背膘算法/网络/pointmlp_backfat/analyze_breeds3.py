# -*- coding: utf-8 -*-
"""分析耳标品种分类 - 直接扫描目录"""
import os, re, sys
import numpy as np
from collections import defaultdict, Counter

# 直接硬编码路径（避免bash编码问题）
DK_BASE = 'G:\\耿来硕数据\\DK'
FMH_BASE = 'G:\\耿来硕数据\\丰美禾数据'
FMH_DATES = ['8.24','8.25','8.26','8.27','10.29','10.30','10.31',
             '11.1','11.2','11.3','11.4','11.5']

def extract_prefix(dirname):
    """从目录名提取品种字母前缀，如 55LY0923 -> LY, 687T204 -> T"""
    m = re.match(r'^\d+([A-Za-z]+)', dirname)
    if m:
        return m.group(1).upper()
    m = re.match(r'^([A-Za-z]+)', dirname)
    if m:
        return m.group(1).upper()
    return None

def main():
    # debug: check paths
    import sys
    print(f"Python: {sys.executable}")
    print(f"DK_BASE exists: {os.path.isdir(DK_BASE)} -> {DK_BASE}")
    print(f"FMH_BASE exists: {os.path.isdir(FMH_BASE)} -> {FMH_BASE}")
    if os.path.isdir(FMH_BASE):
        print(f"FMH contents: {os.listdir(FMH_BASE)[:5]}")
    # also try with forward slashes
    alt = 'G:/耿来硕数据/丰美禾数据'
    print(f"Alt path exists: {os.path.isdir(alt)}")

    print("=" * 70)
    print("耳标品种分类分析")
    print("=" * 70)

    # ==============================
    # 丰美禾数据
    # ==============================
    print("\n【丰美禾数据】")

    all_fmh = []  # (date, dirname, prefix)
    for date_str in FMH_DATES:
        date_dir = os.path.join(FMH_BASE, date_str)
        if not os.path.isdir(date_dir):
            print(f"  警告: {date_str} 目录不存在")
            continue
        for pig_dir in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dir)
            if not os.path.isdir(pig_path):
                continue
            # 检查有没有master目录（有数据的猪）
            master_path = os.path.join(pig_path, 'master')
            if not os.path.isdir(master_path):
                continue
            prefix = extract_prefix(pig_dir)
            all_fmh.append((date_str, pig_dir, prefix or '无前缀'))

    print(f"  总猪目录数: {len(all_fmh)}")

    # 按前缀分组
    by_prefix = defaultdict(list)
    for date_str, dirname, prefix in all_fmh:
        by_prefix[prefix].append((date_str, dirname))

    print(f"\n  {'品种前缀':<10} {'目录数':>8} {'占比':>8}  耳标样例")
    print(f"  {'-'*70}")
    for prefix, items in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        pct = len(items) / len(all_fmh) * 100
        # 去重的耳标样例
        unique_dirs = sorted(set(dn for _, dn in items))[:5]
        extra = f" ...(共{len(set(dn for _,dn in items))}个唯一耳标)" if len(set(dn for _,dn in items)) > 5 else ""
        print(f"  {prefix:<10} {len(items):>8} {pct:>7.1f}%  {', '.join(unique_dirs)}{extra}")

    # 跨日期重复统计
    by_dirname = defaultdict(list)
    for date_str, dirname, prefix in all_fmh:
        by_dirname[dirname].append(date_str)
    n_unique = len(by_dirname)
    n_repeat = sum(1 for dates in by_dirname.values() if len(dates) > 1)
    print(f"\n  唯一耳标数: {n_unique}")
    print(f"  出现多次（多天采集）的耳标: {n_repeat}")

    # 按品种统计跨日期重复
    if n_repeat > 0:
        repeat_by_prefix = defaultdict(int)
        for dirname, dates in by_dirname.items():
            if len(dates) > 1:
                p = extract_prefix(dirname) or '无前缀'
                repeat_by_prefix[p] += 1
        print(f"\n  重复耳标按品种:")
        for p, cnt in sorted(repeat_by_prefix.items(), key=lambda x: -x[1]):
            print(f"    {p}: {cnt}头有多次采集")

    # 每个品种的每日分布
    print(f"\n  --- 各品种在各日期的分布 ---")
    print(f"  {'品种':<6}", end='')
    for d in FMH_DATES:
        print(f" {d:>5}", end='')
    print(f"  {'合计':>6}")
    print(f"  {'-'*90}")

    for prefix in sorted(by_prefix.keys(), key=lambda x: -len(by_prefix[x])):
        items = by_prefix[prefix]
        date_counts = Counter(date for date, _ in items)
        print(f"  {prefix:<6}", end='')
        total = 0
        for d in FMH_DATES:
            c = date_counts.get(d, 0)
            total += c
            print(f" {c:>5}", end='')
        print(f"  {total:>6}")

    # ==============================
    # 省农科院数据
    # ==============================
    print(f"\n\n【省农科院(DK)数据】")

    # nkyfirst
    nky1 = os.path.join(DK_BASE, 'nkyfirst')
    if os.path.isdir(nky1):
        pigs = sorted([d for d in os.listdir(nky1) if os.path.isdir(os.path.join(nky1, d))])
        dong2 = [p for p in pigs if p.startswith('2-') or p.startswith('c2-')]
        dong3 = [p for p in pigs if p.startswith('3-') or p.startswith('c3-')]
        print(f"\n  nkyfirst: {len(pigs)}头")
        print(f"    栋2: {len(dong2)}头 -> {', '.join(dong2[:6])} ...")
        print(f"    栋3: {len(dong3)}头 -> {', '.join(dong3[:6])} ...")

    # nkysecond
    nky2 = os.path.join(DK_BASE, 'nkysecond')
    if os.path.isdir(nky2):
        pigs = sorted([d for d in os.listdir(nky2) if os.path.isdir(os.path.join(nky2, d))])
        dong6 = [p for p in pigs if p.startswith('6-')]
        dong7 = [p for p in pigs if p.startswith('7-')]
        print(f"\n  nkysecond: {len(pigs)}头")
        print(f"    栋6: {len(dong6)}头 -> {', '.join(dong6[:6])} ...")
        print(f"    栋7: {len(dong7)}头 -> {', '.join(dong7[:6])} ...")

    # nkythird
    nky3 = os.path.join(DK_BASE, 'nkythird')
    if os.path.isdir(nky3):
        all_3 = []
        for kinect in sorted(os.listdir(nky3)):
            kp = os.path.join(nky3, kinect)
            if not os.path.isdir(kp):
                continue
            for pig in os.listdir(kp):
                if os.path.isdir(os.path.join(kp, pig)):
                    all_3.append((kinect, pig))

        # 分析耳标后缀：数字部分（如 255, 277, 143）
        # 实际上nkythird的耳标像 32360L, 32473, 233 等（从Word里看到的）
        print(f"\n  nkythird: {len(all_3)}个目录")
        # 按前缀
        pre_groups = defaultdict(list)
        for kinect, pig in all_3:
            if pig.startswith('wc'):
                pre_groups['wc(无/差)'].append(pig)
            elif pig.startswith('w'):
                pre_groups['w(无)'].append(pig)
            elif pig.startswith('c'):
                pre_groups['c(差异)'].append(pig)
            elif pig.startswith('k'):
                pre_groups['k'].append(pig)
            else:
                pre_groups['正常'].append(pig)
        for g, ps in sorted(pre_groups.items(), key=lambda x: -len(x[1])):
            print(f"    {g}: {len(ps)}头, 例: {', '.join(sorted(ps)[:4])}")

        # nkythird的耳标后缀分析
        # 格式: {prefix}{dong}-{ear_id}
        # ear_id 是数字标识，不含品种信息
        # 品种信息只在Word文档的"猪只品种"列中
        print(f"\n    注: nkythird耳标为纯数字ID，品种信息需查Word文档'猪只品种'列")

    # ==============================
    # 总结
    # ==============================
    print(f"\n\n{'='*70}")
    print("总结: 可用于分组训练的品种类别")
    print(f"{'='*70}")

    print(f"""
  丰美禾数据（主要数据源，755头有标签）:
""")
    for prefix, items in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        if prefix == '无前缀':
            continue
        n = len(set(dn for _, dn in items))
        print(f"    {prefix}: ~{n}头唯一猪")

    print(f"""
  省农科院数据（163头有标签）:
    耳标为 栋号-编号 格式，不含品种前缀
    品种信息在Word文档中，需要额外提取

  建议:
    - 丰美禾数据可直接按耳标前缀(LY/T/A等)分组训练
    - 省农科院数据需要从Word文档中读取'猪只品种'列才能分类
    - 重复采集的猪可以用于验证模型的时间稳定性
""")

if __name__ == '__main__':
    main()
