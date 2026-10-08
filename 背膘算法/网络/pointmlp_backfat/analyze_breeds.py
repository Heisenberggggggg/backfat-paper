#!/usr/bin/env python3
"""
分析耳标号中的品种/类别信息
v1 - 2026-04-14
"""
import sys, os, re
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
import train_v4 as tv4
import numpy as np

def extract_breed_prefix(ear_tag):
    """从耳标中提取品种前缀"""
    tag = ear_tag.strip()
    # 匹配开头的字母部分 (如 LY, T, A 等)
    m = re.match(r'^([A-Za-z]+)', tag)
    if m:
        return m.group(1).upper()
    return None

def main():
    dk_base = r'G:\耿来硕数据\DK'
    fmh_base = r'G:\耿来硕数据\丰美禾数据'

    print("=" * 70)
    print("耳标品种分类分析")
    print("=" * 70)

    # 1. 加载注册表
    registry = tv4.build_dataset_registry_v2(dk_base, fmh_base)
    print(f"\n总猪数: {len(registry)}")

    # 2. 按数据源分组，提取耳标信息
    source_tags = defaultdict(list)  # source -> [(uid, ear_tag, backfat)]

    for uid, (data_dir, bf, nf) in registry.items():
        if uid.startswith('nky1_'):
            src = 'nkyfirst'
            ear_tag = uid.replace('nky1_', '')
        elif uid.startswith('nky2_'):
            src = 'nkysecond'
            ear_tag = uid.replace('nky2_', '')
        elif uid.startswith('nky3_'):
            src = 'nkythird'
            ear_tag = uid.replace('nky3_', '')
        elif uid.startswith('fmh_'):
            src = 'fengmeihe'
            # fmh_date_dirname -> extract dirname
            parts = uid.split('_', 2)
            ear_tag = parts[2] if len(parts) >= 3 else uid
        else:
            src = 'unknown'
            ear_tag = uid
        source_tags[src].append((uid, ear_tag, bf))

    # 3. 分析丰美禾的耳标模式
    print("\n" + "=" * 70)
    print("丰美禾数据耳标分析")
    print("=" * 70)

    fmh_tags = source_tags.get('fengmeihe', [])
    prefix_counter = Counter()
    prefix_details = defaultdict(list)  # prefix -> [(ear_tag, backfat)]
    no_prefix = []

    for uid, ear_tag, bf in fmh_tags:
        prefix = extract_breed_prefix(ear_tag)
        if prefix:
            prefix_counter[prefix] += 1
            prefix_details[prefix].append((ear_tag, bf))
        else:
            no_prefix.append((ear_tag, bf))

    print(f"\n  总计: {len(fmh_tags)} 头")
    print(f"\n  --- 按耳标前缀分类 ---")
    print(f"  {'前缀':<10} {'数量':>6} {'占比':>8} {'背膘范围':>14} {'均值±标准差':>16}")
    print(f"  {'-'*60}")

    for prefix, count in sorted(prefix_counter.items(), key=lambda x: -x[1]):
        bfs = [bf for _, bf in prefix_details[prefix]]
        bfs_arr = np.array(bfs)
        pct = count / len(fmh_tags) * 100
        print(f"  {prefix:<10} {count:>6} {pct:>7.1f}%  {bfs_arr.min():>5.1f}-{bfs_arr.max():<5.1f}mm  {bfs_arr.mean():.1f}±{bfs_arr.std():.1f}mm")

    if no_prefix:
        print(f"\n  无字母前缀: {len(no_prefix)} 头")
        for tag, bf in sorted(no_prefix)[:20]:
            print(f"    {tag}: {bf:.1f}mm")

    # 4. 详细展示每个前缀的耳标样例
    print(f"\n  --- 各前缀耳标样例 ---")
    for prefix, count in sorted(prefix_counter.items(), key=lambda x: -x[1]):
        samples = sorted(prefix_details[prefix], key=lambda x: x[0])[:8]
        sample_str = ", ".join(f"{t}" for t, _ in samples)
        if count > 8:
            sample_str += f" ... (共{count}个)"
        print(f"  {prefix}: {sample_str}")

    # 5. 分析DK数据的耳标模式
    print("\n" + "=" * 70)
    print("省农科院(DK)数据耳标分析")
    print("=" * 70)

    for src_name in ['nkyfirst', 'nkysecond', 'nkythird']:
        tags = source_tags.get(src_name, [])
        if not tags:
            continue
        print(f"\n  --- {src_name} ({len(tags)}头) ---")

        # DK的耳标格式不同，分析其模式
        patterns = Counter()
        for uid, ear_tag, bf in tags:
            # 检查是否有字母前缀
            prefix = extract_breed_prefix(ear_tag)
            if prefix:
                # 去除前缀后的部分
                patterns[prefix] += 1
            else:
                # 检查数字模式 (如 2-33, 6-25, 13-255)
                m = re.match(r'^(\d+)-', ear_tag)
                if m:
                    dong = m.group(1)
                    patterns[f"栋{dong}"] += 1
                else:
                    patterns["其他"] += 1

        for pat, cnt in sorted(patterns.items(), key=lambda x: -x[1]):
            samples = [(t, bf) for _, t, bf in tags if (
                (pat.startswith("栋") and t.startswith(pat.replace("栋", "") + "-")) or
                (not pat.startswith("栋") and t.upper().startswith(pat))
            )][:5]
            bfs = [bf for _, bf in samples]
            if bfs:
                sample_tags = ", ".join(f"{t}" for t, _ in samples)
                print(f"    {pat}: {cnt}头, 例: {sample_tags}")

        # 背膘统计
        bfs_all = np.array([bf for _, _, bf in tags])
        print(f"    总体背膘: {bfs_all.min():.1f}-{bfs_all.max():.1f}mm, 均值{bfs_all.mean():.1f}±{bfs_all.std():.1f}mm")

    # 6. 全局品种汇总
    print("\n" + "=" * 70)
    print("全局品种/耳标类型汇总")
    print("=" * 70)

    all_breeds = defaultdict(list)  # breed -> [(uid, backfat, source)]

    for src_name, tags in source_tags.items():
        for uid, ear_tag, bf in tags:
            prefix = extract_breed_prefix(ear_tag)
            if prefix and src_name == 'fengmeihe':
                all_breeds[prefix].append((uid, bf, src_name))
            elif src_name == 'nkyfirst':
                # nkyfirst: 2-X 和 3-X 系列
                m = re.match(r'^(\d+)-', ear_tag)
                if m:
                    all_breeds[f"DK-栋{m.group(1)}"].append((uid, bf, src_name))
            elif src_name == 'nkysecond':
                m = re.match(r'^(\d+)-', ear_tag)
                if m:
                    all_breeds[f"DK-栋{m.group(1)}"].append((uid, bf, src_name))
            elif src_name == 'nkythird':
                prefix_nky3 = extract_breed_prefix(ear_tag)
                if prefix_nky3:
                    all_breeds[f"DK3-{prefix_nky3}"].append((uid, bf, src_name))
                else:
                    m = re.match(r'^(\d+)-', ear_tag)
                    if m:
                        all_breeds[f"DK-栋{m.group(1)}"].append((uid, bf, src_name))

    print(f"\n  {'类别':<15} {'数量':>6} {'背膘范围':>14} {'均值±标准差':>16} {'数据源':>12}")
    print(f"  {'-'*70}")

    for breed, items in sorted(all_breeds.items(), key=lambda x: -len(x[1])):
        bfs = np.array([bf for _, bf, _ in items])
        srcs = set(s for _, _, s in items)
        src_str = ",".join(sorted(srcs))
        if len(src_str) > 12:
            src_str = src_str[:12]
        print(f"  {breed:<15} {len(items):>6}  {bfs.min():>5.1f}-{bfs.max():<5.1f}mm  {bfs.mean():.1f}±{bfs.std():.1f}mm  {src_str}")

    # 7. 检查是否有跨日期重复的耳标（同一头猪多次测量）
    print("\n" + "=" * 70)
    print("丰美禾跨日期重复耳标检测（同一头猪多次采集）")
    print("=" * 70)

    fmh_by_tag = defaultdict(list)  # ear_tag -> [(date, backfat)]
    for uid, ear_tag, bf in fmh_tags:
        parts = uid.split('_', 2)
        date = parts[1] if len(parts) >= 3 else '?'
        fmh_by_tag[ear_tag].append((date, bf))

    repeat_tags = {t: dates for t, dates in fmh_by_tag.items() if len(dates) > 1}
    print(f"\n  唯一耳标数: {len(fmh_by_tag)}")
    print(f"  出现多次的耳标: {len(repeat_tags)}")

    if repeat_tags:
        print(f"\n  --- 多次出现的耳标 (前20个) ---")
        for tag, dates in sorted(repeat_tags.items(), key=lambda x: -len(x[1]))[:20]:
            dates_str = ", ".join(f"{d}({bf:.0f}mm)" for d, bf in sorted(dates))
            bf_vals = [bf for _, bf in dates]
            bf_range = max(bf_vals) - min(bf_vals)
            flag = " *** 背膘变化大!" if bf_range > 5 else ""
            print(f"    {tag}: {len(dates)}次 -> {dates_str}{flag}")

    print("\n" + "=" * 70)
    print("分析完成")


if __name__ == '__main__':
    main()
