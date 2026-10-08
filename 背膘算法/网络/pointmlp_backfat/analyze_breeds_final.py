# -*- coding: utf-8 -*-
import os, re, sys
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import paths from train_v4 (which has correct encoding)
import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--dk_base', default=os.path.join('G:', os.sep, '\u8015\u6765\u7855\u6570\u636e', 'DK'))
_p.add_argument('--fmh_base', default=os.path.join('G:', os.sep, '\u8015\u6765\u7855\u6570\u636e', '\u4e30\u7f8e\u79be\u6570\u636e'))
_args = _p.parse_args([])
DK_BASE = _args.dk_base
FMH_BASE = _args.fmh_base

FMH_DATES = ['8.24','8.25','8.26','8.27','10.29','10.30','10.31',
             '11.1','11.2','11.3','11.4','11.5']

def extract_prefix(dirname):
    m = re.match(r'^\d+([A-Za-z]+)', dirname)
    if m: return m.group(1).upper()
    m = re.match(r'^([A-Za-z]+)', dirname)
    if m: return m.group(1).upper()
    return None

def main():
    print(f"DK_BASE: {DK_BASE}, exists={os.path.isdir(DK_BASE)}")
    print(f"FMH_BASE: {FMH_BASE}, exists={os.path.isdir(FMH_BASE)}")

    print("=" * 70)
    print("[1] FMH ear tag analysis")
    print("=" * 70)

    all_fmh = []
    for date_str in FMH_DATES:
        date_dir = os.path.join(FMH_BASE, date_str)
        if not os.path.isdir(date_dir):
            print(f"  WARN: {date_str} not found")
            continue
        count = 0
        for pig_dir in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dir)
            if not os.path.isdir(pig_path):
                continue
            master = os.path.join(pig_path, 'master')
            if not os.path.isdir(master):
                continue
            prefix = extract_prefix(pig_dir)
            all_fmh.append((date_str, pig_dir, prefix or 'NO_PREFIX'))
            count += 1
        print(f"  {date_str}: {count} pigs")

    print(f"\n  Total FMH pig dirs: {len(all_fmh)}")

    by_prefix = defaultdict(list)
    for d, dn, p in all_fmh:
        by_prefix[p].append((d, dn))

    print(f"\n  {'Prefix':<12} {'Count':>6} {'Pct':>7}  Examples")
    print(f"  {'-'*70}")
    for prefix, items in sorted(by_prefix.items(), key=lambda x: -len(x[1])):
        pct = len(items) / len(all_fmh) * 100 if all_fmh else 0
        unique = sorted(set(dn for _, dn in items))
        examples = ', '.join(unique[:5])
        extra = f" ... (total {len(unique)} unique)" if len(unique) > 5 else ""
        print(f"  {prefix:<12} {len(items):>6} {pct:>6.1f}%  {examples}{extra}")

    # Cross-date repeats
    by_dir = defaultdict(list)
    for d, dn, p in all_fmh:
        by_dir[dn].append(d)
    n_unique = len(by_dir)
    n_repeat = sum(1 for v in by_dir.values() if len(v) > 1)
    print(f"\n  Unique ear tags: {n_unique}")
    print(f"  Multi-day ear tags: {n_repeat}")

    if n_repeat:
        repeat_by_p = defaultdict(int)
        for dn, dates in by_dir.items():
            if len(dates) > 1:
                p = extract_prefix(dn) or 'NO_PREFIX'
                repeat_by_p[p] += 1
        print(f"\n  Multi-day by breed:")
        for p, c in sorted(repeat_by_p.items(), key=lambda x: -x[1]):
            print(f"    {p}: {c}")

    # Date x Breed matrix
    print(f"\n  --- Breed x Date distribution ---")
    header = f"  {'Breed':<8}"
    for d in FMH_DATES:
        header += f" {d:>5}"
    header += f"  {'Total':>6}"
    print(header)
    print(f"  {'-'*90}")
    for prefix in sorted(by_prefix.keys(), key=lambda x: -len(by_prefix[x])):
        items = by_prefix[prefix]
        dc = Counter(d for d, _ in items)
        row = f"  {prefix:<8}"
        t = 0
        for d in FMH_DATES:
            c = dc.get(d, 0)
            t += c
            row += f" {c:>5}"
        row += f"  {t:>6}"
        print(row)

    # ==============================
    # DK data
    # ==============================
    print(f"\n{'='*70}")
    print("[2] DK (NKY) ear tag analysis")
    print(f"{'='*70}")

    for batch, bname in [('nkyfirst','NKY1'), ('nkysecond','NKY2')]:
        bdir = os.path.join(DK_BASE, batch)
        if not os.path.isdir(bdir):
            continue
        pigs = sorted(d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d)))
        dong_groups = defaultdict(list)
        for p in pigs:
            m = re.match(r'^[c]*(\d+)-', p)
            if m:
                dong_groups[m.group(1)].append(p)
        print(f"\n  {bname} ({batch}): {len(pigs)} pigs")
        for dong, ps in sorted(dong_groups.items()):
            print(f"    Dong {dong}: {len(ps)} pigs -> {', '.join(ps[:5])} ...")

    nky3 = os.path.join(DK_BASE, 'nkythird')
    if os.path.isdir(nky3):
        all3 = []
        for kinect in sorted(os.listdir(nky3)):
            kp = os.path.join(nky3, kinect)
            if not os.path.isdir(kp):
                continue
            for pig in os.listdir(kp):
                if os.path.isdir(os.path.join(kp, pig)):
                    all3.append((kinect, pig))
        print(f"\n  NKY3 (nkythird): {len(all3)} dirs")
        grp = defaultdict(list)
        for k, p in all3:
            if p.startswith('wc'): grp['wc(no label)'].append(p)
            elif p.startswith('w'): grp['w(no label)'].append(p)
            elif p.startswith('c'): grp['c(variant)'].append(p)
            elif p.startswith('k'): grp['k'].append(p)
            else: grp['normal'].append(p)
        for g, ps in sorted(grp.items(), key=lambda x: -len(x[1])):
            dongs = set()
            for p in ps:
                m = re.match(r'^[cwk]*(\d+)-', p)
                if m: dongs.add(m.group(1))
            dong_str = f" (dong: {','.join(sorted(dongs))})" if dongs else ""
            print(f"    {g}: {len(ps)} pigs{dong_str}, e.g. {', '.join(sorted(ps)[:3])}")

    print(f"\n{'='*70}")
    print("Done")

if __name__ == '__main__':
    main()
