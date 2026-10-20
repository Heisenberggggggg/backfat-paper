# -*- coding: utf-8 -*-
"""
v6d_table1.py — 稿件 v2 的 Table 1（只含 A 场，按品系 × 胎次；加卡尺读数列）

来源 / 背景
  DECISIONS 2026-10-14：本文只用 A 场，Table 1 删去 B 场与按测定季拆分的行，增加卡尺读数列。
  v1 的 Table 1 由 2026-10-12 现算、未留脚本；本脚本把全部列重算一遍并留 log，便于追溯。

输入：runs/v1_tabular/preds.csv（uid, group, line, parity, gest_day, y）、runs/v1b_caliper/preds.csv（caliper, caliper_flag）
输出：runs/v6_table1/{log.txt, table1.csv}
用法：D:\\Anaconda\\python.exe v6d_table1.py
"""
import csv
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
OUT = os.path.join(HERE, 'runs', 'v6_table1')


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    with open(V1, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(CAL, encoding='utf-8-sig') as f:
        cal = {r['uid']: (np.nan if r['caliper_flag'] == 'implausible' else fnum(r['caliper']))
               for r in csv.DictReader(f)}
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    group = np.array([r['group'] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    c = np.array([cal.get(r['uid'], np.nan) for r in rows])
    subsets = {'All': np.ones(len(y), bool),
               'Danish line, parity 1': (line == 'DAN') & (parity == 1),
               'Danish line, parity >= 2': (line == 'DAN') & (parity >= 2),
               'American line, parity 1': (line == 'NAM') & (parity == 1),
               'American line, parity >= 2': (line == 'NAM') & (parity >= 2)}
    log(f'[{time.strftime("%F %T")}] v6d Table 1 (Farm A only) | records {len(y)}, sows {len(set(group))}')
    hdr = ['subset', 'records', 'sows', 'backfat_mean', 'backfat_sd', 'backfat_min', 'backfat_max',
           'caliper_n', 'caliper_mean', 'caliper_sd', 'gest_n', 'gest_mean', 'gest_sd']
    out = []
    for name, m in subsets.items():
        cm, gm = m & ~np.isnan(c), m & ~np.isnan(gest)
        row = [name, int(m.sum()), len(set(group[m])), y[m].mean(), y[m].std(ddof=1), y[m].min(), y[m].max(),
               int(cm.sum()), c[cm].mean(), c[cm].std(ddof=1), int(gm.sum()), gest[gm].mean(), gest[gm].std(ddof=1)]
        out.append(row)
        log(f'  {name:27s} records {row[1]:3d} sows {row[2]:3d} | backfat {row[3]:.2f} ± {row[4]:.2f} '
            f'({row[5]:g}–{row[6]:g}) | caliper {row[8]:.2f} ± {row[9]:.2f} (n {row[7]}) | '
            f'gestation day {row[11]:.0f} ± {row[12]:.0f} (n {row[10]})')
    with open(os.path.join(OUT, 'table1.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows([[v if isinstance(v, (str, int)) else round(float(v), 3) for v in r] for r in out])
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
