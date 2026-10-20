# -*- coding: utf-8 -*-
"""
v6o_make_subset.py — 建立剔除后子集的 runs 根目录（runs_v6o/），供 v6o_rerun_subset.py 在其上重跑全部分析

背景：DECISIONS 2026-10-20（二）。剔除名单与标签修正见 v6o_make_exclusions.py → v6o_exclusions.csv、v6o_label_fixes.csv
（663 条 → 622 条：背膘 > 25 mm 26 条、流产 15 条；fmh_8.26_661A0035 背膘 18 → 14）。

做法（只读 runs/ 与数据集，不改动任何旧结果）：
  1. index_v2_v6o.csv = index_v2.csv 加两处标签修正（其余逐字相同）；
  2. 外层 5 折**在 622 条上按原规则重新划分**：背膘五分位分层、按个体分组、StratifiedGroupKFold(5, shuffle, seed 42)，
     与 cv_depthmap_labelfix.py / v1_tabular_baseline.py 完全相同（v1_tabular 会重建折并逐头核对 baseline 列，不一致即中止）；
     → labelfix_new_s128/test_preds.csv 与 framepreds_new_s128/test_preds.csv（pred 列为旧 663 条模型的预测，仅作占位，
       新分析不用；cropped-depth 的逐帧预测由 v3b master 重跑后另行生成）；
  3. 横截面缓存 v6_anatomy/_cache/profiles_v2.npz：按 v4b / v6f 的帧质控在 622 条上重算帧列表，从旧缓存中按帧路径取子集
     （帧质控只用每头猪自己的中位数，所以剔除别的猪不改变保留猪的帧；脚本内 assert）；
  4. 冻结基础模型特征 a3_foundation/feat_*.npz：按新的猪顺序取子集（特征提取与样本无关，不重算）。
输入：twoview/{index_v2.csv, v6o_exclusions.csv, v6o_label_fixes.csv}、runs/{labelfix_new_s128, v6_anatomy/_cache,
      a3_foundation}、yolo+FastSAM/hmap_dataset/metadata.csv
输出：<out>/{index_v2_v6o.csv, labelfix_new_s128/test_preds.csv, framepreds_new_s128/test_preds.csv,
      v6_anatomy/_cache/profiles_v2.npz, a3_foundation/feat_*.npz, _seed/{log.txt, v6o_exclusions.csv, v6o_label_fixes.csv}}
用法：python v6o_make_subset.py [--out runs_v6o]   （已存在的输出文件不覆盖，需 --force）
"""
import argparse
import csv
import os
import shutil
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
OLD = os.path.join(HERE, 'runs')                      # 旧结果根目录（只读）
INDEX = os.path.join(HERE, 'index_v2.csv')
EXCL = os.path.join(HERE, 'v6o_exclusions.csv')
FIXES = os.path.join(HERE, 'v6o_label_fixes.csv')
HMETA = os.path.join(HERE, '..', 'yolo+FastSAM', 'hmap_dataset', 'metadata.csv')
FEATS = ['feat_dinov3-vitb16_relh.npz', 'feat_dinov3-vitb16_hmap.npz', 'feat_dinov2-base_relh.npz']
AREA_REL, LEN_REL = 0.8, 0.85                          # 与 v4b / v6f 默认帧质控相同
SEED = 42


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write(path, rows, fields, force):
    if os.path.exists(path) and not force:
        sys.exit(f'已存在，不覆盖：{path}（确需重建请加 --force）')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def hmap_frame_paths(uids):
    """复现 v4b.load_hmaps / v6f.load_hmaps_paths 的帧选择（不读图像），返回保留帧的 frame_path 列表。"""
    pos = set(uids)
    rows = [r for r in read(HMETA) if r['uid'] in pos and r['ok'] == '1']
    by = {}
    for r in rows:
        by.setdefault(r['uid'], []).append(r)
    keep = []
    for u, rs in by.items():
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        keep += [r for r in rs if float(r['area_cm2']) >= AREA_REL * ma and float(r['length_mm']) >= LEN_REL * ml]
    return [r['frame_path'] for r in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'runs_v6o'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, '_seed'), exist_ok=True)
    logf = open(os.path.join(out, '_seed', 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] v6o_make_subset → {out}')
    excl = {r['uid']: r['reason'] for r in read(EXCL)}
    fixes = {r['uid']: float(r['new']) for r in read(FIXES) if r['field'] == 'backfat'}
    log(f'   exclusions {len(excl)} ({sorted(set(excl.values()))}); label fixes {fixes}')

    # 1. 修正后的索引
    with open(INDEX, encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        fields, idx_rows = rd.fieldnames, list(rd)
    for r in idx_rows:
        if r['uid'] in fixes:
            log(f'   index fix {r["uid"]}: backfat {r["backfat"]} → {fixes[r["uid"]]:g}')
            r['backfat'] = f'{fixes[r["uid"]]:g}'
    write(os.path.join(out, 'index_v2_v6o.csv'), idx_rows, fields, a.force)
    bf = {r['uid']: float(r['backfat']) for r in idx_rows}

    # 2. 622 条与新的外层折（与 v1_tabular_baseline.py 的重建逻辑逐行相同）
    old = read(os.path.join(OLD, 'labelfix_new_s128', 'test_preds.csv'))
    rows = sorted([r for r in old if r['uid'] not in excl], key=lambda r: r['uid'])
    assert len(old) - len(rows) == len(excl), '剔除名单中有不在原分析集里的 uid'
    for r in rows:
        y = fixes.get(r['uid'], float(r['y_new']))
        assert y == bf[r['uid']], r['uid']
        r['y_used'] = r['y_new'] = f'{y:g}'
    n = len(rows)
    y = np.array([float(r['y_new']) for r in rows])
    groups = np.array([r['group'] for r in rows])
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    fold_of, base = np.zeros(n, int), np.zeros(n)
    for k, (trv, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                                  .split(np.zeros(n), strat, groups)):
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED + k)
        tr_i, _ = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
        fold_of[te] = k + 1
        base[te] = y[trv[tr_i]].mean()
    for r, b in zip(rows, base):
        r['baseline'] = f'{b:.3f}'
    write(os.path.join(out, 'labelfix_new_s128', 'test_preds.csv'), rows,
          ['uid', 'group', 'y_used', 'y_new', 'y_old', 'pred', 'baseline', 'match_level'], a.force)
    frows = [dict(uid=r['uid'], group=r['group'], fold=int(k), y=r['y_new'], pred=r['pred'], baseline=r['baseline'])
             for r, k in zip(rows, fold_of)]
    write(os.path.join(out, 'framepreds_new_s128', 'test_preds.csv'), frows,
          ['uid', 'group', 'fold', 'y', 'pred', 'baseline'], a.force)
    uids = [r['uid'] for r in rows]
    log(f'   records {n}, sows {len(set(groups))}; backfat mean {y.mean():.2f} SD {y.std(ddof=1):.2f} '
        f'range {y.min():g}–{y.max():g}')
    for k in range(1, 6):
        m = fold_of == k
        log(f'   fold {k}: {m.sum()} records, {len(set(groups[m]))} sows, backfat mean {y[m].mean():.2f}')
    assert all(len(set(fold_of[groups == g])) == 1 for g in set(groups)), '同一头猪跨折'

    # 3. 横截面缓存取子集
    src = os.path.join(OLD, 'v6_anatomy', '_cache', 'profiles_v2.npz')
    dst = os.path.join(out, 'v6_anatomy', '_cache', 'profiles_v2.npz')
    if os.path.exists(dst) and not a.force:
        sys.exit(f'已存在，不覆盖：{dst}')
    z = np.load(src, allow_pickle=True)
    old_paths = list(z['paths'])
    new_paths = hmap_frame_paths(uids)
    keep_set = set(uids)
    assert new_paths == [p for p in old_paths if p.split('\\')[0].split('/')[0] in keep_set], \
        '帧列表不是旧列表的有序子集'
    pos = {p: i for i, p in enumerate(old_paths)}
    sel = np.array([pos[p] for p in new_paths])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, prof=z['prof'][sel], valid=z['valid'][sel], paths=np.array(new_paths, dtype=object), f_pos=z['f_pos'])
    log(f'   profiles cache: {len(old_paths)} → {len(new_paths)} frames → {dst}')

    # 4. 基础模型特征取子集
    for fn in FEATS:
        s = os.path.join(OLD, 'a3_foundation', fn)
        d = os.path.join(out, 'a3_foundation', fn)
        if not os.path.exists(s):
            log(f'   (skip, not found) {s}')
            continue
        if os.path.exists(d) and not a.force:
            sys.exit(f'已存在，不覆盖：{d}')
        z = np.load(s, allow_pickle=True)
        ou = list(z['uids'])
        pi = {u: i for i, u in enumerate(ou)}
        pig_sel = np.array([pi[u] for u in uids])
        remap = {old_i: new_i for new_i, old_i in enumerate(pig_sel)}
        fsel = np.array([i for i, o in enumerate(z['owner']) if int(o) in remap])
        os.makedirs(os.path.dirname(d), exist_ok=True)
        np.savez(d, uids=np.array(uids), owner=np.array([remap[int(o)] for o in z['owner'][fsel]]),
                 frame_feat=z['frame_feat'][fsel], pig_feat=z['pig_feat'][pig_sel])
        log(f'   {fn}: pigs {len(ou)} → {len(uids)}, frames {len(z["owner"])} → {len(fsel)}')

    for p in (EXCL, FIXES):
        shutil.copy2(p, os.path.join(out, '_seed', os.path.basename(p)))
    log(f'[{time.strftime("%F %T")}] done')


if __name__ == '__main__':
    main()
