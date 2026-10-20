# -*- coding: utf-8 -*-
"""
v6o_make_exclusions.py — 生成 2026-10-20 的剔除名单与标签修正表（只在笔记本上运行一次）

背景（DECISIONS 2026-10-20（二））：
  - 导师决定剔除背膘 > 25 mm 的记录（B 超在该范围不准；保留 25 mm，与吕年珅稿 9–25 mm 一致）；
  - 本人确认：吕年珅 labels最新.json 相对 labels.json 删掉的、标签 ≤ 25 mm 的耳标是流产母猪
    （吕年珅、导师、林厂长已确认），我们的数据按同一名单剔除；
  - 吕年珅修正的原始表中，对我们有影响的两处：
      fmh_8.26_661A0035 背膘 18 → 14（8.26 节原表写成 A335；胎次 2、配种 7.24 与 A0035 一致；我们此前按同季回退匹配成 18）
      fmh_11.1_3A0762   背膘 70 → 20（录入错误；该记录一直不在分析集里，见下）
规则：
  1. 背膘 > 25 mm：按我们按日期匹配、修正后的标签逐条判断（不是按吕年珅「同耳标只取一个值」的标签）；
  2. 流产：耳标在吕年珅剔除名单中且其标签 ≤ 25 mm → 剔除该耳标在本数据中的全部记录（本数据中均只有 1 条）；
  3. fmh_11.1_3A0762 只修正标签、不加回分析集：它没有俯视裁剪深度图（depthmap_dataset 中无此猪），
     而表示 / 视角比较要求所有模型用同一批猪。
输入：吕年珅标签 json（--lv-dir）、index_v2.csv、runs/framepreds_new_s128/test_preds.csv（原 663 条分析集）
输出：twoview/v6o_exclusions.csv、twoview/v6o_label_fixes.csv（供 v6o_make_subset.py 使用）
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6o_make_exclusions.py [--lv-dir H:\\耿来硕数据\\丰美禾数据\\吕年珅标签]
"""
import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
FOLDS_663 = os.path.join(HERE, 'runs', 'framepreds_new_s128', 'test_preds.csv')
INDEX = os.path.join(HERE, 'index_v2.csv')
BF_MAX = 25.0
LABEL_FIXES = [
    ('fmh_8.26_661A0035', 'backfat', 18.0, 14.0,
     '吕年珅修正表：8.26 节 A335 → A0035（胎次 2、配种 7.24 与 A0035 一致）；原按同季回退匹配为 18'),
    ('fmh_11.1_3A0762', 'backfat', 70.0, 20.0,
     '吕年珅修正表：录入错误 70 → 20；无俯视裁剪深度图，不加回分析集'),
]


def norm_tag(t):
    """耳标标准化：去栏位数字前缀（80A287 → A287）、字母 O → 0、去数字部分前导零（A0035 → A35）。"""
    t = str(t).strip().upper().replace('O', '0')
    m = re.match(r'^\d*([A-Z]+)0*(\d+)$', t)
    return (m.group(1) + m.group(2)) if m else t


def lv_tags(path):
    d = json.load(open(path, encoding='utf-8'))
    tags = {}
    for k, v in d.items():
        m = re.match(r'^(.*)_(\d+)\.npy$', k)
        tags.setdefault(m.group(1) if m else k, float(v))
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lv-dir', default=r'H:\耿来硕数据\丰美禾数据\吕年珅标签')
    a = ap.parse_args()
    old = lv_tags(os.path.join(a.lv_dir, 'labels.json'))
    new = lv_tags(os.path.join(a.lv_dir, 'labels最新.json'))
    removed = {t: v for t, v in old.items() if t not in new}
    aborted = {norm_tag(t): (t, v) for t, v in removed.items() if v <= BF_MAX}
    print(f'吕年珅 json：{len(old)} → {len(new)} 个耳标，删去 {len(removed)} 个；其中标签 ≤ {BF_MAX:g} mm（流产）{len(aborted)} 个')

    idx = {r['uid']: r for r in csv.DictReader(open(INDEX, encoding='utf-8-sig'))}
    fixes = {u: new_v for u, _, _, new_v, _ in LABEL_FIXES}
    for u, _, old_v, _, _ in LABEL_FIXES:
        assert float(idx[u]['backfat']) == old_v, f'{u}: index_v2 backfat 不是 {old_v}'
    recs = list(csv.DictReader(open(FOLDS_663, encoding='utf-8-sig')))
    out = []
    for r in recs:
        u = r['uid']
        x = idx[u]
        y = fixes.get(u, float(r['y']))
        reasons = []
        if y > BF_MAX:
            reasons.append('backfat_gt25')
        hit = aborted.get(norm_tag(x['ear_key']))
        if hit:
            reasons.append('aborted_lv')
        if reasons:
            out.append(dict(uid=u, group=r['group'], batch=x['batch'], breed=x['breed'], parity=x['parity'],
                            backfat=y, reason='+'.join(reasons), lv_tag=hit[0] if hit else '',
                            lv_label=hit[1] if hit else ''))
    with open(os.path.join(HERE, 'v6o_exclusions.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(HERE, 'v6o_label_fixes.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'field', 'old', 'new', 'source'])
        w.writerows(LABEL_FIXES)
    n_gt = sum('backfat_gt25' in r['reason'] for r in out)
    n_ab = sum('aborted_lv' in r['reason'] for r in out)
    keep = len(recs) - len(out)
    print(f'原分析集 {len(recs)} 条；剔除 {len(out)} 条（> {BF_MAX:g} mm {n_gt}，流产 {n_ab}，两者皆是 '
          f'{n_gt + n_ab - len(out)}）→ 保留 {keep} 条')
    print('→ v6o_exclusions.csv、v6o_label_fixes.csv')


if __name__ == '__main__':
    main()
