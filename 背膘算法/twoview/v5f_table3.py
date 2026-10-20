# -*- coding: utf-8 -*-
"""
v5f_table3.py — 汇总 Table 3（A 场 663 头，同一套 5 折的折外预测）：所有行来自已有输出，唯一新算的是 R + P（TabPFN，GPU，
                与 v4h 相同的拟合设置与种子 42+k），使「记录」各行与融合各行出自同一设置

来源 / 背景
  DECISIONS 2026-10-13（四）：主结果统一用高度图 CNN；稿件 Results 3.3 需要一张可追溯的总表。
  v1 的 T2 TabPFN（记录 + 上季）在训练 + 验证集上拟合，与 v4h 的「记录」（只在外层训练集上拟合）设置略不同
  （同为 R：0.154 vs 0.158）→ 本脚本按 v4h 设置补算 R + P。

行（预测变量组 → 来源）
  Training mean                     v1_tabular/preds.csv: mean
  R; R+C; R+image; R+C+image; R+P+C; R+P+C+image           v4_hmap/nested_fusion/preds.csv（GPU）
  R+P                               本脚本（TabPFN，GPU，random_state 42+k）
  CNN, height map（3 种子平均）      v4_hmap/hmap/test_preds.csv: pred_mean
  CNN, cropped depth map（3 种子）   v3_views/master_s128/test_preds.csv: pred_mean
  B whole-body dims                 v4_hmap/body_dims/preds.csv
  C caliper; V virtual caliper; G image geometry; R+P+C+image+G   v5_caliper/fusion/preds.csv（GPU）
  配对比较（group bootstrap 95% CI + Wilcoxon）：各行 vs R；CNN 高度图 vs 裁剪深度图 / 卡尺 / R + P / 体尺

输出：runs/v5_table3/{log.txt, table3.csv, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5f_table3.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
import a3_foundation_regression as A3  # noqa: E402
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402

RUNS = RUNS_ROOT
OUT = os.path.join(RUNS, 'v5_table3')


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return {r['uid']: r for r in csv.DictReader(f)}


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

    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] v5f Table 3 assembly')
    with open(os.path.join(RUNS, 'v1_tabular', 'preds.csv'), encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])

    src = {'nest': read(os.path.join(RUNS, 'v4_hmap', 'nested_fusion', 'preds.csv')),
           'hmap': read(os.path.join(RUNS, 'v4_hmap', 'hmap', 'test_preds.csv')),
           'crop': read(os.path.join(RUNS, 'v3_views', 'master_s128', 'test_preds.csv')),
           'body': read(os.path.join(RUNS, 'v4_hmap', 'body_dims', 'preds.csv')),
           'v5c': read(os.path.join(RUNS, 'v5_caliper', 'fusion', 'preds.csv'))}
    for k, d in src.items():
        assert all(u in d for u in uid), f'{k} 缺猪'
        assert np.array_equal(fold, [int(d[u]['fold']) for u in uid]), f'{k} 折不一致'
        assert np.allclose(y, [float(d[u]['y']) for u in uid]), f'{k} 标签不一致'
    col = lambda s, c: np.array([float(src[s][u][c]) for u in uid])
    # v4h 与 v5c(GPU) 的 R 只有 GPU 浮点级差异（2026-10-13 实测 max 0.022 mm、mean 0.001 mm）
    assert np.abs(col('nest', 'R') - col('v5c', 'R')).max() < 0.05, 'v4h 与 v5c(GPU) 的 R 不一致'

    # R + P：与 v4h 同设置（外层训练集拟合，random_state 42+k，GPU 优先）
    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    RP = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        X = np.hstack([R, P])
        RP[te] = A3.fit_predict_tabpfn(X[tr], y[tr], X[te], 42 + k)
    log(f'   R+P fitted | device {sorted(A3.DEVICE_USED)}')

    preds = {
        'Training mean': np.array([float(r['mean']) for r in rows]),
        'R': col('nest', 'R'),
        'R+P': RP,
        'CNN, cropped depth map': col('crop', 'pred_mean'),
        'CNN, height map': col('hmap', 'pred_mean'),
        'B, whole-body dimensions': col('body', 'B'),
        'V, virtual caliper': col('v5c', 'V'),
        'G, image geometry': col('v5c', 'G'),
        'C, manual caliper': col('v5c', 'C'),
        'R+C': col('nest', 'R+cal'),
        'R+image': col('nest', 'R+img'),
        'R+C+image': col('nest', 'R+cal+img'),
        'R+P+C': col('nest', 'R+prev+cal'),
        'R+P+C+image': col('nest', 'R+prev+cal+img'),
        'R+P+C+image+G': col('v5c', 'R+P+C+img+G'),
    }
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel                        ' + ''.join(f'{s:>22s}' for s in subsets) + '   (R² / MAE)')
    for nm in preds:
        log(f'   {nm:26s}' + ''.join(f'{table[nm][s]["R2"]:>13.3f} / {table[nm][s]["MAE"]:.2f}' for s in subsets))

    pairs = [(nm, 'R') for nm in preds if nm not in ('R', 'Training mean')]
    pairs += [('CNN, height map', 'CNN, cropped depth map'), ('CNN, height map', 'C, manual caliper'),
              ('CNN, height map', 'R+P'), ('CNN, height map', 'B, whole-body dimensions'),
              ('R+P+C+image+G', 'R+P+C+image')]
    comp = {}
    log('\npaired comparisons (Δ = a − b): ΔR² (group bootstrap 95% CI), ΔMAE, Wilcoxon p on |error|')
    for a_, b_ in pairs:
        lab = f'{a_} vs {b_}'
        comp[lab] = {}
        for s in ('all', 'DAN_parity1', 'NAM_multiparous', 'no_prev_season'):
            m = subsets[s]
            bs = group_bootstrap(group[m], y[m], preds[a_][m], preds[b_][m])
            pw = float(wilcoxon(np.abs(preds[a_][m] - y[m]), np.abs(preds[b_][m] - y[m])).pvalue)
            comp[lab][s] = dict(**bs, wilcoxon_p=pw)
        c = comp[lab]['all']
        log(f'   {lab:52s} all ΔR² {c["dR2"][1]:+.3f} ({c["dR2"][0]:+.3f}, {c["dR2"][2]:+.3f}) '
            f'ΔMAE {c["dMAE"][1]:+.3f} p={c["wilcoxon_p"]:.2g} | DAN_p1 ΔR² {comp[lab]["DAN_parity1"]["dR2"][1]:+.3f} '
            f'p={comp[lab]["DAN_parity1"]["wilcoxon_p"]:.2g} | first ΔR² {comp[lab]["no_prev_season"]["dR2"][1]:+.3f} '
            f'p={comp[lab]["no_prev_season"]["wilcoxon_p"]:.2g}')

    with open(os.path.join(OUT, 'table3.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model'] + [f'{s}_{k}' for s in subsets for k in ('n', 'R2', 'MAE', 'RMSE', 'CCC')])
        for nm in preds:
            w.writerow([nm] + [round(float(table[nm][s][k]), 4) if k != 'n' else table[nm][s][k]
                               for s in subsets for k in ('n', 'R2', 'MAE', 'RMSE', 'CCC')])
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(preds))
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], y[i]] + [round(float(preds[k][i]), 3) for k in preds])
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(table=table, comparisons=comp), f, ensure_ascii=False, indent=1, default=float)
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
