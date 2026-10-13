# -*- coding: utf-8 -*-
"""
v5e_tabpfn_seed_robustness.py — Q0–Q3 与几何特征增量对 TabPFN 随机种子是否稳健

来源 / 背景
  DECISIONS 2026-10-13（五）：同特征同种子下 TabPFN 的 CPU 与 GPU 结果相差 0.02–0.03（R²），与 Q0–Q3 增量
  （+0.04～+0.20）同一量级 → 检查主比较是否依赖 TabPFN 的随机种子。
  第一层（高度图 CNN 嵌套分数）不变：测试猪 = v4h 外层模型（seed 42），训练猪 = v4h 内层 OOF；只重跑第二层 TabPFN。
  种子组：random_state = 42+k（与 v4h 相同）、142+k、242+k（k = 外层折号）；另报 3 组预测取平均（种子集成）的结果。

比较：Q0 R+img vs R；Q1 R+img vs R+cal；Q2 R+cal+img vs R+cal；Q3 R+prev+cal+img vs R+prev+cal；
      G R+img+G vs R+img（G = v5b 几何特征，同 v5c）

输入：同 v5c（runs/v1_tabular、runs/v1b_caliper、runs/v5_caliper/features.csv、runs/v4_hmap/nested_fusion）
输出：runs/v5_robust/{log.txt, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5e_tabpfn_seed_robustness.py
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
sys.path.insert(0, HERE)
import a3_foundation_regression as A3  # noqa: E402
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v5c_virtual_caliper_fusion import CAL, FEAT, G_COLS, NEST, V1, fnum  # noqa: E402

OUT = os.path.join(HERE, 'runs', 'v5_robust')
SEED_OFFSETS = (42, 142, 242)


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(V1, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(CAL, encoding='utf-8-sig') as f:
        cal_map = {r['uid']: (fnum(r['caliper']) if r['caliper_flag'] != 'implausible' else np.nan)
                   for r in csv.DictReader(f)}
    with open(FEAT, encoding='utf-8-sig') as f:
        feat = {r['uid']: r for r in csv.DictReader(f)}
    with open(os.path.join(NEST, 'preds.csv'), encoding='utf-8-sig') as f:
        nest = {r['uid']: r for r in csv.DictReader(f)}
    with open(os.path.join(NEST, 'inner_scores.csv'), encoding='utf-8-sig') as f:
        inner = {r['uid']: r for r in csv.DictReader(f)}
    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    assert np.array_equal(fold, [int(nest[u]['fold']) for u in uid]), '折不一致'
    cal = np.array([cal_map.get(u, np.nan) for u in uid])
    G = np.array([[fnum(feat[u].get(c)) for c in G_COLS] for u in uid])
    outer_score = np.array([fnum(nest[u]['img_raw (outer CNN, seed 42)']) for u in uid])
    folds = sorted(set(fold))
    inner_by_fold = {k: np.array([fnum(inner[u][f'inner_score_outer{k}']) for u in uid]) for k in folds}

    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    C = cal[:, None]
    sets = {'R': lambda I: R, 'R+img': lambda I: np.hstack([R, I]), 'R+cal': lambda I: np.hstack([R, C]),
            'R+cal+img': lambda I: np.hstack([R, C, I]), 'R+prev+cal': lambda I: np.hstack([R, P, C]),
            'R+prev+cal+img': lambda I: np.hstack([R, P, C, I]), 'R+img+G': lambda I: np.hstack([R, I, G])}
    pairs = [('R+img', 'R', 'Q0'), ('R+img', 'R+cal', 'Q1'), ('R+cal+img', 'R+cal', 'Q2'),
             ('R+prev+cal+img', 'R+prev+cal', 'Q3'), ('R+img+G', 'R+img', 'G over CNN')]
    log(f'[{time.strftime("%F %T")}] v5e TabPFN seed robustness | seed offsets {SEED_OFFSETS}')

    preds = {s: {k: np.zeros(len(y)) for k in sets} for s in SEED_OFFSETS}
    for s in SEED_OFFSETS:
        for k in folds:
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            I = np.where(fold == k, outer_score, inner_by_fold[k])[:, None]
            for name, fn in sets.items():
                X = fn(I)
                preds[s][name][te] = A3.fit_predict_tabpfn(X[tr], y[tr], X[te], s + k)
        log(f'  seed offset {s} done | device {sorted(A3.DEVICE_USED)} | {(time.time() - t0) / 60:.1f} min')
    preds['ensemble'] = {k: np.mean([preds[s][k] for s in SEED_OFFSETS], 0) for k in sets}

    res = {}
    log('\nR² per model: ' + ' | '.join(str(s) for s in preds))
    for name in sets:
        log(f'   {name:16s} ' + '  '.join(f'{metrics(y, preds[s][name])["R2"]:.3f}' for s in preds))
    log('\npaired comparisons (Δ = a − b, all 663): ΔR² (group bootstrap 95% CI), Wilcoxon p')
    for a_, b_, lab in pairs:
        res[lab] = {}
        parts = []
        for s in preds:
            bs = group_bootstrap(group, y, preds[s][a_], preds[s][b_])
            pw = float(wilcoxon(np.abs(preds[s][a_] - y), np.abs(preds[s][b_] - y)).pvalue)
            res[lab][str(s)] = dict(**bs, wilcoxon_p=pw)
            parts.append(f'{s}: {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f}) p={pw:.2g}')
        log(f'   {lab:10s} ' + ' | '.join(parts))
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        cols = [(s, k) for s in preds for k in sets]
        w.writerow(['uid', 'group', 'fold', 'y'] + [f'{k}|{s}' for s, k in cols])
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], y[i]] + [round(float(preds[s][k][i]), 3) for s, k in cols])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
