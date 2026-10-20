# -*- coding: utf-8 -*-
"""
v6a_camera_caliper_fusion.py — 不含生产记录的「相机 × 人工卡尺 × 图像几何」组合（本文新范围，N1）

来源 / 背景
  DECISIONS 2026-10-14：导师要求本文只做「背膘无接触式测定方法」，多场、跨季结果移出；本人确认生产记录只留
  「R 单独」一行作参照，不做「记录 + X」融合网格。v4h / v5c 的组合都含生产记录（R），本脚本补算不含 R 的版本。
  09_研究成果总表 §4 N1。（命名：v6_conformal_triage.py 是旧的 V6，本脚本为 v6a，二者无关。）

做法（与 v5c 完全相同，只换特征组）
  - 同一套 5 折；图像分数 img = v4h 无泄漏 CNN 分数（测试猪 = 外层模型 seed 42 预测，训练猪 = 内层 OOF），不重新训练
  - 第二层 TabPFN（A3.fit_predict_tabpfn，GPU；种子 42 + k）
  - 特征组：R（参照）/ C / img / G / B / C+img / img+G / img+B / C+G / C+img+G
    C = 人工卡尺；G = 虚拟卡尺角 + 曲率 + 肩宽、肋宽、臀宽、臀高（v5b，f 0.60、D 150）；B = v4e 整体体尺 6 列
  - 配对比较：group bootstrap 95% CI + Wilcoxon（|误差|）；子集只用全体、丹系头胎、新美系经产（不用跨季子集）

输入：runs/v1_tabular/preds.csv、runs/v1b_caliper/preds.csv、runs/v5_caliper/features.csv、
      runs/v4_hmap/nested_fusion/{preds.csv, inner_scores.csv}、hmap_dataset/metadata.csv（v4e 体尺）
输出：runs/v6_camera_caliper/{log.txt, result.json, preds.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6a_camera_caliper_fusion.py
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
from v4e_body_dims_fusion import pig_dims  # noqa: E402
from v5c_virtual_caliper_fusion import CAL, FEAT, G_COLS, NEST, V1, fnum  # noqa: E402

OUT = os.path.join(RUNS_ROOT, 'v6_camera_caliper')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nest', default=NEST, help='嵌套图像分数目录（preds.csv + inner_scores.csv）；默认 v4h（ResNet-18）')
    ap.add_argument('--out', default=OUT, help='输出目录；2026-10-15 起新主模型用 runs/v6_camera_caliper_anatomy')
    a = ap.parse_args()
    nest_dir, out_dir = a.nest, a.out
    A3.FORCE_CPU = False  # 全文统一 GPU（DECISIONS 2026-10-13（五））
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'log.txt'), 'a', encoding='utf-8')

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
    with open(os.path.join(nest_dir, 'preds.csv'), encoding='utf-8-sig') as f:
        nest = {r['uid']: r for r in csv.DictReader(f)}
    with open(os.path.join(nest_dir, 'inner_scores.csv'), encoding='utf-8-sig') as f:
        inner = {r['uid']: r for r in csv.DictReader(f)}

    uid = [r['uid'] for r in rows]
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    group = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    assert all(u in feat and u in nest and u in inner for u in uid), 'uid 不一致'
    assert np.array_equal(fold, [int(nest[u]['fold']) for u in uid]), '折不一致'
    cal = np.array([cal_map.get(u, np.nan) for u in uid])
    G = np.array([[fnum(feat[u].get(c)) for c in G_COLS] for u in uid])
    B, _ = pig_dims(uid)
    outer_score = np.array([fnum(nest[u]['img_raw (outer CNN, seed 42)']) for u in uid])
    folds = sorted(set(fold))
    inner_by_fold = {k: np.array([fnum(inner[u][f'inner_score_outer{k}']) for u in uid]) for k in folds}
    for k in folds:
        assert not np.isnan(inner_by_fold[k][fold != k]).any(), f'外层 {k} 训练猪缺内层分数'
    log(f'[{time.strftime("%F %T")}] v6a | nest {nest_dir} | pigs {len(y)}; '
        f'caliper missing {int(np.isnan(cal).sum())}; G cols {", ".join(G_COLS)}')

    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    C = cal[:, None]
    sets = {
        'R': lambda I: R, 'C': lambda I: C, 'img': lambda I: I, 'G': lambda I: G, 'B': lambda I: B,
        'C+img': lambda I: np.hstack([C, I]), 'img+G': lambda I: np.hstack([I, G]),
        'img+B': lambda I: np.hstack([I, B]), 'C+G': lambda I: np.hstack([C, G]),
        'C+img+G': lambda I: np.hstack([C, I, G]),
    }
    preds = {k: np.zeros(len(y)) for k in sets}
    for k in folds:
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        I = np.where(fold == k, outer_score, inner_by_fold[k])[:, None]
        for name, fn in sets.items():
            X = fn(I)
            preds[name][te] = A3.fit_predict_tabpfn(X[tr], y[tr], X[te], 42 + k)
        log(f'  fold {k} done | device {sorted(A3.DEVICE_USED)} | {(time.time() - t0) / 60:.1f} min')

    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel      subset             n    MAE    RMSE   R2     CCC')
    for nm in preds:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:10s} {s:16s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:.3f}  {r["CCC"]:.3f}')
    pairs = [('img', 'C', 'camera vs manual caliper (stacked image score)'),
             ('C+img', 'C', 'camera over manual caliper'),
             ('C+img', 'img', 'manual caliper over camera'),
             ('img+G', 'img', 'image geometry over CNN score'),
             ('img+B', 'img', 'whole-body dims over CNN score'),
             ('C+img+G', 'C+img', 'image geometry over caliper+camera'),
             ('G', 'C', 'image geometry vs manual caliper'),
             ('C+G', 'C', 'image geometry over manual caliper'),
             ('img', 'R', 'camera vs production records (reference)')]
    comp = {}
    log('\npaired comparisons (Δ = a − b): group bootstrap 95% CI; Wilcoxon on |error|')
    for a_, b_, lab in pairs:
        comp[lab] = {}
        for s, m in subsets.items():
            bs = group_bootstrap(group[m], y[m], preds[a_][m], preds[b_][m])
            pw = float(wilcoxon(np.abs(preds[a_][m] - y[m]), np.abs(preds[b_][m] - y[m])).pvalue)
            comp[lab][s] = dict(**bs, wilcoxon_p=pw)
            log(f'   {lab:48s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(G_cols=G_COLS, table=table, comparisons=comp), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(out_dir, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'y'] + list(preds))
        for i, u in enumerate(uid):
            w.writerow([u, group[i], fold[i], y[i]] + [round(float(preds[k][i]), 3) for k in preds])
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {out_dir}')


if __name__ == '__main__':
    main()
