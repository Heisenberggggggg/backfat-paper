# -*- coding: utf-8 -*-
"""
v4g_hmap_reliability.py — 用高度图 ResNet-18 的 OOF 预测重算 A1 / A2 / A6 与 V6 保形区间（不训练）

来源 / 背景
  DECISIONS 2026-10-13：视觉主模型由俯视深度图换为标准化高度图（R² 0.309 → 0.378）。C4（可靠性与不确定性）的
  A1 个体内变化、A2 帧级重复性、A6 误差结构、V6 保形区间原先基于俯视深度图，需要用高度图预测更新。

做法（方法完全复用原脚本，只换输入）
  - A1 / A2 / A6：import analysis_a1_a2_a6 的 a1 / a2 / a6 函数；把 runs/v1_tabular/preds.csv 的 vision_resnet18 列
    替换为高度图预测。A2 帧级预测取 **seed 42 单模型**（与原 A2 的单模型逐帧预测可比）；猪级 A1 / A6 用 3 种子平均。
  - V6：import v6_conformal_triage 的 conf_q / wilson / auroc；测试折的校准分数 = 其余 4 折的 OOF 绝对残差；
    marginal 与 Mondrian（品系 × 胎次 4 组）区间，α = 0.10 / 0.20；覆盖率按子集报告。另报帧间 SD（seed 42）与
    |误差| 的 Spearman、|误差| > 3 mm 的 AUROC（复测分流能否成立）。
  - DK：用丰美禾 OOF 残差分位数，作用于 v4d 的 DK 高度图预测，未校准 vs 每批 5 / 10 头截距校准（200 次）后的覆盖率。

输入：runs/v4_hmap/hmap/{test_preds.csv, frame_preds.csv}、runs/v1_tabular/preds.csv、runs/v4_hmap/dk_external/preds.csv
输出：runs/v4_hmap/reliability/{log.txt, result.json, a1_pairs.csv, a2_k_curve.csv}
用法（需要 pandas + statsmodels）：D:\\Anaconda\\python.exe v4g_hmap_reliability.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
import analysis_a1_a2_a6 as AA  # noqa: E402
from v6_conformal_triage import auroc, conf_q, wilson  # noqa: E402

HM = os.path.join(RUNS_ROOT, 'v4_hmap', 'hmap')
DKP = os.path.join(RUNS_ROOT, 'v4_hmap', 'dk_external', 'preds.csv')
OUT = os.path.join(RUNS_ROOT, 'v4_hmap', 'reliability')
ALPHAS = (0.10, 0.20)


def main():
    os.makedirs(OUT, exist_ok=True)
    AA.OUT = OUT  # a1 / a2 把 a1_pairs.csv、a2_k_curve.csv 写到这里
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] reliability on height-map ResNet-18 OOF predictions')
    df = pd.read_csv(AA.PIG, encoding='utf-8-sig')
    hm = pd.read_csv(os.path.join(HM, 'test_preds.csv'), encoding='utf-8-sig').set_index('uid')
    assert set(df['uid']) == set(hm.index)
    assert np.allclose(df['y'].values, hm.loc[df['uid'], 'y'].values)
    df['vision_resnet18'] = hm.loc[df['uid'], 'pred_mean'].values   # 替换为高度图 3 种子平均
    fr = pd.read_csv(os.path.join(HM, 'frame_preds.csv'), encoding='utf-8-sig')
    fr = fr[fr['seed'] == 42][['uid', 'group', 'fold', 'frame_idx', 'y', 'pred']]
    log(f'   pigs {len(df)}; seed-42 frames {len(fr)} (frames/pig median {fr.groupby("uid").size().median():.0f})')
    res = dict(A1=AA.a1(df, log), A2=AA.a2(fr, log), A6=AA.a6(df, log))

    # ---- V6 保形区间 ----
    y = df['y'].values
    p = df['vision_resnet18'].values
    fold = df['fold'].values
    line = df['line'].values
    parity = df['parity'].values
    prev = df['prev_bf'].values
    r = np.abs(y - p)
    key = np.array([f'{l}_{"p1" if q == 1 else "multi"}' for l, q in zip(line, parity)])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    v6 = {}
    log('\n=== V6 conformal intervals (height map) ===')
    for a in ALPHAS:
        half = {'marginal': np.zeros(len(y)), 'mondrian': np.zeros(len(y))}
        for k in sorted(set(fold)):
            te, ca = fold == k, fold != k
            q = conf_q(r[ca], a)
            half['marginal'][te] = q
            for g in set(key[te]):
                cg = ca & (key == g)
                qg = conf_q(r[cg], a) if cg.sum() >= int(np.ceil(1 / a)) else q
                half['mondrian'][te & (key == g)] = qg if np.isfinite(qg) else q
        v6[a] = {}
        for meth, h in half.items():
            cov = r <= h
            v6[a][meth] = {s: dict(n=int(m.sum()), coverage=float(cov[m].mean()),
                                   cov_ci=wilson(int(cov[m].sum()), int(m.sum())), width_mean=float(2 * h[m].mean()))
                           for s, m in subsets.items()}
            log(f'   α={a} {meth:9s} ' + ' | '.join(
                f'{s} {v["coverage"]:.3f} [{v["cov_ci"][0]:.2f},{v["cov_ci"][1]:.2f}] w {v["width_mean"]:.2f}'
                for s, v in v6[a][meth].items()))
    fsd = fr.groupby('uid')['pred'].std().reindex(df['uid']).values
    rq = dict(spearman_abs_err=float(spearmanr(fsd, r)[0]), auroc_err_gt3=auroc(fsd, r > 3))
    log(f'   triage signal: frame SD (seed 42) vs |error|: Spearman {rq["spearman_abs_err"]:.3f}, '
        f'AUROC(|e|>3) {rq["auroc_err_gt3"]:.3f}')
    res['V6'] = dict(intervals={str(k): v for k, v in v6.items()}, triage_frame_sd=rq)

    # ---- DK 覆盖率 ----
    if os.path.isfile(DKP):
        with open(DKP, encoding='utf-8-sig') as f:
            dk = list(csv.DictReader(f))
        dy = np.array([float(x['y']) for x in dk])
        dp = np.array([float(x['hmap_mean3']) for x in dk])
        db = np.array([x['batch'] for x in dk])
        bs = sorted(set(db))
        res['DK'] = {}
        log('\n=== DK coverage with Farm A residual quantile (height map) ===')
        for a in ALPHAS:
            q = conf_q(r, a)
            c0 = np.abs(dy - dp) <= q
            out = dict(q_fmh=q, uncalibrated=float(c0.mean()),
                       uncal_per_batch={b: float(c0[db == b].mean()) for b in bs})
            rng = np.random.default_rng(0)
            for kk in (5, 10):
                covs = []
                for _ in range(200):
                    cal = np.concatenate([rng.choice(np.where(db == b)[0], kk, replace=False) for b in bs])
                    rest = np.setdiff1d(np.arange(len(dy)), cal)
                    pc = dp.copy()
                    for b in bs:
                        cb = cal[db[cal] == b]
                        pc[db == b] += np.mean(dy[cb] - dp[cb])
                    covs.append(np.mean(np.abs(dy[rest] - pc[rest]) <= q))
                out[f'per_batch_k{kk}'] = (float(np.mean(covs)), float(np.std(covs)))
            res['DK'][a] = out
            log(f'   α={a}: q ±{q:.2f} mm | uncalibrated {out["uncalibrated"]:.3f} ('
                + ', '.join(f'{b} {v:.2f}' for b, v in out['uncal_per_batch'].items()) + ')'
                + f' | per-batch k=5 {out["per_batch_k5"][0]:.3f}±{out["per_batch_k5"][1]:.3f}'
                + f' | k=10 {out["per_batch_k10"][0]:.3f}±{out["per_batch_k10"][1]:.3f}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
