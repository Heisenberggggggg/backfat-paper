# -*- coding: utf-8 -*-
"""
v4i_hmap_conformal_triage.py — 用高度图 ResNet-18（v4b / v4h / v4d）重算 V6：保形区间（marginal / Mondrian / normalized）
                               + B 超复测分流曲线 + DK 覆盖率（不训练，只用已有 OOF 预测）

来源 / 背景
  DECISIONS 2026-10-13（四）：主结果统一用高度图 CNN。v4g 已用高度图重算 marginal / Mondrian 区间与帧间 SD 的
  分流指标，但 normalized 区间与复测分流曲线（Fig 4d）仍是 V6 的旧模型（DINOv3 堆叠 + 裁剪深度图 ResNet）。
  本脚本把 V6 的完整流程换成高度图输入，使 Fig 4c/d 与 Table 3 使用同一个图像模型。

与 v6_conformal_triage.py 的区别（conf_q / wilson / fit_sigma / auroc / triage 直接 import，流程相同）
  - 模型：cnn（v4b 高度图 ResNet-18，3 种子平均，相机单独）、img（v4h：TabPFN 只用 CNN 分数）、R+img、R+cal、
    R+prev+cal+img（v4h 嵌套融合 OOF）
  - σ̂ 特征：帧间 SD（v4b seed 42 帧级 OOF）、**3 个种子间的预测 SD**（深度集成分歧，取代 V6 的
    |DINOv3 ridge − ResNet| 两模型分歧）、ŷ、新美系、经产、有上季记录
  - DK：v4d hmap_mean3，校准分位数取 FMH cnn 的 OOF 残差

输入：runs/v4_hmap/{hmap/test_preds.csv, hmap/frame_preds.csv, nested_fusion/preds.csv, dk_external/preds.csv}、
      runs/v1_tabular/preds.csv
输出（新目录）：runs/v4_hmap/conformal_triage/{log.txt, result.json, intervals.csv, triage_curve.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v4i_hmap_conformal_triage.py
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v6_conformal_triage import (ALPHAS, ERR_THR, FRACS, N_BOOT, auroc, conf_q, fit_sigma, fnum,  # noqa: E402
                                 read_csv, triage, wilson)

RUNS = os.path.join(HERE, 'runs')
SRC_CNN = os.path.join(RUNS, 'v4_hmap', 'hmap', 'test_preds.csv')
SRC_FRAME = os.path.join(RUNS, 'v4_hmap', 'hmap', 'frame_preds.csv')
SRC_NEST = os.path.join(RUNS, 'v4_hmap', 'nested_fusion', 'preds.csv')
SRC_TAB = os.path.join(RUNS, 'v1_tabular', 'preds.csv')
SRC_DK = os.path.join(RUNS, 'v4_hmap', 'dk_external', 'preds.csv')
OUT = os.path.join(RUNS, 'v4_hmap', 'conformal_triage')

MODELS = ('cnn', 'img', 'R+img', 'R+cal', 'R+prev+cal+img')
SEEDS = ('pred_s42', 'pred_s43', 'pred_s44')


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] v4i height-map conformal intervals + retest triage')
    log(f'   cnn   {SRC_CNN}\n   frame {SRC_FRAME}\n   nest  {SRC_NEST}\n   tab   {SRC_TAB}\n   dk    {SRC_DK}')

    rows = read_csv(SRC_NEST)
    cnn = {r['uid']: r for r in read_csv(SRC_CNN)}
    tab = {r['uid']: r for r in read_csv(SRC_TAB)}
    uid = np.array([r['uid'] for r in rows])
    assert all(u in cnn and u in tab for u in uid), 'nested / cnn / v1 uid 不一致'
    group = np.array([r['group'] for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    for src, name in ((cnn, 'cnn'), (tab, 'v1')):
        assert np.allclose(y, [float(src[u]['y']) for u in uid]), f'nested / {name} 标签不一致'
        assert np.array_equal(fold, [int(src[u]['fold']) for u in uid]), f'nested / {name} 折不一致'
    P = {m: np.array([float(r[m]) for r in rows]) for m in MODELS if m != 'cnn'}
    P['cnn'] = np.array([float(cnn[u]['pred_mean']) for u in uid])
    seed_preds = np.array([[float(cnn[u][s]) for s in SEEDS] for u in uid])
    ens_sd = seed_preds.std(1, ddof=1)
    line = np.array([tab[u]['line'] for u in uid])
    parity = np.array([fnum(tab[u]['parity']) for u in uid])
    prev = np.array([fnum(tab[u]['prev_bf']) for u in uid])

    fr = {}
    for r in read_csv(SRC_FRAME):
        if r['seed'] == '42':
            fr.setdefault(r['uid'], []).append(float(r['pred']))
    assert all(u in fr for u in uid), '帧级预测缺猪'
    frame_sd = np.array([np.std(fr[u], ddof=1) if len(fr[u]) > 1 else 0.0 for u in uid])
    n_frames = np.array([len(fr[u]) for u in uid])
    diff_mean = np.max(np.abs(np.array([np.mean(fr[u]) for u in uid]) - seed_preds[:, 0]))
    log(f'   n={len(y)} pigs, {len(set(group))} groups; frames/pig (seed 42, after QC) median {np.median(n_frames):.0f} '
        f'(range {n_frames.min()}–{n_frames.max()}); |mean(frame) − pred_s42| max {diff_mean:.3f}')

    is_nam = (line == 'NAM').astype(float)
    multi = (parity >= 2).astype(float)
    has_prev = (~np.isnan(prev)).astype(float)
    mondrian_key = np.array([f'{l}_{"p1" if p == 1 else "multi"}' for l, p in zip(line, parity)])
    subsets = {
        'all': np.ones(len(y), bool),
        'DAN_parity1': (line == 'DAN') & (parity == 1),
        'NAM_multiparous': (line == 'NAM') & (parity >= 2),
        'no_prev_season': has_prev == 0,
        'has_prev_season': has_prev == 1,
    }
    log('   subsets: ' + ', '.join(f'{k} {v.sum()}' for k, v in subsets.items()))
    log('   mondrian groups: ' + ', '.join(f'{k} {np.sum(mondrian_key == k)}' for k in sorted(set(mondrian_key))))
    log(f'   σ̂ features: frame_sd (median {np.median(frame_sd):.2f}), seed SD (median {np.median(ens_sd):.2f}), '
        f'ŷ, NAM, multiparous, has_prev')

    folds = sorted(set(fold))
    result = {'n': int(len(y)), 'folds': folds, 'fmh': {}, 'triage': {}, 'dk': {}}
    interval_rows = {u: {'uid': u, 'group': g, 'fold': int(k), 'y': float(t)} for u, g, k, t in
                     zip(uid, group, fold, y)}
    curve_rows = []

    for m in MODELS:
        p = P[m]
        r = y - p
        U = np.column_stack([frame_sd, ens_sd, p, is_nam, multi, has_prev])
        sd = U.std(0)
        U = (U - U.mean(0)) / np.where(sd > 0, sd, 1)

        sigma_test = np.zeros(len(y))
        res_m = {}
        for a in ALPHAS:
            lo = {meth: np.zeros(len(y)) for meth in ('marginal', 'mondrian', 'normalized')}
            hi = {meth: np.zeros(len(y)) for meth in lo}
            for k in folds:
                te, ca = fold == k, fold != k
                q = conf_q(np.abs(r[ca]), a)
                lo['marginal'][te], hi['marginal'][te] = p[te] - q, p[te] + q
                for g in set(mondrian_key[te]):
                    cg = ca & (mondrian_key == g)
                    qg = conf_q(np.abs(r[cg]), a) if cg.sum() >= math.ceil(1 / a) else q
                    if not np.isfinite(qg):
                        qg = q
                    tg = te & (mondrian_key == g)
                    lo['mondrian'][tg], hi['mondrian'][tg] = p[tg] - qg, p[tg] + qg
                floor = float(np.percentile(np.abs(r[ca]), 10))
                ca_idx = np.where(ca)[0]
                scores = np.zeros(len(ca_idx))
                for j in folds:
                    if j == k:
                        continue
                    fit_m = ca & (fold != j)
                    sj = fit_sigma(U[fit_m], np.abs(r[fit_m]), floor)
                    jm = fold[ca_idx] == j
                    scores[jm] = np.abs(r[ca_idx[jm]]) / sj(U[ca_idx[jm]])
                s_te = fit_sigma(U[ca], np.abs(r[ca]), floor)(U[te])
                qn = conf_q(scores, a)
                lo['normalized'][te], hi['normalized'][te] = p[te] - qn * s_te, p[te] + qn * s_te
                if a == ALPHAS[0]:
                    sigma_test[te] = s_te
            res_a = {}
            for meth in lo:
                cov = (y >= lo[meth]) & (y <= hi[meth])
                w = hi[meth] - lo[meth]
                res_a[meth] = {}
                for s, mk in subsets.items():
                    kc, nn = int(cov[mk].sum()), int(mk.sum())
                    res_a[meth][s] = dict(n=nn, coverage=kc / nn, cov_ci=wilson(kc, nn),
                                          width_mean=float(w[mk].mean()), width_median=float(np.median(w[mk])))
                if a == ALPHAS[0]:
                    for i, u in enumerate(uid):
                        interval_rows[u][f'{m}|pred'] = round(float(p[i]), 3)
                        interval_rows[u][f'{m}|{meth}_lo'] = round(float(lo[meth][i]), 3)
                        interval_rows[u][f'{m}|{meth}_hi'] = round(float(hi[meth][i]), 3)
            res_m[f'alpha={a}'] = res_a
        result['fmh'][m] = res_m

        log(f'\n== {m}  (MAE {np.mean(np.abs(r)):.3f}, R² {1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2):.3f})')
        for a in ALPHAS:
            log(f'   α={a} (target {1 - a:.0%}): coverage [Wilson 95% CI] mean width (mm)')
            for meth in ('marginal', 'mondrian', 'normalized'):
                ra = res_m[f'alpha={a}'][meth]
                log(f'     {meth:10s} ' + ' | '.join(
                    f'{s} {ra[s]["coverage"]:.3f} [{ra[s]["cov_ci"][0]:.2f},{ra[s]["cov_ci"][1]:.2f}] '
                    f'w {ra[s]["width_mean"]:.2f}' for s in subsets))

        e = np.abs(r)
        mond_w = np.zeros(len(y))
        for k in folds:
            te, ca = fold == k, fold != k
            for g in set(mondrian_key[te]):
                cg = ca & (mondrian_key == g)
                mond_w[te & (mondrian_key == g)] = conf_q(e[cg] if cg.sum() >= 10 else e[ca], ALPHAS[0])
        rankers = {'sigma_hat': sigma_test, 'frame_sd': frame_sd, 'seed_sd': ens_sd,
                   'mondrian_width': mond_w + 1e-6 * sigma_test, 'oracle': e}
        tri = {'ranking_quality': {}, 'curve': {}}
        for name, sc in rankers.items():
            if name != 'oracle':
                tri['ranking_quality'][name] = dict(spearman_abs_err=float(spearmanr(sc, e)[0]),
                                                    auroc_err_gt3=auroc(sc, e > ERR_THR))
        rng = np.random.default_rng(0)
        for f in FRACS:
            row = {'model': m, 'frac': f}
            rnd = np.array([triage(r, None, f, rng) for _ in range(500)]).mean(0)
            row.update({'random_mae_all': rnd[0], 'random_mae_kept': rnd[1], 'random_gt3_kept': rnd[2]})
            for name, sc in rankers.items():
                a_, b_, c_ = triage(r, sc, f)
                row.update({f'{name}_mae_all': a_, f'{name}_mae_kept': b_, f'{name}_gt3_kept': c_})
            curve_rows.append(row)
            tri['curve'][f] = row
        ug = np.unique(group)
        gidx = {g: np.where(group == g)[0] for g in ug}
        brng = np.random.default_rng(1)
        boot = {f: [] for f in (0.10, 0.20, 0.30)}
        for _ in range(N_BOOT):
            idx = np.concatenate([gidx[g] for g in brng.choice(ug, len(ug), replace=True)])
            eb, sb = r[idx], sigma_test[idx]
            for f in boot:
                _, s_kept, s_gt3 = triage(eb, sb, f)
                boot[f].append((s_kept - np.mean(np.abs(eb)), s_gt3 - np.mean(np.abs(eb) > ERR_THR)))
        tri['sigma_vs_random'] = {}
        for f, v in boot.items():
            v = np.array(v)
            tri['sigma_vs_random'][f] = dict(
                d_mae_kept=[float(x) for x in np.percentile(v[:, 0], [2.5, 50, 97.5])],
                d_gt3_kept=[float(x) for x in np.percentile(v[:, 1], [2.5, 50, 97.5])])
        result['triage'][m] = tri

        log('   ranking quality: ' + ' | '.join(
            f'{k} ρ {v["spearman_abs_err"]:.3f} AUROC(|e|>3) {v["auroc_err_gt3"]:.3f}'
            for k, v in tri['ranking_quality'].items()))
        log('   triage: retest frac → MAE of non-retested / share |e|>3 mm among non-retested')
        log('     frac  random        sigma_hat     frame_sd      seed_sd       mondrian      oracle')
        for f in FRACS:
            c = tri['curve'][f]
            log(f'     {f:4.2f}  ' + '  '.join(f'{c[f"{k}_mae_kept"]:.3f}/{c[f"{k}_gt3_kept"]:.3f}' for k in
                                             ('random', 'sigma_hat', 'frame_sd', 'seed_sd', 'mondrian_width',
                                              'oracle')))
        for f, v in tri['sigma_vs_random'].items():
            log(f'     σ̂ − random @ {f:.0%}: ΔMAE(kept) {v["d_mae_kept"][1]:+.3f} '
                f'({v["d_mae_kept"][0]:+.3f}, {v["d_mae_kept"][2]:+.3f}); '
                f'Δshare|e|>3 {v["d_gt3_kept"][1]:+.3f} ({v["d_gt3_kept"][0]:+.3f}, {v["d_gt3_kept"][2]:+.3f})')

    log('\n== DK external (FMH cnn OOF residual quantile; per-batch intercept calibration k pigs/batch, 200 draws)')
    dk = read_csv(SRC_DK)
    dy = np.array([float(r['y']) for r in dk])
    dbatch = np.array([r['batch'] for r in dk])
    batches = sorted(set(dbatch))
    dp = np.array([float(r['hmap_mean3']) for r in dk])
    e_fmh = np.abs(y - P['cnn'])
    for a in ALPHAS:
        q = conf_q(e_fmh, a)
        cov0 = np.abs(dy - dp) <= q
        out = {'fmh_model': 'cnn', 'q_fmh': q,
               'uncalibrated': dict(coverage=float(cov0.mean()), cov_ci=wilson(int(cov0.sum()), len(dy)),
                                    per_batch={b: float(cov0[dbatch == b].mean()) for b in batches})}
        rng = np.random.default_rng(0)
        for kk in (5, 10):
            covs, pb_covs = [], {b: [] for b in batches}
            for _ in range(200):
                cal = np.concatenate([rng.choice(np.where(dbatch == b)[0], kk, replace=False) for b in batches])
                rest = np.setdiff1d(np.arange(len(dy)), cal)
                pc = dp.copy()
                for b in batches:
                    cb = cal[dbatch[cal] == b]
                    pc[dbatch == b] += np.mean(dy[cb] - dp[cb])
                c = np.abs(dy[rest] - pc[rest]) <= q
                covs.append(c.mean())
                for b in batches:
                    pb_covs[b].append(c[dbatch[rest] == b].mean())
            out[f'per_batch_k{kk}'] = dict(coverage_mean=float(np.mean(covs)), coverage_sd=float(np.std(covs)),
                                           coverage_p5_p95=[float(x) for x in np.percentile(covs, [5, 95])],
                                           per_batch={b: float(np.mean(v)) for b, v in pb_covs.items()})
        result['dk'][f'alpha={a}'] = out
        log(f'   hmap_mean3 α={a}: q_FMH ±{q:.2f} mm | uncalibrated cov {out["uncalibrated"]["coverage"]:.3f} ('
            + ', '.join(f'{b} {v:.2f}' for b, v in out['uncalibrated']['per_batch'].items()) + ')'
            + ''.join(f' | per-batch k={kk}: cov {out[f"per_batch_k{kk}"]["coverage_mean"]:.3f}'
                      f'±{out[f"per_batch_k{kk}"]["coverage_sd"]:.3f} ('
                      + ', '.join(f'{b} {v:.2f}' for b, v in out[f'per_batch_k{kk}']['per_batch'].items()) + ')'
                      for kk in (5, 10)))

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    cols = list(next(iter(interval_rows.values())).keys())
    with open(os.path.join(OUT, 'intervals.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(interval_rows.values())
    with open(os.path.join(OUT, 'triage_curve.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        for row in curve_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    log(f'done in {time.time() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
