# -*- coding: utf-8 -*-
"""
v6c_reliability_single_dataset.py — 本文新范围下的误差结构与相机单独保形区间 / 复测分流（N3，不训练）

来源 / 背景
  DECISIONS 2026-10-14：本文不报告跨季结果。v4g 的误差结构（A6）模型含测定季项；v4i 的 σ̂ 含「有无上季记录」。
  本脚本去掉这两个与跨季有关的变量重算，其余与 v4g / v4i 完全相同（09_研究成果总表 §4 N3）。

做法
  1. 误差结构（同 analysis_a1_a2_a6.a6，去掉 C(season)）：残差 r = 预测 − B 超；
     (a) r ~ 品系 + 胎次组 + 妊娠日龄 + 日龄缺失；(b) 再加 B 超真值 y（条件偏差）；Type II ANOVA；校准斜率 pred ~ y
  2. 保形区间（同 v4i，模型只取 cnn = v4b 高度图 ResNet-18 3 种子平均）：测试折校准分数 = 其余 4 折 OOF |残差|；
     marginal / Mondrian（品系 × 胎次 4 组）/ normalized（σ̂ 特征：帧间 SD（seed 42）、种子间 SD、ŷ、新美系、经产；
     **不含**有无上季记录），α = 0.10 / 0.20；子集：全体、丹系头胎、新美系经产
  3. 复测分流：按 σ̂、帧间 SD、种子间 SD、Mondrian 宽度排序 vs 随机（500 次）与 oracle；σ̂ − 随机 group bootstrap 2000 次

输入：runs/v1_tabular/preds.csv、runs/v4_hmap/hmap/{test_preds.csv, frame_preds.csv}
输出：runs/v6_reliability/{log.txt, result.json, triage_curve.csv}
用法（需要 pandas + statsmodels）：D:\\Anaconda\\python.exe v6c_reliability_single_dataset.py
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.anova import anova_lm

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v6_conformal_triage import ALPHAS, ERR_THR, FRACS, N_BOOT, auroc, conf_q, fit_sigma, triage, wilson  # noqa: E402

PIG = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
HM = os.path.join(HERE, 'runs', 'v4_hmap', 'hmap')
OUT = os.path.join(HERE, 'runs', 'v6_reliability')


def error_structure(df, log):
    d = df.copy()
    d['parity_grp'] = np.where(d['parity'] >= 2, 'multi', 'p1')
    d['gd_missing'] = d['gest_day'].isna().astype(int)
    d['gd'] = (d['gest_day'].fillna(d['gest_day'].median()) - 80) / 30
    d['r'] = d['pred'] - d['y']
    slope = smf.ols('pred ~ y', d).fit().params['y']
    log(f'\n=== error structure, height-map CNN (calibration slope pred~ultrasound = {slope:.3f}); no season term ===')
    out = dict(calibration_slope=float(slope))
    for tag, f in (('a_marginal', 'r ~ C(line) + C(parity_grp) + gd + gd_missing'),
                   ('b_conditional_on_y', 'r ~ y + C(line) + C(parity_grp) + gd + gd_missing')):
        m = smf.ols(f, d).fit()
        an = anova_lm(m, typ=2)
        log(f'-- ({tag}) {f}   R²={m.rsquared:.3f}')
        for term in an.index[:-1]:
            log(f'   {term:18s} F={an.loc[term, "F"]:7.2f}  p={an.loc[term, "PR(>F)"]:.3g}')
        ci = m.conf_int()
        for p_ in m.params.index:
            if p_ != 'Intercept':
                log(f'     coef {p_:28s} {m.params[p_]:+.3f} (95% CI {ci.loc[p_, 0]:+.3f}, {ci.loc[p_, 1]:+.3f})'
                    f'  p={m.pvalues[p_]:.3g}')
        out[tag] = dict(formula=f, r2=float(m.rsquared),
                        anova={t: dict(F=float(an.loc[t, 'F']), p=float(an.loc[t, 'PR(>F)'])) for t in an.index[:-1]},
                        coef={p_: [float(m.params[p_]), float(ci.loc[p_, 0]), float(ci.loc[p_, 1]),
                                   float(m.pvalues[p_])] for p_ in m.params.index})
    strata = d.groupby(['line', 'parity_grp'])['r'].agg(['count', 'mean', 'std'])
    log('   mean residual by line × parity:\n' + strata.round(3).to_string())
    out['strata'] = {f'{a}_{b}': dict(n=int(r['count']), mean=float(r['mean']), sd=float(r['std']))
                     for (a, b), r in strata.iterrows()}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-csv', default=os.path.join(HM, 'test_preds.csv'))
    ap.add_argument('--seed-cols', nargs='+', default=['pred_s42', 'pred_s43', 'pred_s44'])
    ap.add_argument('--mean-col', default='pred_mean')
    ap.add_argument('--frame-csv', default=os.path.join(HM, 'frame_preds.csv'))
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--icc', action='store_true', help='另算帧级重复性（analysis_a1_a2_a6.a2）')
    a = ap.parse_args()
    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] v6c error structure + camera-only conformal / triage without season variables')
    log(f'   pred {a.pred_csv} ({a.mean_col}; seed SD over {a.seed_cols}) | frames {a.frame_csv}')
    df = pd.read_csv(PIG, encoding='utf-8-sig')
    hm = pd.read_csv(a.pred_csv, encoding='utf-8-sig').set_index('uid')
    assert set(df['uid']) == set(hm.index)
    assert np.allclose(df['y'].values, hm.loc[df['uid'], 'y'].values)
    assert np.array_equal(df['fold'].values, hm.loc[df['uid'], 'fold'].values)
    df['pred'] = hm.loc[df['uid'], a.mean_col].values
    seed_preds = hm.loc[df['uid'], a.seed_cols].values
    fr_all = pd.read_csv(a.frame_csv, encoding='utf-8-sig')
    fr_all = fr_all[fr_all['seed'] == 42]
    if a.icc:
        import analysis_a1_a2_a6 as AA
        AA.OUT = out_dir   # a2 把 a2_k_curve.csv 写到这里
        icc_res = AA.a2(fr_all[['uid', 'group', 'fold', 'frame_idx', 'y', 'pred']], log)
    fr = fr_all.groupby('uid')['pred']
    frame_sd = fr.std(ddof=1).reindex(df['uid']).fillna(0).values
    n_frames = fr.size().reindex(df['uid']).values
    log(f'   pigs {len(df)}; frames/pig (seed 42) median {np.median(n_frames):.0f} ({n_frames.min()}–{n_frames.max()})')

    result = dict(error_structure=error_structure(df, log))
    if a.icc:
        result["icc"] = icc_res

    y, p = df['y'].values, df['pred'].values
    r = y - p
    fold, group = df['fold'].values, df['group'].values
    line, parity = df['line'].values, df['parity'].values
    ens_sd = seed_preds.std(1, ddof=1)
    is_nam, multi = (line == 'NAM').astype(float), (parity >= 2).astype(float)
    mondrian_key = np.array([f'{l}_{"p1" if q == 1 else "multi"}' for l, q in zip(line, parity)])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2)}
    U = np.column_stack([frame_sd, ens_sd, p, is_nam, multi])
    sd = U.std(0)
    U = (U - U.mean(0)) / np.where(sd > 0, sd, 1)
    folds = sorted(set(fold))
    log(f'\n=== conformal intervals, height-map CNN alone (MAE {np.mean(np.abs(r)):.3f}); '
        f'σ̂ features: frame SD, seed SD, ŷ, NAM, multiparous (no previous-season indicator)')
    sigma_test = np.zeros(len(y))
    result['conformal'] = {}
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
        log(f'   α={a} (target {1 - a:.0%}): coverage [Wilson 95% CI] mean width (mm)')
        for meth in lo:
            cov = (y >= lo[meth]) & (y <= hi[meth])
            w = hi[meth] - lo[meth]
            res_a[meth] = {}
            for s, mk in subsets.items():
                kc, nn = int(cov[mk].sum()), int(mk.sum())
                res_a[meth][s] = dict(n=nn, coverage=kc / nn, cov_ci=wilson(kc, nn),
                                      width_mean=float(w[mk].mean()), width_median=float(np.median(w[mk])))
            log(f'     {meth:10s} ' + ' | '.join(
                f'{s} {res_a[meth][s]["coverage"]:.3f} [{res_a[meth][s]["cov_ci"][0]:.2f},'
                f'{res_a[meth][s]["cov_ci"][1]:.2f}] w {res_a[meth][s]["width_mean"]:.2f}' for s in subsets))
        result['conformal'][f'alpha={a}'] = res_a

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
    curve_rows = []
    for f in FRACS:
        row = {'model': 'cnn', 'frac': f}
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
        tri['sigma_vs_random'][f] = dict(d_mae_kept=[float(x) for x in np.percentile(v[:, 0], [2.5, 50, 97.5])],
                                         d_gt3_kept=[float(x) for x in np.percentile(v[:, 1], [2.5, 50, 97.5])])
    result['triage'] = tri
    log('\n=== retest triage, height-map CNN alone')
    log('   ranking quality: ' + ' | '.join(f'{k} ρ {v["spearman_abs_err"]:.3f} AUROC(|e|>3) {v["auroc_err_gt3"]:.3f}'
                                         for k, v in tri['ranking_quality'].items()))
    log('     frac  random        sigma_hat     frame_sd      seed_sd       mondrian      oracle')
    for f in FRACS:
        c = tri['curve'][f]
        log(f'     {f:4.2f}  ' + '  '.join(f'{c[f"{k}_mae_kept"]:.3f}/{c[f"{k}_gt3_kept"]:.3f}' for k in
                                         ('random', 'sigma_hat', 'frame_sd', 'seed_sd', 'mondrian_width', 'oracle')))
    for f, v in tri['sigma_vs_random'].items():
        log(f'     σ̂ − random @ {f:.0%}: ΔMAE(kept) {v["d_mae_kept"][1]:+.3f} ({v["d_mae_kept"][0]:+.3f}, '
            f'{v["d_mae_kept"][2]:+.3f}); Δshare|e|>3 {v["d_gt3_kept"][1]:+.3f} ({v["d_gt3_kept"][0]:+.3f}, '
            f'{v["d_gt3_kept"][2]:+.3f})')

    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(out_dir, 'triage_curve.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader()
        for row in curve_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    log(f'done in {time.time() - t0:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
