# -*- coding: utf-8 -*-
"""
analysis_a1_a2_a6.py — 贡献 C4 的三项分析：A1 个体内背膘变化追踪 / A2 帧级重复性 / A6 误差结构审计

来源 / 背景
  05 §8.2、06 §3。全部基于已有的猪级交叉验证预测（不训练新模型）：
    - runs/v1_tabular/preds.csv            猪级：视觉 ResNet-18（labelfix_new_s128）+ 表格模型 OOF 预测
    - runs/framepreds_new_s128/frame_preds.csv  逐帧：同一 ResNet-18 设置重跑（猪级数值与旧运行一致）

思路
  A1  102 个跨季重复个体（8 月季 → 10–11 月季），两季必在同一测试折（按 group_id 分组）。
      Δ = 晚季 − 早季；比较 Δ视觉 / Δ表格(T1_xgb) 与 Δ超声：Pearson、Spearman、CCC、Δ的 MAE、
      变化方向符合率（|Δ超声| ≥ 2 mm 的个体）；对照「预测不变（Δ=0）」。bootstrap 95% CI。
  A2  同一次采集内各帧预测的一致性：单因素随机效应 ICC(1)（不等帧数用 k0 校正）、
      Spearman–Brown 求达到信度 0.90 / 0.95 所需帧数；经验曲线：每头猪随机取 k 帧求均值（200 次）的 MAE / R²。
      注意：这是「同次采集内」重复性，不含跨日、重新上栏的变异。
  A6  猪级残差 r = 预测 − 超声：
      (a) r ~ 品系 + 胎次组 + 妊娠日龄(+缺失) + 季        —— 各群体的平均偏差
      (b) r ~ 超声背膘 + 品系 + 胎次组 + 妊娠日龄 + 季   —— 「同样真实背膘下」是否因品系/胎次系统性高估或低估
      Type II ANOVA；视觉与表格(T1_xgb)各做一次。另报校准斜率（预测对超声回归）。

与旧脚本的区别：新脚本；只读已有预测文件；输出 runs/analysis_a1a2a6/。

用法（需要 pandas + statsmodels）
  D:\\Anaconda\\python.exe analysis_a1_a2_a6.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.anova import anova_lm

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
PIG = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
FRAME = os.path.join(HERE, 'runs', 'framepreds_new_s128', 'frame_preds.csv')
OUT = os.path.join(HERE, 'runs', 'analysis_a1a2a6')
RNG = np.random.default_rng(0)


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 2 * np.cov(a, b, bias=True)[0, 1] / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)


def boot_ci(fn, *arrs, n=2000):
    m = len(arrs[0])
    vals = []
    for _ in range(n):
        i = RNG.integers(0, m, m)
        vals.append(fn(*[x[i] for x in arrs]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ---------------- A1 ----------------
def a1(df, log):
    df = df.copy()
    df['season'] = np.where(df['batch'].astype(str).str.split('.').str[0].astype(int) == 8, 'early', 'late')
    pairs = []
    for g, d in df.groupby('group'):
        if len(d) != 2 or set(d['season']) != {'early', 'late'}:
            continue
        e, l_ = d[d.season == 'early'].iloc[0], d[d.season == 'late'].iloc[0]
        pairs.append(dict(group=g, line=e['line'], parity_early=e['parity'], parity_late=l_['parity'],
                          fold_same=int(e['fold'] == l_['fold']),
                          y_early=e['y'], y_late=l_['y'], dy=l_['y'] - e['y'],
                          dvis=l_['vision_resnet18'] - e['vision_resnet18'],
                          dtab=l_['T1_xgb'] - e['T1_xgb']))
    p = pd.DataFrame(pairs)
    p.to_csv(os.path.join(OUT, 'a1_pairs.csv'), index=False, encoding='utf-8-sig')
    log(f'\n=== A1 within-animal change: {len(p)} pairs (same test fold: {p.fold_same.sum()}) ===')
    log(f'Δultrasound mean {p.dy.mean():+.2f} SD {p.dy.std():.2f} mm; |Δ|≥2 mm: {(p.dy.abs() >= 2).sum()}')
    log(f'parity early→late: {p.groupby(["parity_early", "parity_late"]).size().to_dict()}; '
        f'line: {p.line.value_counts().to_dict()}')
    res = dict(n=len(p), dy_mean=p.dy.mean(), dy_sd=p.dy.std())
    big = p.dy.abs() >= 2
    for name, col in (('vision', 'dvis'), ('tabular_T1_xgb', 'dtab')):
        x, y = p[col].values, p.dy.values
        r, pr = pearsonr(x, y)
        rs = spearmanr(x, y)[0]
        sign = float(np.mean(np.sign(x[big]) == np.sign(y[big])))
        r_ci = boot_ci(lambda a, b: pearsonr(a, b)[0], x, y)
        res[name] = dict(pearson=r, pearson_p=pr, pearson_ci=r_ci, spearman=rs, ccc=ccc(x, y),
                         mae_delta=float(np.mean(np.abs(x - y))), sign_agree_big=sign, n_big=int(big.sum()),
                         pred_delta_sd=float(np.std(x, ddof=1)))
        log(f'{name:15s} r={r:.3f} (95% CI {r_ci[0]:.3f}, {r_ci[1]:.3f}; p={pr:.2g})  Spearman={rs:.3f}  '
            f'CCC={ccc(x, y):.3f}  MAE(Δ)={np.mean(np.abs(x - y)):.2f} mm  '
            f'sign agree (|Δ|≥2 mm, n={big.sum()}) = {sign:.2%}  SD(Δpred)={np.std(x, ddof=1):.2f}')
    res['no_change_mae_delta'] = float(np.mean(np.abs(p.dy)))
    log(f'reference "predict no change": MAE(Δ)={np.mean(np.abs(p.dy)):.2f} mm')
    # v1 补充（2026-10-10）：方向符合率需与「群体趋势」基础比例对照；「人人取群体平均 Δ」参照（留一法均值）
    base_up = float(np.mean(p.dy[big] > 0))
    loo = (p.dy.sum() - p.dy) / (len(p) - 1)
    res['sign_base_rate_up_big'] = base_up
    res['mean_change_loo_mae_delta'] = float(np.mean(np.abs(p.dy - loo)))
    r_ee, p_ee = pearsonr(p.y_early, p.y_late)
    res['corr_early_late_ultrasound'] = dict(r=float(r_ee), p=float(p_ee))
    log(f'base rate: {base_up:.2%} of |Δ|≥2 mm pairs increased → sign agreement above is mostly the population trend')
    log(f'reference "everyone gets the mean Δ" (leave-one-out): MAE(Δ)={res["mean_change_loo_mae_delta"]:.2f} mm')
    log(f'ultrasound early vs late (same animal): r={r_ee:.3f} (p={p_ee:.2g})')
    return res


# ---------------- A2 ----------------
def icc1(f):
    """单因素随机效应 ICC(1)，不等组大小用 k0。f: DataFrame[uid, pred]"""
    g = f.groupby('uid')['pred']
    ni, means = g.size().values, g.mean().values
    N, a = ni.sum(), len(ni)
    grand = f['pred'].mean()
    ssb = np.sum(ni * (means - grand) ** 2)
    ssw = np.sum((f['pred'] - f['uid'].map(g.mean())) ** 2)
    msb, msw = ssb / (a - 1), ssw / (N - a)
    k0 = (N - np.sum(ni ** 2) / N) / (a - 1)
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return float(icc), float(np.sqrt(msw)), float(np.sqrt(max((msb - msw) / k0, 0))), float(np.mean(ni))


def a2(fr, log):
    log(f'\n=== A2 frame-level repeatability: {fr.uid.nunique()} pigs, {len(fr)} frames ===')
    icc, sd_w, sd_b, kbar = icc1(fr)
    sb = lambda k: k * icc / (1 + (k - 1) * icc)
    need = lambda R: R * (1 - icc) / (icc * (1 - R))
    res = dict(ICC1=icc, within_pig_frame_SD=sd_w, between_pig_SD=sd_b, mean_frames=kbar,
               reliability_of_mean_kbar=sb(kbar), frames_for_090=need(0.90), frames_for_095=need(0.95))
    per_fold = {int(k): icc1(d)[0] for k, d in fr.groupby('fold')}
    res['ICC1_per_fold'] = per_fold
    log(f'ICC(1) single frame = {icc:.3f}; within-pig frame SD = {sd_w:.2f} mm; between-pig SD = {sd_b:.2f} mm')
    log(f'reliability of mean of {kbar:.1f} frames (Spearman–Brown) = {sb(kbar):.3f}; '
        f'frames needed for 0.90 = {need(0.90):.1f}, for 0.95 = {need(0.95):.1f}')
    log(f'ICC per fold: { {k: round(v, 3) for k, v in per_fold.items()} }')
    # 经验曲线
    y = fr.groupby('uid')['y'].first()
    groups = {u: d['pred'].values for u, d in fr.groupby('uid')}
    uids = y.index.values
    rows = []
    for k in (1, 2, 3, 5, 10, 15, 20):
        maes, r2s = [], []
        for _ in range(200):
            pk = np.array([RNG.choice(groups[u], size=min(k, len(groups[u])), replace=False).mean() for u in uids])
            maes.append(np.mean(np.abs(pk - y.values)))
            r2s.append(1 - np.sum((y.values - pk) ** 2) / np.sum((y.values - y.values.mean()) ** 2))
        rows.append(dict(k=k, MAE_mean=np.mean(maes), MAE_sd=np.std(maes), R2_mean=np.mean(r2s), R2_sd=np.std(r2s)))
        log(f'  k={k:2d} frames: MAE {np.mean(maes):.3f} ± {np.std(maes):.3f}   R² {np.mean(r2s):.3f} ± {np.std(r2s):.3f}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'a2_k_curve.csv'), index=False, encoding='utf-8-sig')
    res['k_curve'] = rows
    return res


# ---------------- A6 ----------------
def a6(df, log):
    d = df.copy()
    d['season'] = np.where(d['batch'].astype(str).str.split('.').str[0].astype(int) == 8, 'Aug', 'OctNov')
    d['parity_grp'] = np.where(d['parity'] >= 2, 'multi', 'p1')
    d['gd_missing'] = d['gest_day'].isna().astype(int)
    d['gd'] = (d['gest_day'].fillna(d['gest_day'].median()) - 80) / 30
    res = {}
    for name, col in (('vision', 'vision_resnet18'), ('tabular_T1_xgb', 'T1_xgb')):
        d['r'] = d[col] - d['y']
        slope = smf.ols(f'{col} ~ y', d).fit().params['y']
        log(f'\n=== A6 error structure: {name} (calibration slope pred~ultrasound = {slope:.3f}) ===')
        out = dict(calibration_slope=float(slope))
        for tag, f in (('a_marginal', 'r ~ C(line) + C(parity_grp) + gd + gd_missing + C(season)'),
                       ('b_conditional_on_y', 'r ~ y + C(line) + C(parity_grp) + gd + gd_missing + C(season)')):
            m = smf.ols(f, d).fit()
            an = anova_lm(m, typ=2)
            log(f'-- ({tag}) {f}   R²={m.rsquared:.3f}')
            for term in an.index[:-1]:
                log(f'   {term:18s} F={an.loc[term, "F"]:7.2f}  p={an.loc[term, "PR(>F)"]:.3g}')
            for p_ in m.params.index:
                if p_ != 'Intercept':
                    ci = m.conf_int().loc[p_]
                    log(f'     coef {p_:28s} {m.params[p_]:+.3f} (95% CI {ci[0]:+.3f}, {ci[1]:+.3f})')
            out[tag] = dict(formula=f, r2=float(m.rsquared),
                            anova={t: dict(F=float(an.loc[t, 'F']), p=float(an.loc[t, 'PR(>F)']))
                                   for t in an.index[:-1]},
                            coef={p_: float(m.params[p_]) for p_ in m.params.index})
        strata = d.groupby(['line', 'parity_grp'])['r'].agg(['count', 'mean', 'std'])
        log('   mean residual by line × parity:\n' + strata.round(3).to_string())
        out['strata'] = {f'{a}_{b}': dict(n=int(r['count']), mean=float(r['mean']), sd=float(r['std']))
                         for (a, b), r in strata.iterrows()}
        res[name] = out
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    log(f'[{time.strftime("%F %T")}] inputs: {PIG} | {FRAME}')
    df = pd.read_csv(PIG, encoding='utf-8-sig')
    fr = pd.read_csv(FRAME, encoding='utf-8-sig')
    # 一致性核对：逐帧均值应等于猪级视觉预测（同一设置重跑，数值相同）
    chk = fr.groupby('uid')['pred'].mean().reindex(df['uid']).values - df['vision_resnet18'].values
    log(f'check frame-mean vs pig-level vision pred: max |diff| = {np.nanmax(np.abs(chk)):.3f} mm')
    res = dict(A1=a1(df, log), A2=a2(fr, log), A6=a6(df, log))
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
