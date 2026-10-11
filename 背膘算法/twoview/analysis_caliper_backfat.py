# -*- coding: utf-8 -*-
"""
analysis_caliper_backfat.py — 体况卡尺评分与 P2 背膘的关系（描述性），以及相机与卡尺的信息重叠

来源 / 背景
  本人 2026-10-11 问「体况评分和背膘的关系」。卡尺读数来自 v1b_caliper_vs_vision.py 的解析
  （runs/v1b_caliper/preds.csv：caliper 列；含斜杠取后段，纯整数 4/5 也是读数）。

内容
  1. 卡尺 ~ 背膘：总体、品系、品系 × 胎次、季节的 Pearson / Spearman
  2. 关系是否因品系 / 胎次不同：OLS 背膘 ~ 卡尺 × 品系 + 胎次组 + 季（Type II ANOVA；交互项 = 斜率差异）
  3. 是否线性：按卡尺读数分档的背膘均值 ± SD
  4. 相机与卡尺的信息重叠：r(相机预测, 卡尺)；偏相关 r(相机, 背膘 | 卡尺) 与 r(卡尺, 背膘 | 相机)
     （相机用交叉验证的猪级 OOF 预测，不含训练泄漏）

与旧脚本的区别：新脚本，只读已有文件；输出 runs/analysis_caliper/。

用法：D:\\Anaconda\\python.exe analysis_caliper_backfat.py
"""
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
SRC = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
OUT = os.path.join(HERE, 'runs', 'analysis_caliper')


def partial_r(x, y, z):
    """r(x, y | z)：x、y 各对 z 线性回归后残差的相关。"""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return pearsonr(rx, ry)


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    d = pd.read_csv(SRC, encoding='utf-8-sig')
    v1 = pd.read_csv(V1, encoding='utf-8-sig')[['uid', 'batch', 'gest_day']]
    d = d.merge(v1, on='uid', how='left')
    d = d[d.caliper.notna()].copy()
    d['season'] = np.where(d['batch'].astype(str).str.split('.').str[0].astype(int) == 8, 'Aug', 'OctNov')
    d['parity_grp'] = np.where(d['parity'] >= 2, 'multi', 'p1')
    log(f'[{time.strftime("%F %T")}] n={len(d)} pigs with caliper; caliper mean {d.caliper.mean():.2f} '
        f'SD {d.caliper.std():.2f} range {d.caliper.min():.1f}–{d.caliper.max():.1f}')

    log('\n1. caliper vs backfat correlation')

    def corr_row(name, g):
        r, p = pearsonr(g.caliper, g.y)
        rs = spearmanr(g.caliper, g.y)[0]
        b = np.polyfit(g.caliper, g.y, 1)[0]
        log(f'   {name:22s} n={len(g):4d}  r={r:.3f} (p={p:.2g})  Spearman={rs:.3f}  slope={b:.2f} mm/unit  '
            f'caliper {g.caliper.mean():.2f}±{g.caliper.std():.2f}  backfat {g.y.mean():.1f}±{g.y.std():.1f}')
    corr_row('all', d)
    for k, g in d.groupby('line'):
        corr_row(f'line={k}', g)
    for (a, b), g in d.groupby(['line', 'parity_grp']):
        corr_row(f'{a} {b}', g)
    for k, g in d.groupby('season'):
        corr_row(f'season={k}', g)

    log('\n2. does the caliper–backfat relation differ by line / parity? (OLS, Type II)')
    for f in ('y ~ caliper * C(line) + C(parity_grp) + C(season)',
              'y ~ caliper * C(parity_grp) + C(line) + C(season)'):
        m = smf.ols(f, d).fit()
        an = anova_lm(m, typ=2)
        log(f'   {f}   R²={m.rsquared:.3f}')
        for t in an.index[:-1]:
            log(f'      {t:28s} F={an.loc[t, "F"]:7.2f}  p={an.loc[t, "PR(>F)"]:.3g}')
        for p_ in m.params.index:
            if p_ != 'Intercept':
                ci = m.conf_int().loc[p_]
                log(f'        coef {p_:32s} {m.params[p_]:+.3f} (95% CI {ci[0]:+.3f}, {ci[1]:+.3f})')

    log('\n3. backfat by caliper reading (linearity)')
    bins = [0, 3.0, 3.5, 3.9, 4.1, 4.3, 4.5, 4.7, 6]
    d['cbin'] = pd.cut(d.caliper, bins, right=False)
    t = d.groupby('cbin', observed=True).y.agg(['count', 'mean', 'std']).round(2)
    log(t.to_string())
    t.to_csv(os.path.join(OUT, 'backfat_by_caliper_bin.csv'), encoding='utf-8-sig')
    log(f'   distinct readings: {d.caliper.round(1).value_counts().sort_index().to_dict()}')

    log('\n4. information overlap between camera (OOF prediction) and caliper')
    x_v, x_c, y = d.vision_resnet18.values, d.caliper.values, d.y.values
    r_vc = pearsonr(x_v, x_c)
    pv = partial_r(x_v, y, x_c)
    pc = partial_r(x_c, y, x_v)
    log(f'   r(camera, caliper)        = {r_vc[0]:.3f} (p={r_vc[1]:.2g})')
    log(f'   r(camera, backfat)        = {pearsonr(x_v, y)[0]:.3f}')
    log(f'   r(caliper, backfat)       = {pearsonr(x_c, y)[0]:.3f}')
    log(f'   partial r(camera, backfat | caliper) = {pv[0]:.3f} (p={pv[1]:.2g})')
    log(f'   partial r(caliper, backfat | camera) = {pc[0]:.3f} (p={pc[1]:.2g})')
    for k, g in d.groupby('line'):
        log(f'   [{k}] r(camera,caliper)={pearsonr(g.vision_resnet18, g.caliper)[0]:.3f}  '
            f'partial camera|caliper={partial_r(g.vision_resnet18.values, g.y.values, g.caliper.values)[0]:.3f}  '
            f'partial caliper|camera={partial_r(g.caliper.values, g.y.values, g.vision_resnet18.values)[0]:.3f}')
    log(f'done -> {OUT}')


if __name__ == '__main__':
    main()
