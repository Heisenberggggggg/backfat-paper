# -*- coding: utf-8 -*-
"""
v1_tabular_baseline.py — V1 纯表格基线（生产记录 → 背膘），与视觉模型同一套猪级 5 折

来源 / 背景
  05 §5 / 06 §5.1 第 1 项。论文贡献 C3 需要回答「已有生产记录时，图像还能多解释多少背膘变异」，
  必须先有只用生产记录的基线。场方确认（05 §2）：A/T = 丹系，LY = 新美系；品系与胎次高度混杂，
  因此除总体外还要报告品系内（丹系头胎、新美系经产、丹系经产）结果。

思路
  - 猪与折：完全复用 runs/labelfix_new_s128/test_preds.csv 的 663 头猪及其顺序，按 cv_depthmap_labelfix.py
    的规则（新标签五分位分层、StratifiedGroupKFold(5, seed=42)、折内 StratifiedGroupKFold(7, seed=42+k) 取验证集）
    重建折，并用其 baseline 列逐头核对，确保与视觉模型的划分完全一致。
  - 特征集
      T1 = 品系 + 胎次 + 妊娠日龄（缺失保留为 NaN / 线性模型用训练集中位数填补 + 缺失指示）
      T2 = T1 + 上季背膘（同一 group_id 在更早一季的有效背膘；第一季记录为缺失）+ 是否有上季记录
    BCS 列格式混杂（如 "5/4.2"），含义待本人确认，暂不使用。
  - 模型：训练集均值；岭回归；XGBoost（折内验证集早停）；TabPFN（若已安装，用训练+验证集拟合，无调参）
  - 指标：MAE / RMSE / R² / Lin CCC / Spearman，总体 + 品系内；与视觉（labelfix_new_s128）配对比较：
    猪级绝对误差 Wilcoxon 符号秩检验 + 按 group 的 bootstrap（2000 次）ΔR²、ΔMAE 95% CI。

与旧脚本的区别：新脚本，不改动任何已有文件；输出 runs/v1_tabular/。

用法
  D:\\Anaconda\\envs\\pytorch\\python.exe v1_tabular_baseline.py [--no-tabpfn]
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, 'index_v2.csv')
VISION = os.path.join(HERE, 'runs', 'labelfix_new_s128', 'test_preds.csv')
OUT = os.path.join(HERE, 'runs', 'v1_tabular')
BF_MIN, BF_MAX = 7.0, 32.0
LINE = {'A': 'DAN', 'T': 'DAN', 'LY': 'NAM'}   # 场方确认：A/T 丹系，LY 新美系


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def season_key(batch):
    """batch 形如 '8.24' / '10.29' / '11.3'；只需区分季：8 月为早季，10–11 月为晚季。"""
    return int(str(batch).split('.')[0])


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    ccc = 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    sp = spearmanr(y, p)[0] if np.std(p) > 0 else float('nan')
    return dict(n=len(y), MAE=mae, RMSE=rmse, R2=r2, CCC=ccc, Spearman=sp)


def load():
    idx = {}
    with open(INDEX, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['farm'] == 'FMH':
                idx[r['uid']] = r
    # 每个 group 的各季有效背膘（用于「上季背膘」）
    by_group = {}
    for r in idx.values():
        bf = fnum(r['backfat'])
        if BF_MIN <= bf <= BF_MAX and not int(r['label_conflict'] or 0):
            by_group.setdefault(r['group_id'], []).append((season_key(r['batch']), bf, r['uid']))
    rows = []
    with open(VISION, encoding='utf-8-sig') as f:
        for v in csv.DictReader(f):
            r = idx[v['uid']]
            s = season_key(r['batch'])
            prev = [bf for (ss, bf, u) in by_group.get(r['group_id'], []) if ss < s]
            rows.append(dict(uid=v['uid'], group=v['group'], y=float(v['y_new']), vis=float(v['pred']),
                             vis_base=float(v['baseline']), line=LINE[r['breed']], parity=fnum(r['parity']),
                             gest_day=fnum(r['gest_day']), batch=r['batch'],
                             prev_bf=prev[-1] if prev else np.nan))
    return rows


def design(rows, fs):
    """原始特征矩阵（含 NaN），供树模型 / TabPFN 使用。"""
    cols = ['is_NAM', 'parity', 'gest_day'] + (['prev_bf', 'has_prev'] if fs == 'T2' else [])
    X = []
    for r in rows:
        x = [1.0 if r['line'] == 'NAM' else 0.0, r['parity'], r['gest_day']]
        if fs == 'T2':
            x += [r['prev_bf'], 0.0 if np.isnan(r['prev_bf']) else 1.0]
        X.append(x)
    return np.array(X, float), cols


def linear_design(X, tr_mask, cols):
    """线性模型：NaN 用训练集中位数填补 + 缺失指示；胎次取「经产」指示；妊娠日龄加平方项。"""
    out = []
    for j, c in enumerate(cols):
        col = X[:, j]
        miss = np.isnan(col)
        med = np.nanmedian(col[tr_mask]) if np.any(~np.isnan(col[tr_mask])) else 0.0
        col = np.where(miss, med, col)
        if c == 'parity':
            out.append((col >= 2).astype(float))
        elif c == 'gest_day':
            z = (col - 80.0) / 30.0
            out += [z, z ** 2, miss.astype(float)]
        else:
            out.append(col)
    return np.column_stack(out)


def fit_xgb(X, y, tr, va, seed):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=3, min_child_weight=5,
                         subsample=0.8, reg_lambda=1.0, objective='reg:pseudohubererror', huber_slope=1.0,
                         early_stopping_rounds=100, random_state=seed, n_jobs=4, base_score=float(y[tr].mean()))
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    return m


def fit_tabpfn(X, y, trv, seed):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(random_state=seed)
    m.fit(X[trv], y[trv])
    return m


def group_bootstrap(groups, y, pa, pb, n=2000, seed=0):
    """按 group 重抽样，返回 (ΔR², ΔMAE) 的 2.5/50/97.5 分位；Δ = a − b。"""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in ug}
    dr2, dmae = [], []
    for _ in range(n):
        idx = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        ma, mb = metrics(y[idx], pa[idx]), metrics(y[idx], pb[idx])
        dr2.append(ma['R2'] - mb['R2'])
        dmae.append(ma['MAE'] - mb['MAE'])
    q = lambda v: [float(np.percentile(v, x)) for x in (2.5, 50, 97.5)]
    return dict(dR2=q(dr2), dMAE=q(dmae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-tabpfn', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    rows = load()
    n = len(rows)
    y = np.array([r['y'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    vis = np.array([r['vis'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([r['parity'] for r in rows])
    prev_bf = np.array([r['prev_bf'] for r in rows])
    log(f'[{time.strftime("%F %T")}] pigs={n} groups={len(set(groups))} '
        f'gest_day missing={int(np.isnan([r["gest_day"] for r in rows]).sum())} '
        f'prev_bf available={int((~np.isnan(prev_bf)).sum())}')

    use_tabpfn = not a.no_tabpfn
    if use_tabpfn:
        try:
            import tabpfn  # noqa: F401
        except ImportError:
            log('TabPFN not installed -> skipped')
            use_tabpfn = False

    # ---- 重建折，并与视觉运行的 baseline 列逐头核对 ----
    strat = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.seed)
    folds, fold_of, base = [], np.zeros(n, int), np.zeros(n)
    for k, (trv, te) in enumerate(sgkf.split(np.zeros(n), strat, groups)):
        inner = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=a.seed + k)
        tr_i, va_i = next(inner.split(np.zeros(len(trv)), strat[trv], groups[trv]))
        folds.append((trv, trv[tr_i], trv[va_i], te))
        fold_of[te] = k + 1
        base[te] = y[trv[tr_i]].mean()
    vb = np.array([r['vis_base'] for r in rows])
    max_diff = float(np.max(np.abs(base - vb)))
    log(f'fold check vs vision baseline column: max |diff| = {max_diff:.4f} '
        f'({"IDENTICAL folds" if max_diff < 1e-2 else "MISMATCH!"})')
    if max_diff >= 1e-2:
        sys.exit('fold mismatch, abort')

    preds = {'mean': base.copy(), 'vision_resnet18': vis.copy()}
    for fs in ('T1', 'T2'):
        Xr, cols = design(rows, fs)
        p_lin, p_xgb, p_pfn = np.zeros(n), np.zeros(n), np.zeros(n)
        for k, (trv, tr, va, te) in enumerate(folds):
            mask = np.zeros(n, bool)
            mask[tr] = True
            Xl = linear_design(Xr, mask, cols)
            mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-9
            lin = Ridge(alpha=1.0).fit((Xl[tr] - mu) / sd, y[tr])
            p_lin[te] = lin.predict((Xl[te] - mu) / sd)
            m = fit_xgb(Xr, y, tr, va, a.seed + k)
            p_xgb[te] = m.predict(Xr[te])
            if use_tabpfn:
                try:
                    p_pfn[te] = fit_tabpfn(Xr, y, trv, a.seed + k).predict(Xr[te])
                except Exception as e:  # 权重下载 / 许可等问题
                    log(f'TabPFN failed ({type(e).__name__}: {str(e)[:300]}) -> skipped')
                    use_tabpfn = False
            log(f'  {fs} fold {k + 1}: ridge MAE {np.mean(np.abs(p_lin[te] - y[te])):.3f}  '
                f'xgb MAE {np.mean(np.abs(p_xgb[te] - y[te])):.3f} (trees {m.best_iteration + 1})'
                + (f'  tabpfn MAE {np.mean(np.abs(p_pfn[te] - y[te])):.3f}' if use_tabpfn else ''))
        preds[f'{fs}_ridge'] = p_lin
        preds[f'{fs}_xgb'] = p_xgb
        if use_tabpfn:
            preds[f'{fs}_tabpfn'] = p_pfn

    # ---- 汇总：总体 + 品系内 ----
    subsets = {
        'all': np.ones(n, bool),
        'DAN_parity1': (line == 'DAN') & (parity == 1),
        'DAN_multiparous': (line == 'DAN') & (parity >= 2),
        'NAM_multiparous': (line == 'NAM') & (parity >= 2),
        'has_prev_season': ~np.isnan(prev_bf),
        'no_prev_season': np.isnan(prev_bf),   # 2026-10-11 补：首次测定（无上季记录），视觉的主要应用场景
    }
    table = {name: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for name, p in preds.items()}
    log('\nmodel                 subset              n    MAE   RMSE     R2    CCC  Spear')
    for name in preds:
        for s in subsets:
            r = table[name][s]
            log(f'{name:20s}  {s:17s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:6.3f}  '
                f'{r["CCC"]:.3f}  {r["Spearman"]:.3f}')

    # ---- 视觉 vs 各表格模型：配对检验 ----
    comps = {}
    log('')
    for name in preds:
        if name in ('mean', 'vision_resnet18'):
            continue
        for s in ('all', 'DAN_parity1', 'NAM_multiparous', 'no_prev_season', 'has_prev_season'):
            m = subsets[s]
            ea, eb = np.abs(vis[m] - y[m]), np.abs(preds[name][m] - y[m])
            w = wilcoxon(ea, eb)
            bs = group_bootstrap(groups[m], y[m], vis[m], preds[name][m])
            comps[f'vision_vs_{name}|{s}'] = dict(wilcoxon_p=float(w.pvalue), **bs)
            log(f'vision vs {name:12s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} (95% CI {bs["dR2"][0]:+.3f}, '
                f'{bs["dR2"][2]:+.3f})  ΔMAE {bs["dMAE"][1]:+.3f} (95% CI {bs["dMAE"][0]:+.3f}, '
                f'{bs["dMAE"][2]:+.3f})  Wilcoxon p={w.pvalue:.3g}')

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), n=n, fold_check_max_diff=max_diff, table=table, comparisons=comps),
                  f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        keys = list(preds)
        w.writerow(['uid', 'group', 'fold', 'line', 'parity', 'gest_day', 'batch', 'prev_bf', 'y'] + keys)
        for i, r in enumerate(rows):
            w.writerow([r['uid'], r['group'], fold_of[i], r['line'], r['parity'], r['gest_day'], r['batch'],
                        r['prev_bf'], r['y']] + [round(float(preds[kk][i]), 3) for kk in keys])
    log(f'done in {time.time() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
