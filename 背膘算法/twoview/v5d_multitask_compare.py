# -*- coding: utf-8 -*-
"""
v5d_multitask_compare.py — v5a 四种训练目标 vs v4b 基线（同折、同内层验证集、同种子）的逐头配对比较

比较内容
  1. 背膘：各模式 3 种子平均预测 vs v4b（pred_mean）：R²、MAE、校准斜率；ΔR² / ΔMAE 的 group bootstrap 95% CI、
     |误差| 的 Wilcoxon；子集：全部、丹系头胎、新美系经产、首次测定
  2. 相机预测的卡尺读数（cal / caldims 的辅助头，OOF）vs 人工卡尺：Pearson / Spearman（全部、丹系、新美系、两季），
     与 v5b 虚拟卡尺角并列；以及与背膘的相关
  3. 方差头（nll）：猪级预测方差与 |误差| 的 Spearman、|误差| > 3 mm 的 AUROC；按方差复测 10 / 20 / 30% 的
     未复测猪 MAE vs 随机（group bootstrap）——复测分流在网络直接输出方差时是否成立

输入：runs/v4_hmap/hmap/test_preds.csv、runs/v5_multitask/{cal,caldims,lds,nll}/test_preds.csv、
      runs/v1_tabular/preds.csv、runs/v1b_caliper/preds.csv、runs/v5_caliper/features.csv
输出：runs/v5_multitask/compare/{log.txt, result.json}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5d_multitask_compare.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cv_depthmap_labelfix import metrics  # noqa: E402
from v1_tabular_baseline import group_bootstrap  # noqa: E402
from v6_conformal_triage import auroc, triage  # noqa: E402

RUNS = os.path.join(HERE, 'runs')
BASE = os.path.join(RUNS, 'v4_hmap', 'hmap', 'test_preds.csv')
MODES = ('cal', 'caldims', 'lds', 'nll')
OUT = os.path.join(RUNS, 'v5_multitask', 'compare')


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
    log(f'[{time.strftime("%F %T")}] v5d multitask vs v4b baseline')
    base = read(BASE)
    tab = read(os.path.join(RUNS, 'v1_tabular', 'preds.csv'))
    calr = read(os.path.join(RUNS, 'v1b_caliper', 'preds.csv'))
    feat = read(os.path.join(RUNS, 'v5_caliper', 'features.csv'))
    uid = sorted(base)
    y = np.array([float(base[u]['y']) for u in uid])
    group = np.array([base[u]['group'] for u in uid])
    fold = np.array([int(base[u]['fold']) for u in uid])
    line = np.array([tab[u]['line'] for u in uid])
    parity = np.array([fnum(tab[u]['parity']) for u in uid])
    prev = np.array([fnum(tab[u]['prev_bf']) for u in uid])
    camp = np.array(['Aug' if u.startswith('fmh_8.') else 'OctNov' for u in uid])
    cal = np.array([fnum(calr[u]['caliper']) if calr[u]['caliper_flag'] != 'implausible' else np.nan for u in uid])
    vang = np.array([fnum(feat[u].get('angle_f0.60_D150')) for u in uid])
    pb = np.array([float(base[u]['pred_mean']) for u in uid])
    subsets = {'all': np.ones(len(y), bool), 'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2), 'no_prev_season': np.isnan(prev)}
    res = {'baseline': metrics(y, pb), 'modes': {}}
    res['baseline']['slope'] = float(np.polyfit(y, pb, 1)[0])
    log(f'   baseline v4b: R² {res["baseline"]["R2"]:.3f} MAE {res["baseline"]["MAE"]:.3f} '
        f'slope {res["baseline"]["slope"]:.3f}')

    log('\n1. backfat: mode vs v4b (Δ = mode − v4b; group bootstrap 95% CI; Wilcoxon on |error|)')
    P = {}
    for m in MODES:
        r = read(os.path.join(RUNS, 'v5_multitask', m, 'test_preds.csv'))
        assert np.array_equal(fold, [int(r[u]['fold']) for u in uid]), f'{m} 折不一致'
        P[m] = r
        p = np.array([float(r[u]['pred_mean']) for u in uid])
        mm = metrics(y, p)
        mm['slope'] = float(np.polyfit(y, p, 1)[0])
        res['modes'][m] = {'metrics': mm, 'vs_v4b': {}}
        log(f'   {m:8s} R² {mm["R2"]:.3f} MAE {mm["MAE"]:.3f} slope {mm["slope"]:.3f}')
        for s, k in subsets.items():
            bs = group_bootstrap(group[k], y[k], p[k], pb[k])
            pw = float(wilcoxon(np.abs(p[k] - y[k]), np.abs(pb[k] - y[k])).pvalue)
            res['modes'][m]['vs_v4b'][s] = dict(**bs, wilcoxon_p=pw)
            log(f'      [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  p={pw:.3g}')

    log('\n2. camera-predicted caliper (aux head, OOF) vs manual caliper: Pearson / Spearman')
    res['caliper'] = {}
    groups = {'all': np.ones(len(y), bool), 'DAN': line == 'DAN', 'NAM': line == 'NAM',
              'Aug': camp == 'Aug', 'OctNov': camp == 'OctNov'}
    cands = {f'{m}_aux': np.array([fnum(P[m][u]['aux_caliper']) for u in uid]) for m in ('cal', 'caldims')}
    cands['virtual_angle'] = vang
    for name, v in cands.items():
        res['caliper'][name] = {}
        parts = []
        for g, k in groups.items():
            ok = k & ~np.isnan(v) & ~np.isnan(cal)
            okb = k & ~np.isnan(v)
            d = dict(r_cal=float(pearsonr(v[ok], cal[ok])[0]), rho_cal=float(spearmanr(v[ok], cal[ok])[0]),
                     r_bf=float(pearsonr(v[okb], y[okb])[0]), n=int(ok.sum()))
            res['caliper'][name][g] = d
            parts.append(f'{g} cal {d["r_cal"]:+.3f}/{d["rho_cal"]:+.3f} bf {d["r_bf"]:+.3f}')
        log(f'   {name:14s} ' + ' | '.join(parts))
    parts = []
    for g, k in groups.items():
        ok = k & ~np.isnan(cal)
        parts.append(f'{g} bf {pearsonr(cal[ok], y[ok])[0]:+.3f}')
    log(f'   {"manual caliper":14s} ' + ' | '.join(parts))

    log('\n3. variance head (nll): can predicted variance flag inaccurate pigs?')
    pn = np.array([float(P['nll'][u]['pred_mean']) for u in uid])
    lv = np.array([fnum(P['nll'][u]['aux_logvar']) for u in uid])
    e = np.abs(y - pn)
    rq = dict(spearman=float(spearmanr(lv, e)[0]), auroc_gt3=auroc(lv, e > 3.0))
    log(f'   Spearman(logvar, |e|) {rq["spearman"]:+.3f}; AUROC(|e|>3) {rq["auroc_gt3"]:.3f}')
    ug = np.unique(group)
    gidx = {g: np.where(group == g)[0] for g in ug}
    rng = np.random.default_rng(1)
    boot = {f: [] for f in (0.10, 0.20, 0.30)}
    for _ in range(2000):
        idx = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug), replace=True)])
        for f in boot:
            _, kept, _ = triage(y[idx] - pn[idx], lv[idx], f)
            boot[f].append(kept - np.mean(e[idx]))
    rq['retest_vs_random'] = {}
    for f, v in boot.items():
        q = [float(x) for x in np.percentile(v, [2.5, 50, 97.5])]
        rq['retest_vs_random'][f] = q
        log(f'   retest {f:.0%} by variance: ΔMAE(non-retested) vs random {q[1]:+.3f} ({q[0]:+.3f}, {q[2]:+.3f})')
    res['nll_triage'] = rq
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done in {time.time() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
