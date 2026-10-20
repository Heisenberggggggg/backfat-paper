# -*- coding: utf-8 -*-
"""
v1b_caliper_vs_vision.py — 人工体况卡尺评分 vs 相机：同一套猪级 5 折下的对照

来源 / 背景
  本人 2026-10-11 确认：index_v2 的 bcs 列中，「5/4.2」斜杠后的 4.2 是人工用体况卡尺测得的评分，
  斜杠前的数不用管。卡尺是现场常用的人工体况工具，「相机能否替代人工卡尺」是 gComm 读者关心的对照。

解析规则
  - 含 '/'：取最后一个 '/' 之后的数；含空格（如 "3 3.0"）：取最后一段；字母 O 视为 0（"4.O"）
  - 纯小数（如 "4.4"）：直接取值
  - 纯整数（"4" / "5"，共 48 条）：本人 10-11 确认也是卡尺读数 → 取 4.0 / 5.0（v0 运行曾按缺失处理，
    输出另存为 runs/v1b_caliper/*_v0_intmissing.*）
  - 解析值 < 2（"///.8" → 0.8，明显录入错误）→ 缺失

思路
  - 猪与折：直接读取 runs/v1_tabular/preds.csv 的 fold 列（该列已与视觉模型逐头核对一致）
  - 模型（均在每折的「训练+验证」组上拟合，测试折只预测一次；无超参选择）
      caliper_lin      背膘 ~ 卡尺评分（线性校准，即现场把卡尺读数换算成背膘的最简做法）
      T1cal_tabpfn     TabPFN(品系 + 胎次 + 妊娠日龄 + 卡尺)
      T2cal_tabpfn     TabPFN(品系 + 胎次 + 妊娠日龄 + 上季背膘 + 是否有上季 + 卡尺)
  - 对照：视觉 ResNet-18、T1_tabpfn、T2_tabpfn（取自 v1_tabular/preds.csv）
  - 评估只在有有效卡尺读数的猪上进行；总体 + 品系内 + 首次测定；配对比较同 V1（Wilcoxon + 按 group bootstrap）

与旧脚本的区别：新脚本，只读已有文件；输出 runs/v1b_caliper/。

用法
  set HF_HUB_DISABLE_XET=1（仅首次下载 TabPFN 权重时需要）
  D:\\Anaconda\\envs\\pytorch\\python.exe v1b_caliper_vs_vision.py
"""
import csv
import json
import os
import re
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
sys.path.insert(0, HERE)
from v1_tabular_baseline import metrics, group_bootstrap  # noqa: E402

INDEX = INDEX_PATH
PREDS = os.path.join(RUNS_ROOT, 'v1_tabular', 'preds.csv')
OUT = os.path.join(RUNS_ROOT, 'v1b_caliper')


def parse_caliper(s):
    s = (s or '').strip()
    if '/' in s:
        s = s.split('/')[-1]
    elif ' ' in s:
        s = s.split()[-1]
    elif re.fullmatch(r'\d', s):
        return float(s), 'int_only'   # 2026-10-11 本人确认：只写「4」「5」的也是卡尺读数（v0 曾按缺失处理）
    s = s.replace('O', '0').replace('o', '0')
    try:
        v = float(s)
    except ValueError:
        return np.nan, 'unparsable'
    if v < 2:
        return np.nan, 'implausible'
    return v, 'ok'


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main():
    from scipy.stats import wilcoxon
    from tabpfn import TabPFNRegressor
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    bcs = {}
    with open(INDEX, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            bcs[r['uid']] = r['bcs']
    with open(PREDS, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    cal, flags = np.zeros(n), []
    for i, r in enumerate(rows):
        cal[i], fl = parse_caliper(bcs[r['uid']])
        flags.append(fl)
    flags = np.array(flags)
    log(f'[{time.strftime("%F %T")}] pigs={n}; caliper parse: '
        + ', '.join(f'{k}={int((flags == k).sum())}' for k in np.unique(flags)))

    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    groups = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    is_nam = (line == 'NAM').astype(float)
    has_prev = (~np.isnan(prev)).astype(float)
    ok = np.isin(flags, ['ok', 'int_only'])

    X1 = np.column_stack([is_nam, parity, gest, cal])
    X2 = np.column_stack([is_nam, parity, gest, prev, has_prev, cal])
    preds = {k: np.array([float(r[k]) for r in rows]) for k in ('vision_resnet18', 'T1_tabpfn', 'T2_tabpfn')}
    p_lin, p_t1, p_t2 = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for k in sorted(set(fold)):
        tr = np.where((fold != k) & ok)[0]
        te = np.where((fold == k) & ok)[0]
        lin = LinearRegression().fit(cal[tr, None], y[tr])
        p_lin[te] = lin.predict(cal[te, None])
        p_t1[te] = TabPFNRegressor(random_state=42 + k).fit(X1[tr], y[tr]).predict(X1[te])
        p_t2[te] = TabPFNRegressor(random_state=42 + k).fit(X2[tr], y[tr]).predict(X2[te])
        log(f'  fold {k}: train {len(tr)} test {len(te)}  caliper slope {lin.coef_[0]:.2f} mm/unit  '
            f'MAE lin {np.mean(np.abs(p_lin[te] - y[te])):.3f}  T1cal {np.mean(np.abs(p_t1[te] - y[te])):.3f}  '
            f'T2cal {np.mean(np.abs(p_t2[te] - y[te])):.3f}  vision {np.mean(np.abs(preds["vision_resnet18"][te] - y[te])):.3f}')
    preds.update(caliper_lin=p_lin, T1cal_tabpfn=p_t1, T2cal_tabpfn=p_t2)

    subsets = {
        'all_with_caliper': ok,
        'DAN_parity1': ok & (line == 'DAN') & (parity == 1),
        'NAM_multiparous': ok & (line == 'NAM') & (parity >= 2),
        'no_prev_season': ok & np.isnan(prev),
        'has_prev_season': ok & ~np.isnan(prev),
    }
    table = {name: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for name, p in preds.items()}
    log('\nmodel                 subset              n    MAE   RMSE     R2    CCC  Spear')
    for name in preds:
        for s in subsets:
            r = table[name][s]
            log(f'{name:20s}  {s:17s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:6.3f}  '
                f'{r["CCC"]:.3f}  {r["Spearman"]:.3f}')

    comps = {}
    log('')
    vis = preds['vision_resnet18']
    for name in ('caliper_lin', 'T1cal_tabpfn', 'T2cal_tabpfn'):
        for s, m in subsets.items():
            w = wilcoxon(np.abs(vis[m] - y[m]), np.abs(preds[name][m] - y[m]))
            bs = group_bootstrap(groups[m], y[m], vis[m], preds[name][m])
            comps[f'vision_vs_{name}|{s}'] = dict(wilcoxon_p=float(w.pvalue), **bs)
            log(f'vision vs {name:13s} [{s:16s}] ΔR² {bs["dR2"][1]:+.3f} (95% CI {bs["dR2"][0]:+.3f}, '
                f'{bs["dR2"][2]:+.3f})  ΔMAE {bs["dMAE"][1]:+.3f} (95% CI {bs["dMAE"][0]:+.3f}, '
                f'{bs["dMAE"][2]:+.3f})  Wilcoxon p={w.pvalue:.3g}')

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(parse_counts={k: int((flags == k).sum()) for k in np.unique(flags)}, table=table,
                       comparisons=comps), f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(OUT, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        keys = list(preds)
        w.writerow(['uid', 'group', 'fold', 'line', 'parity', 'y', 'bcs_raw', 'caliper', 'caliper_flag'] + keys)
        for i, r in enumerate(rows):
            w.writerow([r['uid'], r['group'], fold[i], line[i], r['parity'], y[i], bcs[r['uid']],
                        '' if np.isnan(cal[i]) else cal[i], flags[i]]
                       + ['' if np.isnan(preds[kk][i]) else round(float(preds[kk][i]), 3) for kk in keys])
    log(f'done in {time.time() - t0:.0f}s -> {OUT}')


if __name__ == '__main__':
    main()
