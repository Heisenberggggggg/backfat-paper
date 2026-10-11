# -*- coding: utf-8 -*-
"""
a3_foundation_regression.py — A3 + 公平融合：冻结基础模型特征 × 生产记录 × 卡尺 → TabPFN（同一套猪级 5 折）

来源 / 背景
  DECISIONS 2026-10-11（二）：相机（ResNet-18）≈ 人工卡尺，但「记录 + 卡尺」强于相机单独。公平比较需要：
    Q1  记录 + 相机   vs  记录 + 卡尺        —— 相机能否替代人工卡尺
    Q2  记录 + 卡尺 + 相机 vs 记录 + 卡尺    —— 在卡尺与记录之上相机是否还有增量
    Q3  同上再加上季背膘
  冻结特征无需训练 → 与表格变量一起进 TabPFN，无 OOF 堆叠泄漏。

输入
  runs/v1_tabular/preds.csv（猪、折、品系、胎次、妊娠日龄、上季背膘、ResNet-18 OOF 预测）
  runs/v1b_caliper/preds.csv（卡尺读数；纯整数 4/5 已按读数处理）
  runs/a3_foundation/feat_<model>_<enc>.npz（a3_extract_foundation_features.py 的猪级特征）

方法
  - 每折只在「训练 + 验证」组上：标准化 → PCA(--pca, 默认 16) → 与表格列拼接 → TabPFNRegressor；测试折预测一次
  - 视觉单独另报 RidgeCV（全维标准化特征，alpha 用训练集内 5 折选择）作参照
  - 特征组：vis / R / R+vis / R+cal / R+cal+vis / R+prev+cal / R+prev+cal+vis
    （R = 品系 + 胎次 + 妊娠日龄；prev = 上季背膘 + 是否有上季；cal = 卡尺）
  - 评估全部 663 头（卡尺缺 1 头的仅在含 cal 的组为 NaN，TabPFN 原生处理缺失值）；
    子集：全部 / 丹系头胎 / 新美系经产 / 首次测定 / 有上季记录
  - 配对比较：Wilcoxon + 按 group bootstrap（复用 v1_tabular_baseline.group_bootstrap）

输出：runs/a3_foundation/<model>_<enc>/{log.txt, result.json, preds.csv}

用法
  D:\\Anaconda\\envs\\pytorch\\python.exe a3_foundation_regression.py --model dinov3-vitb16 --enc relh
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v1_tabular_baseline import metrics, group_bootstrap  # noqa: E402

V1 = os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
FEAT_DIR = os.path.join(HERE, 'runs', 'a3_foundation')


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


DEVICE_USED = set()
_REG = {}
FORCE_CPU = False   # --device cpu：不建 CUDA 上下文（Windows 上 CUDA 上下文本身占数 GB 可提交内存）


def fit_predict_tabpfn(Xtr, ytr, Xte, seed):
    """GPU 优先，OOM 时退回 CPU；结果与设备无关（数值仅有浮点级差异）。
    本机约束（2026-10-11 实测）：桌面程序常占约 4 GB 显存、系统可提交内存仅约 5 GB →
      - 每个设备只建一个 TabPFNRegressor 并复用（反复新建会重复加载 876 MB 权重，耗尽可提交内存）
      - 不开 memory_saving_mode（该模式在 fit 的 _rebuild_raw_space_bardist 处 segfault）"""
    import gc
    import torch
    from tabpfn import TabPFNRegressor

    def run(dev):
        if dev not in _REG:
            _REG[dev] = TabPFNRegressor(random_state=seed, device=dev)
        reg = _REG[dev]
        reg.set_params(random_state=seed)
        p = reg.fit(Xtr, ytr).predict(Xte)
        gc.collect()
        DEVICE_USED.add(dev)
        return p

    if FORCE_CPU:
        return run('cpu')
    if torch.cuda.is_available():
        try:
            return run('cuda')
        except Exception as e:  # TabPFNCUDAOutOfMemoryError / torch OOM / 主机 "not enough memory"
            if 'memory' not in (type(e).__name__ + str(e)).lower():
                raise
            _REG.pop('cuda', None)
            gc.collect()
            torch.cuda.empty_cache()
    return run('cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--enc', required=True)
    # 2026-10-11：默认 32 → 16。本机可用显存约 4 GB，TabPFN v3.5 在 38 列时 OOM、22 列时峰值 1.6 GB（实测）
    ap.add_argument('--pca', type=int, default=16)
    ap.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    # 2026-10-11 v2：pca = 视觉以 PCA(--pca) 维进 TabPFN（v1，信息损失大：视觉单独 R² 仅 0.10–0.18）；
    #               stack = 嵌套堆叠：每个外层折内，对训练猪用「按 group 的内层 5 折」全维 RidgeCV 得到 OOF 视觉评分，
    #                       测试猪用在全部训练猪上拟合的 RidgeCV 预测；该 1 列评分与表格变量一起进 TabPFN（无泄漏）
    ap.add_argument('--vis', choices=['pca', 'stack'], default='pca')
    a = ap.parse_args()
    global FORCE_CPU
    FORCE_CPU = a.device == 'cpu'
    out = os.path.join(FEAT_DIR, f'{a.model}_{a.enc}' + ('' if a.vis == 'pca' else f'_{a.vis}'))
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    with open(V1, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with open(CAL, encoding='utf-8-sig') as f:
        cal_map = {r['uid']: fnum(r['caliper']) for r in csv.DictReader(f)}
    fz = np.load(os.path.join(FEAT_DIR, f'feat_{a.model}_{a.enc}.npz'))
    assert list(fz['uids']) == [r['uid'] for r in rows], 'pig order mismatch'
    F = fz['pig_feat'].astype(np.float64)
    n = len(rows)
    y = np.array([float(r['y']) for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    groups = np.array([r['group'] for r in rows])
    line = np.array([r['line'] for r in rows])
    parity = np.array([fnum(r['parity']) for r in rows])
    gest = np.array([fnum(r['gest_day']) for r in rows])
    prev = np.array([fnum(r['prev_bf']) for r in rows])
    cal = np.array([cal_map.get(r['uid'], np.nan) for r in rows])
    resnet = np.array([float(r['vision_resnet18']) for r in rows])
    R = np.column_stack([(line == 'NAM').astype(float), parity, gest])
    P = np.column_stack([prev, (~np.isnan(prev)).astype(float)])
    C = cal[:, None]
    log(f'[{time.strftime("%F %T")}] model={a.model} enc={a.enc} vis={a.vis} pigs={n} feat_dim={F.shape[1]} pca={a.pca} '
        f'caliper missing={int(np.isnan(cal).sum())}')

    sets = {'vis': (None, True), 'R': (R, False), 'R+vis': (R, True), 'R+cal': (np.hstack([R, C]), False),
            'R+cal+vis': (np.hstack([R, C]), True), 'R+prev+cal': (np.hstack([R, P, C]), False),
            'R+prev+cal+vis': (np.hstack([R, P, C]), True)}
    preds = {k: np.zeros(n) for k in sets}
    preds['vis_ridge'] = np.zeros(n)
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        sc = StandardScaler().fit(F[tr])
        rg = RidgeCV(alphas=np.logspace(-1, 4, 20), cv=5).fit(sc.transform(F[tr]), y[tr])
        preds['vis_ridge'][te] = rg.predict(sc.transform(F[te]))
        if a.vis == 'pca':
            pca = PCA(a.pca, random_state=0).fit(sc.transform(F[tr]))
            Z = pca.transform(sc.transform(F))
            msg = [f'  fold {k}: PCA var {pca.explained_variance_ratio_.sum():.2f}, ridge alpha {rg.alpha_:.1f}']
        else:
            from sklearn.model_selection import GroupKFold
            Z = np.zeros((n, 1))
            for itr, ite in GroupKFold(5).split(tr, groups=groups[tr]):
                isc = StandardScaler().fit(F[tr[itr]])
                irg = RidgeCV(alphas=np.logspace(-1, 4, 20), cv=5).fit(isc.transform(F[tr[itr]]), y[tr[itr]])
                Z[tr[ite], 0] = irg.predict(isc.transform(F[tr[ite]]))
            Z[te, 0] = preds['vis_ridge'][te]
            r_in = np.corrcoef(Z[tr, 0], y[tr])[0, 1]
            msg = [f'  fold {k}: stack ridge alpha {rg.alpha_:.1f}, inner-OOF r(train) {r_in:.3f}']
        for name, (tab, use_vis) in sets.items():
            X = np.hstack([x for x in (tab, Z if use_vis else None) if x is not None])
            preds[name][te] = fit_predict_tabpfn(X[tr], y[tr], X[te], 42 + k)
            msg.append(f'{name} {np.mean(np.abs(preds[name][te] - y[te])):.3f}')
        log(' | '.join(msg) + f' | device {sorted(DEVICE_USED)}')
    preds['resnet18'] = resnet

    subsets = {'all': np.ones(n, bool),
               'DAN_parity1': (line == 'DAN') & (parity == 1),
               'NAM_multiparous': (line == 'NAM') & (parity >= 2),
               'no_prev_season': np.isnan(prev),
               'has_prev_season': ~np.isnan(prev)}
    table = {nm: {s: metrics(y[m], p[m]) for s, m in subsets.items()} for nm, p in preds.items()}
    log('\nmodel                 subset              n    MAE   RMSE     R2    CCC  Spear')
    for nm in preds:
        for s in subsets:
            r = table[nm][s]
            log(f'{nm:20s}  {s:17s} {r["n"]:4d}  {r["MAE"]:.3f}  {r["RMSE"]:.3f}  {r["R2"]:6.3f}  '
                f'{r["CCC"]:.3f}  {r["Spearman"]:.3f}')

    pairs = [('vis', 'resnet18', 'foundation vs ResNet-18 (vision only)'),
             ('vis', 'R', 'vision only vs records'),
             ('R+vis', 'R', 'Q0 imaging increment over records'),
             ('R+vis', 'R+cal', 'Q1 camera vs caliper (with records)'),
             ('R+cal+vis', 'R+cal', 'Q2 camera increment over records+caliper'),
             ('R+prev+cal+vis', 'R+prev+cal', 'Q3 camera increment over records+prev+caliper')]
    comps = {}
    log('')
    for pa, pb, desc in pairs:
        for s in ('all', 'DAN_parity1', 'NAM_multiparous', 'no_prev_season'):
            m = subsets[s]
            w = wilcoxon(np.abs(preds[pa][m] - y[m]), np.abs(preds[pb][m] - y[m]))
            bs = group_bootstrap(groups[m], y[m], preds[pa][m], preds[pb][m])
            comps[f'{pa}_vs_{pb}|{s}'] = dict(desc=desc, wilcoxon_p=float(w.pvalue), **bs)
            log(f'{desc:48s} [{s:15s}] ΔR² {bs["dR2"][1]:+.3f} ({bs["dR2"][0]:+.3f}, {bs["dR2"][2]:+.3f})  '
                f'ΔMAE {bs["dMAE"][1]:+.3f} ({bs["dMAE"][0]:+.3f}, {bs["dMAE"][2]:+.3f})  Wilcoxon p={w.pvalue:.3g}')

    with open(os.path.join(out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(a), table=table, comparisons=comps), f, ensure_ascii=False, indent=1,
                  default=float)
    with open(os.path.join(out, 'preds.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        keys = list(preds)
        w.writerow(['uid', 'group', 'fold', 'y'] + keys)
        for i, r in enumerate(rows):
            w.writerow([r['uid'], r['group'], fold[i], y[i]] + [round(float(preds[kk][i]), 3) for kk in keys])
    log(f'done in {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
