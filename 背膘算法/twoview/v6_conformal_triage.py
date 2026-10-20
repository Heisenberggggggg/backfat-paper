# -*- coding: utf-8 -*-
"""
v6_conformal_triage.py — V6 保形预测区间 + B 超复测分流（只用已有 OOF 预测，不重新训练）

来源 / 背景
  06 §5.1 必做第 8 项；05 §3 ⑦、§8.2 A5。DECISIONS 10-11（五）：DK 需每批补测约 5 头 B 超校正截距。

数据（全部为已有输出，同一套 5 折）
  FMH：runs/a3_foundation/dinov3-vitb16_relh_stack/preds.csv（OOF；嵌套堆叠，无泄漏）
       + runs/v1_tabular/preds.csv（品系 / 胎次 / 上季背膘，按 uid 对齐）
       + runs/framepreds_new_s128/frame_preds.csv（ResNet-18 帧级 OOF 预测 → 猪内帧间 SD）
  DK ：runs/v7_dk_external/preds.csv（只用丰美禾训练的模型）

方法
  1. 区间（α = 0.10 主结果，另报 0.20）。测试折 k 的校准分数 = 其余 4 折的 OOF 残差（每个分数都来自
     未见过该猪的模型；与 CV+ / Barber 2021 同思路，覆盖率保证为近似）。
     a. marginal：|y − ŷ| 的保形分位数 ⌈(n+1)(1−α)⌉/n → 等宽区间
     b. mondrian：按 品系 × 胎次（丹系头胎 / 丹系经产 / 新美系头胎 / 新美系经产）分组计算分位数；
        组内校准数 < ⌈1/α⌉ 时退回 marginal
     c. normalized：分数 |y − ŷ| / σ̂(x)。σ̂ 为线性模型 |r| ~ 不确定性特征（帧间 SD、两种视觉模型分歧
        |vis_ridge − resnet18|、ŷ、品系、经产、有上季记录），**交叉拟合**：校准折 j 的分数用不含 j 的
        其余 3 个校准折拟合的 σ̂；测试折用 4 个校准折拟合的 σ̂。σ̂ 下限取校准 |r| 的 10% 分位
  2. 评估：覆盖率（Wilson 95% CI）、平均 / 中位宽度，按子集（全部、丹系头胎、新美系经产、首次测定、
     有上季记录）分别报告 → 条件覆盖是否失衡
  3. 复测分流：按排序分数把前 f（0–50%）的猪送 B 超（误差记 0）。报告全群 MAE、未复测猪 MAE、
     未复测猪中 |误差| > 3 mm 的比例。排序分数：随机（期望值）、σ̂（normalized 区间宽度）、帧间 SD、
     视觉分歧、mondrian 组宽度、oracle（真实 |误差|，上界）。σ̂ 排序 vs 随机的差按 group bootstrap 2000 次
     给 95% CI。另报 σ̂ 与 |误差| 的 Spearman、|误差|>3 mm 的 AUROC
  4. DK 外测：用 FMH OOF 残差的分位数（ResNet-18 → resnet18；DINOv3-relh ridge → vis_ridge）。
     未校准 vs 每批随机 k = 5 / 10 头 B 超截距校准（与 v7b 相同抽样，200 次），在非校准猪上评估覆盖率与宽度

输出：runs/v6_conformal/{log.txt, result.json, intervals.csv, triage_curve.csv}

用法：D:\\Anaconda\\envs\\pytorch\\python.exe v6_conformal_triage.py
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
from v6o_paths import INDEX_PATH, RUNS_ROOT  # noqa: E402  2026-10-20：runs 根目录 / 标签索引可切换（见 v6o_paths.py）
RUNS = RUNS_ROOT
SRC_STACK = os.path.join(RUNS, 'a3_foundation', 'dinov3-vitb16_relh_stack', 'preds.csv')
SRC_TAB = os.path.join(RUNS, 'v1_tabular', 'preds.csv')
SRC_FRAME = os.path.join(RUNS, 'framepreds_new_s128', 'frame_preds.csv')
SRC_DK = os.path.join(RUNS, 'v7_dk_external', 'preds.csv')
OUT = os.path.join(RUNS, 'v6_conformal')

# 主报告模型：相机单独（冻结 DINOv3 + TabPFN）、ResNet-18、记录 + 相机、记录 + 卡尺（人工对照）、最佳融合
MODELS = ('vis', 'resnet18', 'R+vis', 'R+cal', 'R+prev+cal+vis')
ALPHAS = (0.10, 0.20)
FRACS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
ERR_THR = 3.0  # mm
N_BOOT = 2000


def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def conf_q(scores, alpha):
    """保形分位数：第 ⌈(n+1)(1−α)⌉ 小的分数；n 不足时返回 inf。"""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float('inf') if k > n else float(s[k - 1])


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def fit_sigma(U, absr, floor):
    """σ̂(x) = max(floor, [1, U]·β)，β 由最小二乘拟合 |r|。"""
    A = np.hstack([np.ones((len(U), 1)), U])
    beta, *_ = np.linalg.lstsq(A, absr, rcond=None)
    return lambda V: np.maximum(floor, np.hstack([np.ones((len(V), 1)), V]) @ beta)


def auroc(score, label):
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def triage(err, score, f, rng=None):
    """把 score 最大的 f 比例送复测（误差记 0）；rng 给定时随机选。
    返回 (全群 MAE, 未复测猪 MAE, 未复测猪中 |err|>thr 比例)。"""
    n = len(err)
    m = int(round(f * n))
    order = rng.permutation(n) if rng is not None else np.argsort(-score, kind='stable')
    keep = np.ones(n, bool)
    keep[order[:m]] = False
    e = np.abs(err)
    return (float(e[keep].sum() / n), float(e[keep].mean()) if keep.any() else 0.0,
            float(np.mean(e[keep] > ERR_THR)) if keep.any() else 0.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] V6 conformal intervals + retest triage')
    log(f'   stack {SRC_STACK}\n   tab   {SRC_TAB}\n   frame {SRC_FRAME}\n   dk    {SRC_DK}')

    rows = read_csv(SRC_STACK)
    tab = {r['uid']: r for r in read_csv(SRC_TAB)}
    uid = np.array([r['uid'] for r in rows])
    assert all(u in tab for u in uid), 'stack / v1 uid 不一致'
    group = np.array([r['group'] for r in rows])
    fold = np.array([int(r['fold']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    assert np.allclose(y, [float(tab[u]['y']) for u in uid]), 'stack / v1 标签不一致'
    assert np.array_equal(fold, [int(tab[u]['fold']) for u in uid]), 'stack / v1 折不一致'
    P = {m: np.array([float(r[m]) for r in rows]) for m in MODELS + ('vis_ridge',)}
    line = np.array([tab[u]['line'] for u in uid])
    parity = np.array([fnum(tab[u]['parity']) for u in uid])
    prev = np.array([fnum(tab[u]['prev_bf']) for u in uid])

    # 帧间 SD（ResNet-18 帧级 OOF）
    fr = {}
    for r in read_csv(SRC_FRAME):
        fr.setdefault(r['uid'], []).append(float(r['pred']))
    assert all(u in fr for u in uid), '帧级预测缺猪'
    frame_sd = np.array([np.std(fr[u], ddof=1) if len(fr[u]) > 1 else 0.0 for u in uid])
    n_frames = np.array([len(fr[u]) for u in uid])
    diff_mean = np.max(np.abs(np.array([np.mean(fr[u]) for u in uid]) - P['resnet18']))
    log(f'   n={len(y)} pigs, {len(set(group))} groups; frames/pig median {np.median(n_frames):.0f} '
        f'(range {n_frames.min()}–{n_frames.max()}); |mean(frame) − resnet18 pig pred| max {diff_mean:.3f}')

    vis_dis = np.abs(P['vis_ridge'] - P['resnet18'])
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
    log(f'   σ̂ features: frame_sd (median {np.median(frame_sd):.2f}), |vis_ridge − resnet18| '
        f'(median {np.median(vis_dis):.2f}), ŷ, NAM, multiparous, has_prev')

    folds = sorted(set(fold))
    result = {'n': int(len(y)), 'folds': folds, 'fmh': {}, 'triage': {}, 'dk': {}}
    interval_rows = {u: {'uid': u, 'group': g, 'fold': int(k), 'y': float(t)} for u, g, k, t in
                     zip(uid, group, fold, y)}
    curve_rows = []

    for m in MODELS:
        p = P[m]
        r = y - p
        U = np.column_stack([frame_sd, vis_dis, p, is_nam, multi, has_prev])
        sd = U.std(0)
        U = (U - U.mean(0)) / np.where(sd > 0, sd, 1)

        sigma_test = np.zeros(len(y))
        res_m = {}
        for a in ALPHAS:
            lo = {meth: np.zeros(len(y)) for meth in ('marginal', 'mondrian', 'normalized')}
            hi = {meth: np.zeros(len(y)) for meth in lo}
            for k in folds:
                te, ca = fold == k, fold != k
                # a. marginal
                q = conf_q(np.abs(r[ca]), a)
                lo['marginal'][te], hi['marginal'][te] = p[te] - q, p[te] + q
                # b. mondrian
                for g in set(mondrian_key[te]):
                    cg = ca & (mondrian_key == g)
                    qg = conf_q(np.abs(r[cg]), a) if cg.sum() >= math.ceil(1 / a) else q
                    if not np.isfinite(qg):
                        qg = q
                    tg = te & (mondrian_key == g)
                    lo['mondrian'][tg], hi['mondrian'][tg] = p[tg] - qg, p[tg] + qg
                # c. normalized（交叉拟合：校准折 j 的分数用去掉 k、j 的 3 折拟合的 σ̂）
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

        # 3. 复测分流
        e = np.abs(r)
        mond_w = np.zeros(len(y))
        for k in folds:  # mondrian 组宽度（α=0.10）；同组并列时用 σ̂ 打破
            te, ca = fold == k, fold != k
            for g in set(mondrian_key[te]):
                cg = ca & (mondrian_key == g)
                mond_w[te & (mondrian_key == g)] = conf_q(e[cg] if cg.sum() >= 10 else e[ca], ALPHAS[0])
        rankers = {'sigma_hat': sigma_test, 'frame_sd': frame_sd, 'vis_disagree': vis_dis,
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
        # σ̂ 排序 vs 随机：group bootstrap（随机分流时未复测猪 MAE / 超阈比例的期望 = 全群值）
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
        log('     frac  random        sigma_hat     frame_sd      vis_disagree  mondrian      oracle')
        for f in FRACS:
            c = tri['curve'][f]
            log(f'     {f:4.2f}  ' + '  '.join(f'{c[f"{k}_mae_kept"]:.3f}/{c[f"{k}_gt3_kept"]:.3f}' for k in
                                             ('random', 'sigma_hat', 'frame_sd', 'vis_disagree', 'mondrian_width',
                                              'oracle')))
        for f, v in tri['sigma_vs_random'].items():
            log(f'     σ̂ − random @ {f:.0%}: ΔMAE(kept) {v["d_mae_kept"][1]:+.3f} '
                f'({v["d_mae_kept"][0]:+.3f}, {v["d_mae_kept"][2]:+.3f}); '
                f'Δshare|e|>3 {v["d_gt3_kept"][1]:+.3f} ({v["d_gt3_kept"][0]:+.3f}, {v["d_gt3_kept"][2]:+.3f})')

    # 4. DK 外测
    log('\n== DK external (FMH OOF residual quantile; per-batch intercept calibration k pigs/batch, 200 draws)')
    dk = read_csv(SRC_DK)
    dy = np.array([float(r['y']) for r in dk])
    dbatch = np.array([r['batch'] for r in dk])
    batches = sorted(set(dbatch))
    for dk_m, fmh_m in (('resnet18_mean3', 'resnet18'), ('dinov3_relh_ridge', 'vis_ridge')):
        dp = np.array([float(r[dk_m]) for r in dk])
        e_fmh = np.abs(y - P[fmh_m])
        result['dk'][dk_m] = {}
        for a in ALPHAS:
            q = conf_q(e_fmh, a)
            cov0 = np.abs(dy - dp) <= q
            out = {'fmh_model': fmh_m, 'q_fmh': q,
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
            result['dk'][dk_m][f'alpha={a}'] = out
            log(f'   {dk_m:18s} α={a}: q_FMH ±{q:.2f} mm | uncalibrated cov {out["uncalibrated"]["coverage"]:.3f} ('
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
