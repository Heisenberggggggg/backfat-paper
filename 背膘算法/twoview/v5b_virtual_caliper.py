# -*- coding: utf-8 -*-
"""
v5b_virtual_caliper.py — 在标准化高度图上复现体况卡尺：最后肋骨处背部横截面角度（虚拟卡尺）+ 分段体尺

来源 / 背景
  本人 10-13：体况卡尺（读数如 4.2、4.4）卡在最后一根肋骨处、按指针落在 5 个色区的位置读数（只是读数，无物理单位）；
  希望在视觉上加新内容。卡尺两板贴住背部两侧，指针反映背部横截面的「尖 / 平」程度：越瘦背越尖（角度小），越肥越平。
  高度图（v4a）保留了物理尺度的背部三维形状 → 可以数字化地复现这把卡尺，并补充常规体尺中缺的肩宽、臀宽、臀高等。
  先例：US 12,465,024 B1（背部三点背角估背膘）、Jian 2024（臀部参数）——写作时须引用并区分。

每帧算法（高度图 128 行（体宽方向）× 256 列（体长方向），7.8125 mm/px，像素 = 离地高度 mm）
  1. 栏杆列：列最大高度 > 背高 + 120 mm（栏杆在猪背上方）或列宽 < 体部列宽中位数的 60%（栏杆遮挡造成的缺口），左右各膨胀 1 列
  2. 体长范围：宽度 ≥ 80 mm 的列的首末列；头尾判定：两端各 20% 体长的列顶高度（95% 分位）中位数较低的一端为头
     （限位栏内猪低头；10-13 抽查 8 头中正常帧头均在低端）
  3. 横截面：在距头 f·体长处（f 主值 0.60 = 最后肋骨的解剖近似，另报 0.50–0.75）取 ±2 列（±16 mm）非栏杆列的逐行中位数；
     目标列落在栏杆上时向两侧 ±6 列内找最近的有效窗口
  4. 脊线 = 横截面有效行范围中点附近 ±5 行内的最高点；两侧各取距脊线 23 mm 至 D mm（D = 100 / 125 / 150 / 175，主值 150）
     的点，最小二乘拟合直线，下降角 θ = atan(−斜率)；**虚拟卡尺角 = 180° − θ左 − θ右**（越平越大）；另拟合
     h = a + b·d²（|d| ≤ D）给出背部曲率 b（越平越接近 0）
  5. 分段体尺：肩宽（f 0.20–0.40 最大列宽）、肋宽（f 0.60 列宽）、臀宽（f 0.75–0.92 最大列宽）、
     肩高（f 0.20–0.40 列顶高度中位数）、臀高（f 0.75–0.92 列顶高度最大值）、肋处脊高
  猪级 = 通过帧质控（同 v4b：ok=1、面积 ≥ 本猪中位 80%、体长 ≥ 85%）各帧的中位数（忽略该帧无效的量）
  主设置（f 0.60、D 150 mm）在看结果前预先指定；其余位置与跨距作敏感性分析。

评估（只描述性，不训练）：虚拟卡尺角 / 曲率 vs 人工卡尺读数、vs 背膘的 Pearson 与 Spearman（A 场全体、丹系、新美系；
  B 场只有背膘）。融合（替换人工卡尺）在 v5c 中做。

输入：yolo+FastSAM/hmap_dataset/{metadata.csv, <uid>/frame_XXX.npy}、runs/v1b_caliper/preds.csv、index_v2.csv
输出：runs/v5_caliper/{log.txt, result.json, features.csv}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe v5b_virtual_caliper.py
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
DM = os.path.join(HERE, '..', 'yolo+FastSAM', 'hmap_dataset')
CAL = os.path.join(HERE, 'runs', 'v1b_caliper', 'preds.csv')
INDEX = os.path.join(HERE, 'index_v2.csv')
OUT = os.path.join(HERE, 'runs', 'v5_caliper')

CELL = 7.8125
POS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
SPANS = (100, 125, 150, 175)
MAIN_POS, MAIN_SPAN = 0.60, 150
D0 = 23.0  # mm，脊线两侧起始距离（避开脊线本身）
META = ('uid', 'farm', 'batch', 'line', 'y', 'caliper', 'n_frames')


def col_top(h, m):
    """每列顶部高度（有值像素的 95% 分位），无值列为 NaN。"""
    hm = np.where(m, h, np.nan)
    with np.errstate(all='ignore'):
        return np.nanpercentile(hm, 95, axis=0)


def frame_features(h):
    m = h > 0
    width = m.sum(0) * CELL
    body = np.where(width >= 80)[0]
    if len(body) < 40:
        return None
    c0, c1 = body[0], body[-1]
    top = col_top(h, m)
    inb = np.zeros(h.shape[1], bool)
    inb[c0:c1 + 1] = True
    back = np.nanmedian(top[inb])
    wmed = np.median(width[inb & (width > 0)])
    colmax = np.where(m.any(0), h.max(0), 0)
    bar = inb & ((colmax > back + 120) | (width < 0.6 * wmed))
    bar = bar | np.roll(bar, 1) | np.roll(bar, -1)
    good = inb & ~bar
    L = c1 - c0
    n20 = max(int(0.2 * L), 3)
    with np.errstate(all='ignore'):
        t_lo = np.nanmedian(np.where(good[c0:c0 + n20], top[c0:c0 + n20], np.nan))
        t_hi = np.nanmedian(np.where(good[c1 - n20:c1 + 1], top[c1 - n20:c1 + 1], np.nan))
    if np.isnan(t_lo) or np.isnan(t_hi):
        return None
    head_left = t_lo <= t_hi
    if not head_left:  # 翻转为头在左
        h, m, width, top, good = h[:, ::-1], m[:, ::-1], width[::-1], top[::-1], good[::-1]
        c0, c1 = h.shape[1] - 1 - c1, h.shape[1] - 1 - c0
    col = lambda f: int(round(c0 + f * L))
    feats = {'head_left': float(head_left), 'length_mm': L * CELL}

    def seg(f_lo, f_hi):
        idx = np.arange(col(f_lo), col(f_hi) + 1)
        return idx[good[idx]]

    s = seg(0.20, 0.40)
    feats['shoulder_w'] = float(width[s].max()) if len(s) else np.nan
    feats['shoulder_h'] = float(np.nanmedian(top[s])) if len(s) else np.nan
    s = seg(0.75, 0.92)
    feats['hip_w'] = float(width[s].max()) if len(s) else np.nan
    feats['hip_h'] = float(np.nanmax(top[s])) if len(s) else np.nan

    for f in POS:
        c = col(f)
        win = None
        for off in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
            idx = np.arange(c + off - 2, c + off + 3)
            idx = idx[(idx >= 0) & (idx < h.shape[1])]
            idx = idx[good[idx]]
            if len(idx) >= 2:
                win = idx
                break
        if win is None:
            continue
        sub = np.where(m[:, win], h[:, win], np.nan)
        with np.errstate(all='ignore'):
            prof = np.nanmedian(sub, axis=1)
        rows = np.where(~np.isnan(prof))[0]
        if len(rows) < 20:
            continue
        mid = (rows[0] + rows[-1]) // 2
        cand = np.arange(max(mid - 5, 0), min(mid + 6, len(prof)))
        cand = cand[~np.isnan(prof[cand])]
        if not len(cand):
            continue
        sp = cand[np.argmax(prof[cand])]
        tag = f'{f:.2f}'
        if f == MAIN_POS:
            feats['rib_w'] = float(width[win].mean())
            feats['rib_spine_h'] = float(prof[sp])
        for D in SPANS:
            dd = np.arange(int(np.ceil(D0 / CELL)), int(D / CELL) + 1)
            thetas = []
            for sgn in (-1, 1):
                r = sp + sgn * dd
                ok = (r >= 0) & (r < len(prof))
                r, d = r[ok], dd[ok] * CELL
                v = prof[r]
                k = ~np.isnan(v)
                if k.sum() < 0.6 * len(dd):
                    thetas = None
                    break
                slope = np.polyfit(d[k], v[k], 1)[0]
                thetas.append(np.degrees(np.arctan(-slope)))
            if thetas is not None:
                feats[f'angle_f{tag}_D{D}'] = 180.0 - thetas[0] - thetas[1]
            r = np.arange(sp - int(D / CELL), sp + int(D / CELL) + 1)
            r = r[(r >= 0) & (r < len(prof))]
            v = prof[r]
            k = ~np.isnan(v)
            if k.sum() >= 0.6 * len(r):
                feats[f'curv_f{tag}_D{D}'] = float(np.polyfit(((r[k] - sp) * CELL) ** 2, v[k], 1)[0])
    return feats


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    t0 = time.time()
    log(f'[{time.strftime("%F %T")}] v5b virtual caliper | main f={MAIN_POS} D={MAIN_SPAN} mm')
    by = {}
    with open(os.path.join(DM, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['ok'] == '1':
                by.setdefault(r['uid'], []).append(r)
    with open(INDEX, encoding='utf-8-sig') as f:
        idx = {r['uid']: r for r in csv.DictReader(f)}
    with open(CAL, encoding='utf-8-sig') as f:
        calr = {r['uid']: r for r in csv.DictReader(f)}

    rows, n_head_left, n_fr, n_fail = [], 0, 0, 0
    for i, (u, rs) in enumerate(sorted(by.items())):
        if u not in idx or not idx[u].get('backfat'):
            continue
        ma = np.median([float(r['area_cm2']) for r in rs])
        ml = np.median([float(r['length_mm']) for r in rs])
        rs = [r for r in rs if float(r['area_cm2']) >= 0.8 * ma and float(r['length_mm']) >= 0.85 * ml]
        fl = []
        for r in rs:
            ff = frame_features(np.load(os.path.join(DM, r['frame_path'])).astype(np.float32))
            n_fr += 1
            if ff is None:
                n_fail += 1
                continue
            n_head_left += ff['head_left']
            fl.append(ff)
        if not fl:
            continue
        allk = sorted({k for ff in fl for k in ff})
        with np.errstate(all='ignore'):
            med = {k: float(np.nanmedian([ff.get(k, np.nan) for ff in fl])) for k in allk}
        med['n_valid_angle'] = int(sum(f'angle_f{MAIN_POS:.2f}_D{MAIN_SPAN}' in ff for ff in fl))
        cr = calr.get(u)
        cal = float(cr['caliper']) if cr and cr['caliper'] and cr['caliper_flag'] != 'implausible' else np.nan
        line = cr['line'] if cr else ('DAN' if rs[0]['farm'] == 'DK' else '')
        rows.append(dict(uid=u, farm=rs[0]['farm'], batch=rs[0]['batch'], line=line,
                         y=float(idx[u]['backfat']), caliper=cal, n_frames=len(fl), **med))
        if (i + 1) % 100 == 0:
            log(f'  {i + 1}/{len(by)} pigs | {(time.time() - t0) / 60:.1f} min')
    log(f'   pigs {len(rows)} (FMH {sum(r["farm"] == "FMH" for r in rows)}, DK {sum(r["farm"] == "DK" for r in rows)}); '
        f'frames {n_fr}, failed {n_fail}; head on low-x side before flip {n_head_left / max(n_fr - n_fail, 1):.1%}')

    keys = sorted({k for r in rows for k in r} - set(META))
    with open(os.path.join(OUT, 'features.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        head = list(META) + keys
        w.writerow(head)
        for r in rows:
            w.writerow([round(r[k], 5) if isinstance(r.get(k), float) else r.get(k, '') for k in head])

    def arr(k, sel):
        return np.array([r.get(k, np.nan) for r in rows if sel(r)], float)

    def corr(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b)
        if ok.sum() < 10:
            return dict(n=int(ok.sum()), pearson=np.nan, spearman=np.nan)
        return dict(n=int(ok.sum()), pearson=float(pearsonr(a[ok], b[ok])[0]),
                    spearman=float(spearmanr(a[ok], b[ok])[0]))

    # A 场相关只在 663 头分析集上算（index_v2 另有 4 条已排除记录：匹配冲突 2、背膘超出 7–32 mm 2；
    # 2026-10-13 前两次运行误含这 4 条，极端背膘把全体 Pearson 拉低到 0.13，见 log 前两段）
    with open(os.path.join(HERE, 'runs', 'v1_tabular', 'preds.csv'), encoding='utf-8-sig') as f:
        analysis = {r['uid'] for r in csv.DictReader(f)}
    fmh = lambda r: r['farm'] == 'FMH' and r['uid'] in analysis
    log(f'   Farm A analysis set: {sum(fmh(r) for r in rows)} pigs')
    groups = {'all': fmh, 'DAN': lambda r: fmh(r) and r['line'] == 'DAN',
              'NAM': lambda r: fmh(r) and r['line'] == 'NAM'}
    res = {'n': len(rows), 'main': f'angle_f{MAIN_POS:.2f}_D{MAIN_SPAN}', 'corr': {}}
    log('\nFarm A: r with manual caliper (Pearson / Spearman) | r with backfat (Pearson / Spearman); all, DAN, NAM')
    show = ([f'angle_f{f:.2f}_D{D}' for f in POS for D in SPANS] + [f'curv_f{MAIN_POS:.2f}_D{D}' for D in SPANS]
            + ['shoulder_w', 'rib_w', 'hip_w', 'shoulder_h', 'rib_spine_h', 'hip_h', 'length_mm', 'caliper'])
    for k in show:
        res['corr'][k] = {}
        parts = []
        for gname, sel in groups.items():
            a = arr(k, sel)
            cc, cy = corr(a, arr('caliper', sel)), corr(a, arr('y', sel))
            res['corr'][k][gname] = dict(caliper=cc, backfat=cy)
            parts.append(f'{gname} cal {cc["pearson"]:+.3f}/{cc["spearman"]:+.3f} '
                         f'bf {cy["pearson"]:+.3f}/{cy["spearman"]:+.3f}')
        log(f'   {k:20s} ' + ' | '.join(parts) + (' ◀ main' if k == res['main'] else ''))
    dk = lambda r: r['farm'] == 'DK'
    log('\nFarm B (no caliper): r with backfat (Pearson / Spearman)')
    res['dk'] = {}
    for k in [f'angle_f{MAIN_POS:.2f}_D{D}' for D in SPANS] + ['shoulder_w', 'rib_w', 'hip_w', 'hip_h']:
        cy = corr(arr(k, dk), arr('y', dk))
        res['dk'][k] = cy
        log(f'   {k:20s} n={cy["n"]} bf {cy["pearson"]:+.3f}/{cy["spearman"]:+.3f}')
    ang = arr(res['main'], fmh)
    log(f'\nmain angle (Farm A): median {np.nanmedian(ang):.1f}°, IQR {np.nanpercentile(ang, 25):.1f}–'
        f'{np.nanpercentile(ang, 75):.1f}°, missing {int(np.isnan(ang).sum())}')
    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    log(f'done in {(time.time() - t0) / 60:.1f} min -> {OUT}')


if __name__ == '__main__':
    main()
