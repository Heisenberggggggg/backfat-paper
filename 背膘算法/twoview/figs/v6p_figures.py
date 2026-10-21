# -*- coding: utf-8 -*-
"""
v6p_figures.py — gComm 稿件图 2–4 重新设计（2026-10-20，按同类期刊常用图型调研结果；图 1 另配插画）

调研结论（详见 十月论文/figures_v3/图型调研与改进方案.md）：
  同类论文几乎都有「预测值–实测值散点图（1:1 线 + R² / MAE）」，方法比对研究还常给一致性图（偏差与一致性界限）；
  可解释性多用「热图叠在图像上 + 沿体长的曲线」（PigFormer 2026）；特征与背膘的关系常用相关热力图（Jian 2024）。
新的图（期刊限 8 个图表，保持 4 图）：
  Fig 2 (a) 预测值–B 超实测值散点（主模型 5 种子平均的折外预测；1:1 线、拟合线、R² / MAE / RMSE）；
        (b) 预测误差–B 超背膘（偏差与 95% 一致性界限 + 分段均值 ± 95% CI + 各段 MAE；替代旧 Fig 4d）；
        (c) 示例猪高度图叠加横截面注意力热带，下方为同一横轴的总体注意力曲线（替代旧 Fig 2a、2b）。
  Fig 3 (a) 统一的 ΔR² 森林图（全体母猪；表示、第二台相机、相机与卡尺 / 生产记录、增加信息源；替代旧 Fig 2c、3c，
        品系内子集见表 2、表 3）；(b) 卡尺读数–背膘（同旧 Fig 3b）；
        (c) 虚拟卡尺敏感性热力图：位置 × 跨距下与人工卡尺、与背膘的 Pearson r（旧 Fig 3a 散点移到补充材料）。
  Fig 4 一行三图：(a) 信度与 R² 随帧数；(b) 90% 保形区间覆盖率；(c) 复测分流（同旧 Fig 4a–c）。
数据：全部从 runs 根目录读取（v6o_paths：设 BACKFAT_RUNS=runs_v6o 即用 622 条重跑结果），每个数字写入
      <runs>/v6p_figures/log.txt。版式、配色、字体沿用 v6m（import），示例猪选取规则沿用 v6i（预先设定，非挑结果）。
输出：十月论文/figures_v3/fig{2,3,4}.{tif,pdf,svg,jpg,png}（可用环境变量 FIG_OUT 改目录）
用法：D:\\Anaconda\\envs\\pytorch\\python.exe figs/v6p_figures.py --fig 2 3 4
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from v6o_paths import RUNS_ROOT  # noqa: E402
import v6i_figures as v6i  # noqa: E402
import v6m_figures as m  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

v6i.RUNS = m.RUNS = RUNS_ROOT
m.OUT = os.environ.get('FIG_OUT') or r'F:/Postgraduate/Lab/十月论文/figures_v3'
m.LOG = os.path.join(RUNS_ROOT, 'v6p_figures', 'log.txt')
log, axmm, plabel, clean, new_fig = m.log, m.axmm, m.plabel, m.clean, m.new_fig
INK, AXC, REF, LIGHT, GRID = m.INK, m.AXC, m.REF, m.LIGHT, m.GRID
C_DAN, C_NAM, C_ALL, C_MAIN = m.C_DAN, m.C_NAM, m.C_ALL, m.C_MAIN
F_POS, CELL = m.F_POS, m.CELL
ATT_CMAP = LinearSegmentedColormap.from_list('att', ['#ffffff', '#9fdcc0', C_MAIN, '#0b5e40'])
HEAT_CMAP = LinearSegmentedColormap.from_list('heat', ['#f7f7f4', '#d9d4f2', '#8f84d6', '#463a99'])  # 单色紫：青绿只留给主模型


MIN_PT, MAX_PT, MIN_LW = 8.0, 10.0, 0.4


def comply(fig, name):
    """gComm 投稿须知 3.4(15)：图中文字 Times New Roman、非粗体、8–10 pt；线宽不低于 0.4 pt。保存前统一修正并记录。"""
    import matplotlib.text as mtext
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch
    n_pt = n_bold = n_lw = 0
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        fs = t.get_fontsize()
        if fs < MIN_PT or fs > MAX_PT:
            t.set_fontsize(min(max(fs, MIN_PT), MAX_PT))
            n_pt += 1
        if t.get_fontproperties().get_weight() not in ('normal', 400, 'regular', 'book'):
            t.set_fontweight('normal')
            n_bold += 1
    for a in fig.findobj(lambda o: isinstance(o, (Line2D, Patch))):
        lw = a.get_linewidth()
        if 0 < lw < MIN_LW:
            a.set_linewidth(MIN_LW)
            n_lw += 1
    for c in fig.findobj(Collection):
        lws = np.atleast_1d(c.get_linewidth())
        if np.any((lws > 0) & (lws < MIN_LW)):
            c.set_linewidth(np.where((lws > 0) & (lws < MIN_LW), MIN_LW, lws))
            n_lw += 1
    log(f'{name}: 字号调到 8–10 pt {n_pt} 处，粗体改常规 {n_bold} 处，线宽补到 0.4 pt {n_lw} 处')


def save(fig, name):
    comply(fig, name)
    m.save(fig, name)


def metrics(y, p):
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return dict(R2=r2, MAE=np.mean(np.abs(y - p)), RMSE=np.sqrt(np.mean((y - p) ** 2)), n=len(y))


def main_preds():
    """主模型（解剖引导网络，5 种子平均，内层验证集选 epoch）的猪级折外预测 + 品系 / 胎次。"""
    rows = v6i.read_csv('v6_anatomy/prof/test_preds.csv')
    v1 = {r['uid']: r for r in v6i.read_csv('v1_tabular/preds.csv')}
    y = np.array([float(r['y']) for r in rows])
    p = np.array([float(r['pred_inner']) for r in rows])
    line = np.array([v1[r['uid']]['line'] for r in rows])
    par = np.array([v6i.fnum(v1[r['uid']]['parity']) for r in rows])
    return y, p, line, par


# ---------------------------------------------------------------- Fig 2
def scatter_lines(ax, x, y, line, rng):
    for ln, mk in (('DAN', 'o'), ('NAM', '^')):
        k = line == ln
        ax.scatter(x[k] + rng.uniform(-0.18, 0.18, k.sum()), y[k], s=6 if ln == 'DAN' else 7, marker=mk,
                   color=m.LINE_COL[ln], alpha=0.45, linewidths=0, zorder=3, rasterized=True)


def fig_accuracy():
    """正文图 3（2026-10-21 起；原图 2）：精度与注意力。"""
    y, p, line, par = main_preds()
    g = metrics(y, p)
    slope, icpt = np.polyfit(y, p, 1)
    log(f'Fig 3a: n {g["n"]}; R² {g["R2"]:.4f}, MAE {g["MAE"]:.4f}, RMSE {g["RMSE"]:.4f}; OLS pred = {icpt:.3f} + '
        f'{slope:.4f} × ultrasound (runs/v6_anatomy/prof/test_preds.csv pred_inner, 5-seed mean)')
    fig = new_fig(150)
    rng = np.random.default_rng(3)
    lo, hi = np.floor(min(y.min(), p.min())) - 1, np.ceil(max(y.max(), p.max())) + 1

    # (a) 预测值–实测值
    axa = axmm(fig, 13, 6, 60, 56)
    axa.plot([lo, hi], [lo, hi], color=REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    scatter_lines(axa, y, p, line, rng)
    xl = np.array([y.min(), y.max()])
    axa.plot(xl, icpt + slope * xl, color=INK, lw=1.2, zorder=4)
    axa.text(hi - 0.6, hi - 1.4, '1:1', ha='right', va='top', fontsize=7, color=INK)
    xt = hi - 0.8
    axa.text(xt, icpt + slope * xt + 0.9, f'Fitted, slope {slope:.2f}', ha='right', va='bottom', fontsize=7,
             bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.6), zorder=6)
    axa.text(lo + 0.8, hi - 0.8, f'R² = {g["R2"]:.3f}\nMAE = {g["MAE"]:.2f} mm\nRMSE = {g["RMSE"]:.2f} mm\n'
             f'n = {g["n"]}', ha='left', va='top', fontsize=8, linespacing=1.25)
    hs = [Line2D([], [], color=m.LINE_COL[ln], marker=mk, lw=0, ms=4, alpha=0.8,
                 label=f'{m.LINE_NAME[ln]} (n = {(line == ln).sum()})') for ln, mk in (('DAN', 'o'), ('NAM', '^'))]
    axa.legend(handles=hs, loc='lower right', handletextpad=0.2, borderaxespad=0.1)
    axa.set_xlim(lo, hi)
    axa.set_ylim(lo, hi)
    axa.set_aspect('equal')
    axa.set_xlabel('Backfat, ultrasound (mm)')
    axa.set_ylabel('Backfat, predicted (mm)')
    clean(axa, grid='both')

    # (b) 误差–实测值（偏差、一致性界限、分段均值）
    e = p - y
    bias, sd = e.mean(), e.std(ddof=1)
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    axb = axmm(fig, 96, 6, 76, 56)
    scatter_lines(axb, y, e, line, rng)
    for v, ls in ((bias, '-'), (loa[0], (0, (3, 2))), (loa[1], (0, (3, 2)))):
        axb.axhline(v, color=INK if ls == '-' else REF, lw=0.8 if ls == '-' else 0.7, ls=ls, zorder=2)
    axb.axhline(0, color=LIGHT, lw=0.6, zorder=1)
    xr = (lo, hi)
    axb.text(xr[1] - 0.3, loa[1] + 0.25, f'+1.96 SD ({loa[1]:+.1f})'.replace('-', '−'), ha='right', va='bottom',
             fontsize=7)
    axb.text(xr[0] + 0.3, loa[0] - 0.25, f'−1.96 SD ({loa[0]:+.1f})'.replace('-', '−'), ha='left', va='top',
             fontsize=7)
    axb.text(xr[1] - 0.3, bias + 0.25, f'Mean bias {bias:+.2f} mm'.replace('-', '−'), ha='right', va='bottom',
             fontsize=7)
    parts = []
    for blo, bhi, lab in v6i.BINS:
        k = np.ones(len(y), bool)
        if blo is not None:
            k &= y >= blo
        if bhi is not None:
            k &= y <= bhi
        if k.sum() < 3:
            continue
        xc = y[k].mean()
        mu, ci = e[k].mean(), m.tcrit(k.sum()) * e[k].std(ddof=1) / np.sqrt(k.sum())
        axb.plot([xc, xc], [mu - ci, mu + ci], color=INK, lw=1.2, solid_capstyle='butt', zorder=5)
        axb.plot(xc, mu, 'D', ms=3.6, mfc='white', mec=INK, mew=1.0, zorder=6)
        axb.text(xc, 11.2, f'{np.abs(e[k]).mean():.1f}', ha='center', va='center', fontsize=7)
        parts.append(f'{lab} n={k.sum()} mean err={mu:+.2f} (±{ci:.2f}) MAE={np.abs(e[k]).mean():.2f}')
    axb.text(lo + 0.3, 11.2, 'MAE', ha='left', va='center', fontsize=7)
    axb.set_xlim(*xr)
    axb.set_ylim(-12.5, 12.5)
    axb.set_xlabel('Backfat, ultrasound (mm)')
    axb.set_ylabel('Predicted − ultrasound (mm)')
    clean(axb)
    log(f'Fig 3b: bias {bias:+.4f} mm, SD {sd:.4f}, 95% LoA ({loa[0]:+.3f}, {loa[1]:+.3f}); bins: ' + '; '.join(parts))

    # (c) 高度图 + 注意力热带；下方总体曲线（同一横轴：% 体长，头在左）
    rows, att, i, uid, fr, ho, c0, c1, good = m.example()
    seed_curves = np.nanmean(att, 1)
    mean_curve = np.nanmean(att, (0, 1))
    res = v6i.load('v6_anatomy/prof/result.json')
    assert np.allclose(mean_curve, res['position_attention_mean'], atol=1e-6), '与 result.json 不一致'
    w = np.nanmean(att[:, i], 0)
    L = c1 - c0
    rows_on = m.body_rows(ho, c0, c1)
    r0, r1 = rows_on[0] - 2, rows_on[-1] + 2
    pct = lambda c: (c - c0) / L * 100  # noqa: E731
    xlim = (-4, 104)
    wmap = 118.0
    hmap = wmap * ((r1 - r0) * CELL) / ((xlim[1] - xlim[0]) / 100 * L * CELL)
    axm = axmm(fig, 13, 76, wmap, hmap)
    hm = np.ma.masked_where(ho <= 0, ho)
    axm.imshow(hm, cmap='Greys', vmin=300, vmax=1100, interpolation='nearest', aspect='auto',
               extent=(pct(-0.5), pct(ho.shape[1] - 0.5), ho.shape[0] - 0.5, -0.5), zorder=1)
    for a, b in m.occluded_runs(c0, c1, good):
        axm.add_patch(Rectangle((pct(a - 0.5), rows_on[0] - 1), pct(b + 0.5) - pct(a - 0.5),
                                rows_on[-1] - rows_on[0] + 2, fc='white', ec='none', alpha=0.72, zorder=2))
    # 注意力色带只染在猪体像素上（类似 Grad-CAM 叠加）；16 个横截面各占一段，段间留缝，不做插值
    ov = np.zeros(ho.shape + (4,))
    hw_px = (F_POS[1] - F_POS[0]) * L / 2 * 0.86
    for k_, f in enumerate(F_POS):
        cc = c0 + f * L
        for col in range(int(np.ceil(cc - hw_px)), int(np.floor(cc + hw_px)) + 1):
            if 0 <= col < ho.shape[1]:
                body = ho[:, col] > 0
                ov[body, col, :3] = ATT_CMAP(w[k_] / w.max())[:3]
                ov[body, col, 3] = 0.72
    axm.imshow(ov, interpolation='nearest', aspect='auto',
               extent=(pct(-0.5), pct(ho.shape[1] - 0.5), ho.shape[0] - 0.5, -0.5), zorder=3)
    axm.text(0, r0 - 0.5, 'Head', ha='left', va='bottom', fontsize=7)
    xb = 100 - 200 / CELL / L * 100
    axm.plot([xb, 100], [r1 + 0.8] * 2, color=INK, lw=1.2, solid_capstyle='butt', zorder=5)
    axm.text((xb + 100) / 2, r1 + 1.6, '200 mm', ha='center', va='top', fontsize=7)
    axm.set_xlim(*xlim)
    axm.set_ylim(r1 + 5, r0 - 3)
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(False)
    cax = axmm(fig, 13 + wmap + 6, 76 + 2, 2.2, hmap - 4)
    sm = plt.cm.ScalarMappable(cmap=ATT_CMAP, norm=plt.Normalize(0, w.max()))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Attention weight\n(example sow)', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.5)

    axc = axmm(fig, 13, 76 + hmap + 3, wmap, 150 - (76 + hmap + 3) - 12)
    x = F_POS * 100
    axc.fill_between(x, seed_curves.min(0), seed_curves.max(0), color=C_MAIN, alpha=0.22, lw=0, zorder=2)
    axc.plot(x, mean_curve, color=C_MAIN, lw=1.5, zorder=3, label='All sows (range over five seeds)')
    axc.plot(x, w, color=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=3, label='Example sow')
    axc.axhline(1 / 16, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axc.text(xlim[1], 1 / 16 + 0.004, 'Uniform (1/16)', ha='right', va='bottom', fontsize=7)
    axc.axvline(60, color=REF, lw=0.7, ls=(0, (1, 1.3)), zorder=1)
    axc.text(59, 0.235, 'Virtual\ncaliper', ha='right', va='top', fontsize=7, linespacing=1.1)
    axc.set_xlim(*xlim)
    axc.set_ylim(0, 0.25)
    axc.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    axc.set_xlabel('Position along the body (% of length from head)')
    axc.set_ylabel('Attention weight')
    axc.legend(loc='upper left', handlelength=1.8, borderaxespad=0.1)
    clean(axc)
    k = int(np.argmax(mean_curve))
    log(f'Fig 3c: example {uid} frame {fr["frame_idx"]} (rule in v6i.example_sow); population attention peak '
        f'{mean_curve[k]:.4f} at {x[k]:.1f}% (runs/v6_anatomy/prof/result.json); example sow peak at '
        f'{F_POS[int(np.argmax(w))] * 100:.1f}%')

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 84.0, 1.0, 'b')
    plabel(fig, 0.5, 70.0, 'c')
    save(fig, 'fig3')


# ---------------------------------------------------------------- Fig 3
DR_ALL = re.compile(r'^\s*(?P<name>.+?)\s*\[all\s*\]\s*ΔR²\s*(?P<d>[+-]\d+\.\d+)\s*\((?P<lo>[+-]\d+\.\d+),\s*'
                    r'(?P<hi>[+-]\d+\.\d+)\)')


def last_dr(rel, name):
    """日志中某比较在全体母猪上的 ΔR²（取最后一次出现，适用于追加写入的日志）。"""
    hit = None
    with open(os.path.join(RUNS_ROOT, rel), encoding='utf-8') as f:
        for i, ln in enumerate(f, 1):
            mm = DR_ALL.match(ln)
            if mm and ' '.join(mm['name'].split()) == name:
                hit = (float(mm['lo']), float(mm['d']), float(mm['hi']), f'runs/{rel}:{i}')
    assert hit, f'{rel}: 找不到 {name} [all]'
    return hit


def sensitivity_grid():
    """v5b 日志「Farm A」段：angle_f{pos}_D{span} 与人工卡尺、与背膘的 Pearson r（全体母猪）。"""
    pat = re.compile(r'^\s*angle_f(?P<f>0\.\d+)_D(?P<d>\d+)\s+all cal (?P<cal>[+-]\d\.\d+)/\S+ bf (?P<bf>[+-]\d\.\d+)/')
    cal, bf, src = {}, {}, None
    with open(os.path.join(RUNS_ROOT, 'v5_caliper', 'log.txt'), encoding='utf-8') as f:
        in_a = False
        for i, ln in enumerate(f, 1):
            if ln.startswith('Farm A: r with manual caliper'):
                in_a, cal, bf, src = True, {}, {}, i          # 取最后一段 Farm A 表
                continue
            if in_a and ln.startswith('Farm B'):
                in_a = False
            mm = pat.match(ln) if in_a else None
            if mm:
                key = (float(mm['f']), int(mm['d']))
                cal[key], bf[key] = float(mm['cal']), float(mm['bf'])
    fs = sorted({k[0] for k in cal})
    ds = sorted({k[1] for k in cal})
    C = np.array([[cal[(f_, d)] for d in ds] for f_ in fs])
    B = np.array([[bf[(f_, d)] for d in ds] for f_ in fs])
    return fs, ds, C, B, src


def fig_compare():
    """正文图 2（2026-10-21 起；原图 3）：表示、视角与卡尺的比较。"""
    v6a = v6i.load('v6_camera_caliper_anatomy/result.json')['comparisons']
    v6h = v6i.load('v6_main_vs_caliper/result.json')['comparisons']

    def jd(src, key):
        d = (v6h[key]['dR2'] if src == 'v6h' else v6a[key]['all']['dR2'])
        js = 'runs/v6_main_vs_caliper/result.json' if src == 'v6h' else 'runs/v6_camera_caliper_anatomy/result.json'
        return (d[0], d[1], d[2], f'{js} "{key}"')

    items = [
        ('Image representation', None),
        ('Height map vs cropped depth map', (last_dr('v4_hmap/compare/log.txt', 'hmap'), False)),
        ('Anatomy-guided network vs ResNet-18', (jd('v6h', 'main vs ResNet-18 baseline (5 seeds) [all]'), True)),
        ('Second camera (reference: top-view height map)', None),
        ('Rear-oblique view alone', (last_dr('v6_dualview_hmap/log.txt', 'rear-oblique cropped (v3b sub)'), False)),
        ('Late fusion of both views', (last_dr('v6_dualview_hmap/log.txt', 'late fusion (mean of the two)'), False)),
        ('Dual-view network', (last_dr('v6_dualview_hmap/log.txt', 'dual-branch (this script)'), False)),
        ('Camera versus other sources', None),
        ('Main network vs caliper score', (jd('v6h', 'main vs caliper (C) [all]'), True)),
        ('Image score vs production records', (jd('v6a', 'camera vs production records (reference)'), False)),
        ('Adding a source', None),
        ('Caliper score added to image score', (jd('v6a', 'manual caliper over camera'), False)),
        ('Image score added to caliper score', (jd('v6a', 'camera over manual caliper'), False)),
        ('Image geometry added to image score', (jd('v6a', 'image geometry over CNN score'), False)),
        ('Body dimensions added to image score', (jd('v6a', 'whole-body dims over CNN score'), False)),
    ]
    for lab, d in items:
        if d:
            (lo, mid, hi, src), _ = d
            log(f'Fig 2a: {lab:<40s} ΔR² {mid:+.3f} ({lo:+.3f}, {hi:+.3f})  <- {src}')

    fig = new_fig(150)
    axa = axmm(fig, 66, 7, 66, 70)

    def draw(y, lab, dd):
        (lo, mid, hi, _), main = dd
        col = C_MAIN if main else C_ALL
        axa.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle='butt', zorder=2)
        axa.plot(mid, y, 'o', ms=4, mfc=col, mec='white', mew=0.6, zorder=3)
        axa.text(1.03, y, m.sfmt(mid), transform=axa.get_yaxis_transform(), ha='left', va='center', fontsize=8)
        axa.text(1.18, y, f'({m.sfmt(lo)}, {m.sfmt(hi)})', transform=axa.get_yaxis_transform(), ha='left',
                 va='center', fontsize=8)

    m.forest_rows(axa, items, (-0.17, 0.29), draw)
    axa.text(1.03, 0.55, 'ΔR²', transform=axa.get_yaxis_transform(), ha='left', va='bottom', fontsize=8)
    axa.text(1.18, 0.55, '95% CI', transform=axa.get_yaxis_transform(), ha='left', va='bottom', fontsize=8)
    axa.set_xticks([-0.1, 0, 0.1, 0.2])
    axa.set_xlabel('Difference in R² (95% CI), all sows')

    # (b) 卡尺读数–背膘
    axb = axmm(fig, 13, 92, 66, 46)
    rows = v6i.read_csv('v1b_caliper/preds.csv')
    cal = np.array([v6i.fnum(r['caliper']) if r['caliper_flag'] != 'implausible' else np.nan for r in rows])
    yy = np.array([float(r['y']) for r in rows])
    line = np.array([r['line'] for r in rows])
    st = m.by_level(axb, cal, yy, line, np.random.default_rng(0), 'b')
    m.line_legend(axb, st)
    axb.set_xlabel('Manual caliper score')
    axb.set_ylabel('Backfat, ultrasound (mm)')
    axb.set_xticks(np.arange(2.2, 5.01, 0.4))
    axb.set_xlim(2.05, 5.15)
    axb.set_ylim(np.floor(yy.min()) - 2, np.ceil(yy.max()) + 4)
    clean(axb)

    # (c) 虚拟卡尺敏感性热力图
    fs, ds, C, B, src = sensitivity_grid()
    log(f'Fig 2c: sensitivity grid from runs/v5_caliper/log.txt (Farm A block at line {src}); positions {fs}, '
        f'spans {ds}; r(manual caliper) {C.min():.3f}–{C.max():.3f}; r(backfat) {B.min():.3f}–{B.max():.3f}')
    vmin, vmax = 0.0, max(C.max(), 0.8)
    for j, (M, title) in enumerate(((C, 'r with manual caliper'), (B, 'r with backfat'))):
        ax = axmm(fig, 96 + j * 36, 96, 28, 38)
        ax.imshow(M, cmap=HEAT_CMAP, vmin=vmin, vmax=vmax, aspect='auto')
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if M[a, b] > 0.55 else INK)
        a0, b0 = fs.index(0.6), ds.index(150)
        ax.add_patch(Rectangle((b0 - 0.5, a0 - 0.5), 1, 1, fill=False, ec=INK, lw=1.2, zorder=4))
        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_yticks(range(len(fs)))
        ax.set_yticklabels([f'{f_ * 100:.0f}' for f_ in fs] if j == 0 else [])
        ax.set_title(title, fontsize=8, pad=3)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j == 0:
            ax.set_ylabel('Position (% of body length)')
        im = ax.images[0]
    cax = axmm(fig, 96 + 36 + 30, 96, 2.0, 38)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8])
    cb.set_label('Pearson r', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.5)
    m.ftext(fig, 96 + 32, 96 + 38 + 7.5, 'Half-span from the spine (mm)', ha='center', va='top')   # 同 2.4 节与 S4

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 0.5, 86.0, 'b')
    plabel(fig, 88.0, 86.0, 'c')
    save(fig, 'fig2')


# ---------------------------------------------------------------- Fig 4（一行三图）
def fig4():
    rel = v6i.load('v6_reliability_anatomy/result.json')
    fig = new_fig(60)
    icc = rel['icc']['ICC1']
    k90 = rel['icc']['frames_for_090']
    kk = np.linspace(1, 20, 400)
    sb = kk * icc / (1 + (kk - 1) * icc)
    kc = v6i.read_csv('v6_reliability_anatomy/a2_k_curve.csv')
    k = np.array([float(r['k']) for r in kc])
    r2m = np.array([float(r['R2_mean']) for r in kc])
    r2s = np.array([float(r['R2_sd']) for r in kc])
    log(f'Fig 4a: ICC1 {icc:.4f}, frames for 0.90 {k90:.3f}; R² k-curve '
        f'{dict(zip(k.astype(int).tolist(), np.round(r2m, 3).tolist()))} (runs/v6_reliability_anatomy)')
    ax_top = axmm(fig, 12, 5, 42, 23)
    ax_bot = axmm(fig, 12, 31, 42, 21)
    ax_top.plot(kk, sb, color=INK, lw=1.2, zorder=3)
    ax_top.plot([1], [icc], 'o', ms=3.2, mfc=INK, mec='white', mew=0.5, zorder=4)
    ax_top.text(1.8, icc, f'Single frame,\nICC {icc:.2f}', ha='left', va='center', fontsize=7, linespacing=1.1)
    ax_top.axhline(0.90, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_top.plot([k90], [0.90], 'o', ms=3.2, mfc=INK, mec='white', mew=0.5, zorder=4)
    ax_top.text(k90 + 0.8, 0.875, f'0.90 at {k90:.1f} frames', ha='left', va='top', fontsize=7)
    ax_top.set_ylim(0.6, 1.02)
    ax_top.set_yticks([0.6, 0.8, 1.0])
    ax_top.set_ylabel('Reliability')
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_bot.plot(k, r2m, color=C_MAIN, lw=1.3, zorder=3)
    ax_bot.errorbar(k, r2m, yerr=r2s, fmt='o', ms=3.0, mfc=C_MAIN, mec='white', mew=0.5, ecolor=C_MAIN,
                    elinewidth=0.8, capsize=0, zorder=4)
    ax_bot.set_ylim(0, 0.42)
    ax_bot.set_yticks([0, 0.2, 0.4])
    ax_bot.set_ylabel('R²')
    ax_bot.set_xlabel('Frames averaged per sow')
    for ax_ in (ax_top, ax_bot):
        ax_.set_xlim(0, 21)
        ax_.set_xticks([1, 5, 10, 15, 20])
        clean(ax_)
    fig.align_ylabels([ax_top, ax_bot])

    axb = axmm(fig, 69, 5, 46, 47)
    cf = rel['conformal']['alpha=0.1']
    groups = [('all', 'All'), ('DAN_parity1', 'Danish,\nparity 1'), ('NAM_multiparous', 'American,\nparity ≥ 2')]
    meths = [('marginal', 'Marginal', m.C_MARG, 'none'), ('mondrian', 'By line and parity', m.C_GRP, None)]
    for j, (mk, _, col, mfc) in enumerate(meths):
        for gi, (gk, _) in enumerate(groups):
            d = cf[mk][gk]
            xx = gi + (j - 0.5) * 0.32
            lo, hi = d['cov_ci']
            axb.plot([xx, xx], [lo, hi], color=col, lw=1.2, solid_capstyle='butt', zorder=2)
            axb.plot(xx, d['coverage'], 'o', ms=3.8, mfc='white' if mfc else col, mec=col, mew=1.0, zorder=3)
            axb.text(xx, lo - 0.005, f'{d["width_mean"]:.1f}', ha='center', va='top', fontsize=6.5)
            log(f'Fig 4b: {mk:<8s} {gk:<15s} coverage {d["coverage"]:.4f} CI ({lo:.4f}, {hi:.4f}) width '
                f'{d["width_mean"]:.3f} mm')
    hs = [Line2D([], [], color=col, marker='o', ms=3.8, mfc='white' if mfc else col, mec=col, mew=1.0, lw=1.2,
                 label=mn) for _, mn, col, mfc in meths]
    axb.axhline(0.90, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axb.legend(handles=hs, loc='upper left', ncol=1, handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)
    axb.set_xticks(range(3))
    axb.set_xticklabels([g[1] for g in groups])
    axb.set_xlim(-0.55, 2.55)
    axb.set_ylim(0.78, 1.0)
    axb.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axb.set_ylabel('Coverage of 90% interval')
    clean(axb)

    axc = axmm(fig, 128, 5, 44, 47)
    tc = v6i.read_csv('v6_reliability_anatomy/triage_curve.csv')
    frc = np.array([float(r['frac']) for r in tc]) * 100
    series = [('random', 'Random', REF, 1.0, (0, (3, 2))), ('frame_sd', 'Frame or seed SD', LIGHT, 0.8, '-'),
              ('seed_sd', None, LIGHT, 0.8, '-'), ('sigma_hat', 'Fitted error scale', C_MAIN, 1.5, '-'),
              ('oracle', 'Oracle (true error)', INK, 1.3, '-')]
    for key, name, col, lw, ls in series:
        v = np.array([float(r[f'{key}_mae_kept']) for r in tc])
        axc.plot(frc, v, color=col, lw=lw, ls=ls, zorder=3, label=name or '_')
    axc.legend(loc='lower left', handlelength=1.8, labelspacing=0.25, borderaxespad=0.1)
    o30 = [float(r['oracle_mae_kept']) for r in tc if abs(float(r['frac']) - 0.3) < 1e-9][0]
    log(f'Fig 4c: oracle MAE kept at 30% = {o30:.4f} mm; start MAE {float(tc[0]["random_mae_kept"]):.4f}')
    axc.set_xlim(0, 52)
    axc.set_ylim(0, 2.8)
    axc.set_xticks([0, 10, 20, 30, 40, 50])
    axc.set_xlabel('Sows re-measured (%)')
    axc.set_ylabel('MAE of remaining sows (mm)')
    clean(axc)

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 58.5, 1.0, 'b')
    plabel(fig, 117.5, 1.0, 'c')
    save(fig, 'fig4')


# ---------------------------------------------------------------- Fig 1（现场照片 + v6m 的处理流程与网络结构）
PHOTO_DIR = r'F:/Postgraduate/Lab/十月论文/figures_v3/photos'
# (文件, 裁剪框 (x0, y0, x1, y1), 标注 [(文字, 指向点, 文字位置, 对齐)])；坐标均为原图像素
PHOTOS = [
    ('platform_wechat.webp', (0, 400, 1091, 1150), [
        ('Top-view camera', (60, 535), (150, 610), 'left'),
        ('Arm, 1,250 mm', (420, 432), (420, 480), 'center'),
        ('Rear-oblique camera\n(60° from vertical)', (712, 600), (560, 700), 'right'),
        ('Mobile frame', (790, 1000), (690, 1080), 'right'),
    ]),
    ('ultrasound_wechat.jpg', (250, 330, 1706, 1279), [
        ('Caliper scoring, then\nB-mode ultrasound at P2', (981, 704), (1250, 470), 'center'),   # 顺序同 2.1 节
        ('Recording', (420, 560), (420, 405), 'center'),
    ]),
]


def photo_panel(fig, x, y, h, fn, box, labels):
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(PHOTO_DIR, fn)).convert('RGB').crop(box))
    w = h * im.shape[1] / im.shape[0]
    ax = axmm(fig, x, y, w, h)
    ax.imshow(im, interpolation='lanczos')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for txt, (px, py), (tx, ty), ha in labels:
        ax.annotate(txt, (px - box[0], py - box[1]), (tx - box[0], ty - box[1]), ha=ha, va='center', fontsize=7,
                    color=INK, linespacing=1.1,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.88),
                    arrowprops=dict(arrowstyle='-', color='white', lw=1.0, shrinkA=0, shrinkB=2))
    return w


def fig1():
    """v6m.fig1 原样绘制处理流程（c）与网络结构（d），整体下移 OFF mm，上方放两张现场照片（a、b）。"""
    PH = 52.0                                   # 照片行高度（mm）
    OFF = PH + 9.0
    names = ('new_fig', 'axmm', 'plabel', 'ftext', 'save', 'FancyArrowPatch')
    orig = {k: getattr(m, k) for k in names}
    hold = {}

    def new_fig2(h):
        hold['fig'] = orig['new_fig'](h + OFF)
        return hold['fig']

    def fap2(posA, posB, **kw):                 # v6m.fig1 内 farrow 用图坐标画箭头，同样下移
        f = hold.get('fig')
        if f is not None and kw.get('transform') is f.transFigure:
            d = OFF / f._h_mm
            posA, posB = (posA[0], posA[1] - d), (posB[0], posB[1] - d)
        return orig['FancyArrowPatch'](posA, posB, **kw)

    m.new_fig = new_fig2
    m.axmm = lambda fig, x, y, w, h, **kw: orig['axmm'](fig, x, y + OFF, w, h, **kw)
    m.plabel = lambda fig, x, y, s: orig['plabel'](fig, x, y + OFF, {'a': 'c', 'b': 'd'}.get(s, s))
    m.ftext = lambda fig, x, y, s, **kw: orig['ftext'](fig, x, y + OFF, s, **kw)
    m.save = lambda fig, name: hold.setdefault('done', fig)
    m.FancyArrowPatch = fap2
    try:
        m.fig1()
    finally:
        for k in names:
            setattr(m, k, orig[k])
    fig = hold['done']
    x = 4.0
    for k, (fn, box, labels) in enumerate(PHOTOS):
        w = photo_panel(fig, x, 6.0, PH, fn, box, labels)
        plabel(fig, x - 3.5, 1.0, 'ab'[k])
        log(f'Fig 1{"ab"[k]}: photo {fn} crop {box}, {w:.1f} × {PH:.1f} mm')
        x += w + 7.0
    save(fig, 'fig1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', nargs='+', type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    log(f'== v6p_figures.py --fig {" ".join(map(str, a.fig))} | runs {RUNS_ROOT} | out {m.OUT}')
    for n in a.fig:
        {1: fig1, 2: fig_compare, 3: fig_accuracy, 4: fig4}[n]()   # 2026-10-21：图 2 与图 3 对调，按正文首次引用顺序


if __name__ == '__main__':
    main()
