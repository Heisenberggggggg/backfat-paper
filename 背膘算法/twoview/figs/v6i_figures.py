# -*- coding: utf-8 -*-
"""
v6i_figures.py — 稿件 v2 的 Fig 1–4（2026-10-16）

  Fig 1 流程与网络结构：俯视彩色帧 + 分割掩膜 → 物理尺度高度图（16 条横截面位置）→ 3 个部位的横截面示例；网络结构示意
  Fig 2 解剖引导网络：(a) 横截面位置注意力沿体长（5 种子均值 + 种子间范围）；(b) 典型高度图叠加该猪的注意力；
        (c) 表示与视角各配置 R²（表 2）
  Fig 3 体况卡尺：(a) 虚拟 vs 人工卡尺；(b) 卡尺读数–背膘分品系；(c) 相机、卡尺与组合的 ΔR² 森林图（新主模型）
  Fig 4 可靠性：(a) 帧数–信度 / R²；(b) 90% 保形区间覆盖率（边际 vs 分组校准）；(c) 复测分流；
        (d) 按 B 超背膘分段的误差（预测 − B 超）——回归向均值（DECISIONS 2026-10-16：借鉴吕年珅稿图 9）

来源 / 背景
  稿件 v2「图表规划」；10 交接 §6 第 2 项。风格沿用 figs/fig3_draft.py（该文件保留不动）。
  数字全部读自已有 runs/* 输出，不重新训练；表 2 中无 JSON 的行按稿件【来源】所列 log 手抄，见 TABLE2 注释。
期刊要求（gComm 模板）：Times New Roman 8–10 pt，RGB，≥ 300 dpi，双栏宽 174 mm。
颜色：dataviz 参考调色板前两槽（蓝 #2a78d6、橙 #eb6834；validate_palette.js 浅色模式全部 PASS）；
      高度图用单色灰度序列；文字用墨色，不用系列色。
输出：F:/Postgraduate/Lab/十月论文/figures/fig{1,2,3,4}.{png,pdf,svg}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe figs/v6i_figures.py --fig 2 3 4
      （Fig 1 需要 ultralytics + 原始录像盘 H:，用于对示例帧重新做 YOLO + FastSAM 分割）
"""
import argparse
import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.normpath(os.path.join(HERE, '..'))
RUNS = os.path.join(TW, 'runs')
DM_HMAP = os.path.normpath(os.path.join(TW, '..', 'yolo+FastSAM', 'hmap_dataset'))
SEG_DIR = os.path.normpath(os.path.join(TW, '..', 'yolo+FastSAM'))
INDEX = os.path.join(TW, 'index_v2.csv')
OUT = r'F:/Postgraduate/Lab/十月论文/figures'
LOG = os.path.join(RUNS, 'v6i_figures', 'log.txt')

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e4e3df'
LINE_COL = {'DAN': BLUE, 'NAM': ORANGE}
LINE_NAME = {'DAN': 'Danish line', 'NAM': 'American line'}
F_POS = np.linspace(0.10, 0.95, 16)     # 同 v6f
H_MAX, CELL = 1200.0, 7.8125           # 同 v4b / v5b
MM = 1 / 25.4

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'mathtext.fontset': 'stix',
})


# ---------------------------------------------------------------- 通用
def load(path):
    with open(os.path.join(RUNS, path), encoding='utf-8') as f:
        return json.load(f)


def read_csv(path):
    with open(path if os.path.isabs(path) else os.path.join(RUNS, path), encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def clean(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def panel_label(fig, ax, s, dx=0.055, dy=0.012):
    bb = ax.get_position()
    fig.text(max(bb.x0 - dx, 0.005), bb.y1 + dy, s, fontsize=10, color=INK, ha='left',
             va='bottom')


def log(s):
    """打印并追加到 runs/v6i_figures/log.txt（图中数字的追溯依据）。"""
    print(s, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f'[{time.strftime("%F %T")}] {s.strip()}' + chr(10))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=400, facecolor='white')
    # 期刊首选格式：TIFF（LZW 压缩）、JPG；彩图 RGB >= 300 dpi
    fig.savefig(os.path.join(OUT, f'{name}.tif'), dpi=400, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(os.path.join(OUT, f'{name}.jpg'), dpi=400, facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)
    log(f'saved -> {os.path.join(OUT, name)}')


def setup_rows():
    """test_preds.csv 的行序 = v6f 的 uids 顺序 = _seed_ckpt 中数组的下标。"""
    rows = read_csv('v6_anatomy/prof/test_preds.csv')
    meta = {r['uid']: r for r in read_csv('v1_tabular/preds.csv')}
    return rows, meta


def seed_attention():
    """每个种子：(n_pigs, 16) 的猪级位置注意力（inner 选 epoch，测试折）。"""
    out = []
    for s in (42, 43, 44, 45, 46):
        z = np.load(os.path.join(RUNS, 'v6_anatomy', 'prof', '_seed_ckpt', f'seed{s}.npz'), allow_pickle=True)
        out.append(z['store'].item()['posw_inner'])
    return np.stack(out)


def col_top(h, m):
    hm = np.where(m, h, np.nan)
    with np.errstate(all='ignore'):
        return np.nanpercentile(hm, 95, axis=0)


def orient(h):
    """同 v6f.frame_profiles 的头尾判定（含退路）：返回头在左的高度图与体长两端列 c0, c1；失败返回 None。"""
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
    n20 = max(int(0.2 * (c1 - c0)), 3)
    with np.errstate(all='ignore'):
        t_lo = np.nanmedian(np.where(good[c0:c0 + n20], top[c0:c0 + n20], np.nan))
        t_hi = np.nanmedian(np.where(good[c1 - n20:c1 + 1], top[c1 - n20:c1 + 1], np.nan))
        if np.isnan(t_lo) or np.isnan(t_hi):
            t_lo, t_hi = np.nanmedian(top[c0:c0 + n20]), np.nanmedian(top[c1 - n20:c1 + 1])
    if np.isnan(t_lo) or np.isnan(t_hi):
        return None
    if t_lo > t_hi:
        h, good = h[:, ::-1], good[::-1]
        c0, c1 = h.shape[1] - 1 - c1, h.shape[1] - 1 - c0
    return h, c0, c1, good


def transverse(h, c, good):
    """列 c 附近 ±2 列（未遮挡）的逐行中位横截面，以脊线为中心；返回 (横向 mm, 相对脊线高度 mm)。"""
    m = h > 0
    for off in (0, -1, 1, -2, 2, -3, 3):
        idx = np.arange(c + off - 2, c + off + 3)
        idx = idx[(idx >= 0) & (idx < h.shape[1])]
        idx = idx[good[idx]]
        if len(idx) >= 2:
            break
    with np.errstate(all='ignore'):
        prof = np.nanmedian(np.where(m[:, idx], h[:, idx], np.nan), axis=1)
    rows = np.where(~np.isnan(prof))[0]
    mid = (rows[0] + rows[-1]) // 2
    cand = np.arange(max(mid - 5, 0), min(mid + 6, len(prof)))
    cand = cand[~np.isnan(prof[cand])]
    sp = cand[np.argmax(prof[cand])]
    x = (np.arange(len(prof)) - sp) * CELL
    return x, prof - prof[sp]


def qc_frames(uid):
    """与 v4b.load_hmaps 相同的帧质控（面积 ≥ 80%、体长 ≥ 85% 中位数）。"""
    rs = [r for r in read_csv(os.path.join(DM_HMAP, 'metadata.csv')) if r['uid'] == uid and r['ok'] == '1']
    ma = np.median([float(r['area_cm2']) for r in rs])
    ml = np.median([float(r['length_mm']) for r in rs])
    return [r for r in rs if float(r['area_cm2']) >= 0.8 * ma and float(r['length_mm']) >= 0.85 * ml], ma


def example_sow(rows, meta, att):
    """示例猪（预先设定的规则，非挑结果）：丹系头胎、背膘 16–19 mm、注意力最接近总体均值者；取面积最接近中位数的帧。"""
    mean_w = np.nanmean(att, (0, 1))
    pig_w = np.nanmean(att, 0)
    best, bd = None, np.inf
    for i, r in enumerate(rows):
        mt = meta.get(r['uid'], {})
        if mt.get('line') != 'DAN' or fnum(mt.get('parity')) != 1 or not 16 <= float(r['y']) <= 19:
            continue
        if np.isnan(pig_w[i]).any():
            continue
        d = np.abs(pig_w[i] - mean_w).sum()
        if d < bd:
            best, bd = i, d
    uid = rows[best]['uid']
    fr, ma = qc_frames(uid)

    def score(r):
        """被栏杆遮挡的体长列越少越好；并列时取面积最接近中位数的帧。"""
        o = orient(np.load(os.path.join(DM_HMAP, r['frame_path'])).astype(np.float32))
        if o is None:
            return (1.0, 0.0)
        _, c0, c1, good = o
        return (1 - good[c0:c1 + 1].mean(), abs(float(r['area_cm2']) - ma))

    f = min(fr, key=score)
    h = np.load(os.path.join(DM_HMAP, f['frame_path'])).astype(np.float32)
    return best, uid, f, h


def show_hmap(ax, h, c0, c1, weights=None, scalebar=True):
    hm = np.ma.masked_where(h <= 0, h)
    rows_on = np.where((h > 0).any(1))[0]
    r0, r1 = max(rows_on[0] - 4, 0), min(rows_on[-1] + 4, h.shape[0] - 1)
    im = ax.imshow(hm, cmap='Greys', vmin=300, vmax=1000, interpolation='nearest', aspect='equal')
    ax.set_ylim(r1, r0)
    ax.set_xlim(max(c0 - 6, 0), min(c1 + 6, h.shape[1] - 1))
    L = c1 - c0
    for k, f in enumerate(F_POS):
        c = c0 + f * L
        if weights is None:
            ax.plot([c, c], [r0 + 1, r1 - 1], color=ORANGE, lw=0.6, alpha=0.9)
        else:
            w = weights[k] / np.nanmax(weights)
            ax.plot([c, c], [r0 + 1, r1 - 1], color=ORANGE, lw=0.4 + 2.6 * w, alpha=0.25 + 0.75 * w,
                    solid_capstyle='butt')
    if scalebar:
        xb = ax.get_xlim()[1] - 4 - 200 / CELL
        ax.plot([xb, xb + 200 / CELL], [r1 - 2, r1 - 2], color=INK, lw=1.4, solid_capstyle='butt')
        ax.text(xb + 100 / CELL, r1 - 4, '200 mm', ha='center', va='bottom', fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


# ---------------------------------------------------------------- Fig 2
# 表 2 数值（R²，全部 663 条）：稿件 Table 2；来源 runs/v4_hmap/compare/log.txt（v4c）、runs/v3_views/compare/log.txt（v3c）、
# runs/v6_dualview_hmap/log.txt（v6b）；五种子两行读自 runs/v6_main_vs_caliper/result.json（v6h）
TABLE2 = [
    ('Image representation (top view)', None),
    ('Cropped depth map, ResNet-18', 0.309),
    ('Height map, ResNet-18', 0.378),
    ('Height map, ResNet-18 (five seeds)', 'v6h:base'),
    ('Height map, anatomy-guided (five seeds)', 'v6h:main'),
    ('Camera views (baseline network)', None),
    ('Rear-oblique, cropped depth map', 0.261),
    ('Top + rear-oblique, cropped, late fusion', 0.317),
    ('Top + rear-oblique, cropped, dual-view', 0.331),
    ('Top height map + rear-oblique, late fusion', 0.365),
    ('Top height map + rear-oblique, dual-view', 0.382),
]


def fig2():
    rows, meta = setup_rows()
    att = seed_attention()                     # (5, n, 16)
    seed_curves = np.nanmean(att, 1)           # (5, 16)
    mean_curve = np.nanmean(att, (0, 1))
    res = load('v6_anatomy/prof/result.json')
    assert np.allclose(mean_curve, res['position_attention_mean'], atol=1e-6), '与 result.json 不一致'

    fig = plt.figure(figsize=(174 * MM, 118 * MM))
    axa = fig.add_axes([0.075, 0.585, 0.40, 0.37])
    axb = fig.add_axes([0.545, 0.60, 0.44, 0.36])
    axc = fig.add_axes([0.375, 0.075, 0.42, 0.40])

    # (a)
    x = F_POS * 100
    axa.fill_between(x, seed_curves.min(0), seed_curves.max(0), color=BLUE, alpha=0.18, lw=0, zorder=2)
    axa.plot(x, mean_curve, color=BLUE, lw=2, zorder=3, solid_capstyle='round')
    axa.plot(x, mean_curve, 'o', ms=4, mfc=BLUE, mec='white', mew=0.8, zorder=4)
    axa.axhline(1 / 16, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axa.text(7, 1 / 16 + 0.004, 'Uniform weighting', ha='left', va='bottom', fontsize=8, color=INK2)
    axa.axvline(60, color=ORANGE, lw=1, ls=(0, (1, 1.5)), zorder=1)
    axa.text(58.5, 0.235, 'Virtual caliper\n(last rib, 60%)', ha='right', va='top', fontsize=8, color=INK)
    k = int(np.argmax(mean_curve))
    axa.text(x[k] + 2.5, mean_curve[k] + 0.004, f'{mean_curve[k]:.2f}', ha='left', va='bottom', fontsize=8,
             color=INK)
    axa.text(12, 0.135, 'Mean of 5 seeds\n(band: range across seeds)', ha='left', va='top', fontsize=8, color=INK2)
    axa.set_xlim(5, 100)
    axa.set_ylim(0, 0.25)
    axa.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
    axa.set_xlabel('Position along the body (% of body length from the head)')
    axa.set_ylabel('Profile attention weight')
    axa.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    clean(axa)

    # (b)
    i, uid, fr, h = example_sow(rows, meta, att)
    ho, c0, c1, _ = orient(h)
    w = np.nanmean(att[:, i], 0)
    im = show_hmap(axb, ho, c0, c1, weights=w)
    axb.text(axb.get_xlim()[0] + 1, axb.get_ylim()[1] - 1, 'Head', ha='left', va='top', fontsize=8.5, color=INK)
    cax = fig.add_axes([0.62, 0.565, 0.22, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Height above floor (mm)', fontsize=8.5, labelpad=1)
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    log(f'   Fig 2b example: {uid} frame {fr["frame_idx"]} backfat {rows[i]["y"]}; '
          f'attention peak {F_POS[int(np.argmax(w))]:.2f}')

    # (c)
    v6h = load('v6_main_vs_caliper/result.json')['table']
    look = {'v6h:main': v6h['main (anatomy-guided, 5 seeds)']['all']['R2']}
    look['v6h:base'] = [v['all']['R2'] for k_, v in v6h.items() if 'ResNet' in k_ and '5' in k_][0]
    y = 0
    yt, yl = [], []
    for lab, v in TABLE2:
        if v is None:
            axc.text(-0.02, y, lab, transform=axc.get_yaxis_transform(), ha='right', va='center', fontsize=8.5,
                     style='italic', color=INK)
        else:
            r2 = look[v] if isinstance(v, str) else v
            main = v == 'v6h:main'
            col = ORANGE if main else BLUE
            axc.plot([0.2, r2], [y, y], color=col, lw=1, alpha=0.35, zorder=2)
            axc.plot(r2, y, 'o', ms=5, mfc=col, mec='white', mew=0.8, zorder=3)
            axc.text(r2 + 0.006, y, f'{r2:.3f}', ha='left', va='center', fontsize=8, color=INK2)
            yt.append(y)
            yl.append(lab)
        y -= 1
    axc.set_yticks(yt)
    axc.set_yticklabels(yl)
    axc.tick_params(axis='y', length=0)
    axc.set_xlim(0.2, 0.42)
    axc.set_ylim(y + 0.4, 0.6)
    axc.set_xlabel('R² (out-of-fold, 663 records)')
    axc.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    axc.spines['left'].set_visible(False)
    clean(axc)

    panel_label(fig, axa, 'a', dx=0.07)
    panel_label(fig, axb, 'b', dx=0.04)
    fig.text(0.005, axc.get_position().y1 + 0.012, 'c', fontsize=10, color=INK)
    save(fig, 'fig2')


# ---------------------------------------------------------------- Fig 3
def scatter_by_line(ax, xs, ys, lines, jitter, rng):
    out = []
    for ln in ('DAN', 'NAM'):
        m = (lines == ln) & ~np.isnan(xs) & ~np.isnan(ys)
        xj = xs[m] + rng.uniform(-jitter, jitter, m.sum())
        ax.scatter(xj, ys[m], s=7, color=LINE_COL[ln], alpha=0.45, linewidths=0, zorder=2)
        b1, b0 = np.polyfit(xs[m], ys[m], 1)
        xl = np.array([np.percentile(xs[m], 2), np.percentile(xs[m], 98)])
        ax.plot(xl, b0 + b1 * xl, color=LINE_COL[ln], lw=1.8, zorder=3, solid_capstyle='round')
        out.append((ln, np.corrcoef(xs[m], ys[m])[0, 1], m.sum()))
    return out


def legend_lines(ax, stats, extra=None):
    for k, (ln, r, n) in enumerate(stats):
        yy = 0.96 - 0.085 * k
        ax.plot([0.04], [yy], transform=ax.transAxes, marker='o', ms=4, color=LINE_COL[ln], clip_on=False)
        ax.text(0.07, yy, f'{LINE_NAME[ln]} (n = {n}), r = {r:.2f}', transform=ax.transAxes, ha='left',
                va='center', fontsize=8.5, color=INK)
    if extra:
        ax.text(0.07, 0.96 - 0.085 * len(stats), extra, transform=ax.transAxes, ha='left', va='center',
                fontsize=8.5, color=INK2)


def fig3():
    fig = plt.figure(figsize=(174 * MM, 130 * MM))
    axa = fig.add_axes([0.075, 0.585, 0.38, 0.37])
    axb = fig.add_axes([0.585, 0.585, 0.395, 0.37])
    axc = fig.add_axes([0.40, 0.075, 0.47, 0.40])

    # (a) 虚拟 vs 人工卡尺（663 头分析集）
    keep = {r['uid'] for r in read_csv('v1_tabular/preds.csv')}
    rows = [r for r in read_csv('v5_caliper/features.csv') if r['farm'] == 'FMH' and r['uid'] in keep]
    cal = np.array([fnum(r['caliper']) for r in rows])
    ang = np.array([fnum(r['angle_f0.60_D150']) for r in rows])
    line = np.array([r['line'] for r in rows])
    stats = scatter_by_line(axa, cal, ang, line, 0.04, np.random.default_rng(1))
    ok = ~np.isnan(cal) & ~np.isnan(ang)
    legend_lines(axa, stats, f'All sows, r = {np.corrcoef(cal[ok], ang[ok])[0, 1]:.2f}')
    axa.set_xlabel('Manual caliper score')
    axa.set_ylabel('Virtual caliper angle (°)')
    axa.grid(color=GRID, lw=0.5, zorder=0)
    clean(axa)

    # (b) 卡尺读数–背膘
    rows = read_csv('v1b_caliper/preds.csv')
    cal = np.array([fnum(r['caliper']) if r['caliper_flag'] != 'implausible' else np.nan for r in rows])
    y = np.array([float(r['y']) for r in rows])
    line = np.array([r['line'] for r in rows])
    stats = scatter_by_line(axb, cal, y, line, 0.04, np.random.default_rng(0))
    legend_lines(axb, stats)
    axb.set_xlabel('Manual caliper score')
    axb.set_ylabel('Backfat, ultrasound (mm)')
    axb.grid(color=GRID, lw=0.5, zorder=0)
    clean(axb)

    # (c) ΔR² 森林图（新主模型；v6h 五种子 + v6a 嵌套图像分数）
    v6a = load('v6_camera_caliper_anatomy/result.json')['comparisons']
    v6h = load('v6_main_vs_caliper/result.json')['comparisons']
    items = [
        ('Camera versus other single sources', None),
        ('Main network vs caliper score', v6h['main vs caliper (C) [all]']['dR2']),
        ('Image score vs caliper score', v6a['camera vs manual caliper (stacked image score)']['all']['dR2']),
        ('Image score vs production records', v6a['camera vs production records (reference)']['all']['dR2']),
        ('Image geometry vs caliper score', v6a['image geometry vs manual caliper']['all']['dR2']),
        ('Increment from adding a source', None),
        ('Image score added to caliper', v6a['camera over manual caliper']['all']['dR2']),
        ('Caliper added to image score', v6a['manual caliper over camera']['all']['dR2']),
        ('Image geometry added to image score', v6a['image geometry over CNN score']['all']['dR2']),
        ('Body dimensions added to image score', v6a['whole-body dims over CNN score']['all']['dR2']),
        ('Image geometry added to caliper + image score', v6a['image geometry over caliper+camera']['all']['dR2']),
    ]
    yy = 0
    yt, yl = [], []
    for lab, d in items:
        if d is None:
            axc.text(-0.02, yy, lab, transform=axc.get_yaxis_transform(), ha='right', va='center', fontsize=8.5,
                     style='italic', color=INK)
        else:
            lo, mid, hi = d
            axc.plot([lo, hi], [yy, yy], color=BLUE, lw=1.6, solid_capstyle='round', zorder=2)
            axc.plot(mid, yy, 'o', ms=4.5, mfc=BLUE, mec='white', mew=0.8, zorder=3)
            axc.text(0.335, yy, f'{mid:+.3f}', ha='right', va='center', fontsize=8, color=INK2)
            yt.append(yy)
            yl.append(lab)
        yy -= 1
    axc.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axc.set_yticks(yt)
    axc.set_yticklabels(yl)
    axc.tick_params(axis='y', length=0)
    axc.set_xlim(-0.05, 0.34)
    axc.set_ylim(yy + 0.4, 0.6)
    axc.set_xlabel('Difference in R² (95% CI, bootstrap over sows)')
    axc.grid(axis='x', color=GRID, lw=0.5, zorder=0)
    axc.spines['left'].set_visible(False)
    clean(axc)

    panel_label(fig, axa, 'a', dx=0.07)
    panel_label(fig, axb, 'b', dx=0.07)
    fig.text(0.005, axc.get_position().y1 + 0.012, 'c', fontsize=10, color=INK)
    save(fig, 'fig3')


# ---------------------------------------------------------------- Fig 4
BINS = [(None, 13, '≤ 13'), (14, 16, '14–16'), (17, 19, '17–19'), (20, 22, '20–22'), (23, 25, '23–25'),
        (26, None, '≥ 26')]


def fig4():
    rel = load('v6_reliability_anatomy/result.json')
    fig = plt.figure(figsize=(174 * MM, 126 * MM))
    axa = fig.add_axes([0.075, 0.60, 0.38, 0.355])
    axb = fig.add_axes([0.585, 0.60, 0.395, 0.355])
    axc = fig.add_axes([0.075, 0.115, 0.38, 0.355])
    axd = fig.add_axes([0.585, 0.115, 0.395, 0.355])

    # (a) 帧数：Spearman–Brown 信度（由单帧 ICC）与经验 R²（单网络，随机抽 k 帧）
    icc = rel['icc']['ICC1']
    kk = np.arange(1, 21)
    sb = kk * icc / (1 + (kk - 1) * icc)
    kc = read_csv('v6_reliability_anatomy/a2_k_curve.csv')
    k = np.array([float(r['k']) for r in kc])
    r2m = np.array([float(r['R2_mean']) for r in kc])
    r2s = np.array([float(r['R2_sd']) for r in kc])
    axa.plot(kk, sb, color=ORANGE, lw=2, zorder=3)
    axa.plot(k, r2m, color=BLUE, lw=2, zorder=3)
    axa.errorbar(k, r2m, yerr=r2s, fmt='o', ms=4, mfc=BLUE, mec='white', mew=0.8, ecolor=BLUE, elinewidth=1,
                 capsize=0, zorder=4)
    k90 = rel['icc']['frames_for_090']
    axa.plot([k90], [0.90], 'o', ms=4.5, mfc=ORANGE, mec='white', mew=0.8, zorder=4)
    axa.text(k90 + 0.6, 0.88, f'0.90 at {k90:.1f} frames', ha='left', va='top', fontsize=8, color=INK)
    axa.text(20, sb[-1] + 0.03, f'Reliability of the mean\n(single-frame ICC {icc:.2f})', ha='right',
             va='bottom', fontsize=8.5, color=INK)
    axa.text(20, r2m[-1] + 0.05, 'R² against ultrasound', ha='right', va='bottom', fontsize=8.5, color=INK)
    axa.set_xlim(0, 21)
    axa.set_ylim(0, 1.08)
    axa.set_xticks([1, 5, 10, 15, 20])
    axa.set_xlabel('Frames averaged per sow')
    axa.set_ylabel('Reliability or R²')
    axa.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    clean(axa)

    # (b) 90% 保形区间覆盖率
    cf = rel['conformal']['alpha=0.1']
    groups = [('all', 'All sows\n\n(n = 663)'), ('DAN_parity1', 'Danish line,\nparity 1\n(n = 386)'),
              ('NAM_multiparous', 'American line,\nparity ≥ 2\n(n = 189)')]
    meths = [('marginal', 'Marginal', BLUE), ('mondrian', 'By line and parity', ORANGE)]
    xg = np.arange(len(groups))
    for j, (mk, mn, col) in enumerate(meths):
        for g, (gk, _) in enumerate(groups):
            d = cf[mk][gk]
            xx = g + (j - 0.5) * 0.28
            lo, hi = d['cov_ci']
            axb.plot([xx, xx], [lo, hi], color=col, lw=1.6, solid_capstyle='round', zorder=2)
            axb.plot(xx, d['coverage'], 'o', ms=5, mfc=col, mec='white', mew=0.8, zorder=3)
            axb.text(xx, lo - 0.006, f'{d["width_mean"]:.1f}', ha='center', va='top', fontsize=8, color=INK2)
        axb.plot([], [], 'o', ms=5, mfc=col, mec='white', label=mn)
    axb.axhline(0.90, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1, label='Nominal 0.90')
    axb.text(-0.45, 0.785, 'Numbers: mean interval width (mm)', ha='left', va='bottom', fontsize=8, color=INK2)
    axb.legend(loc='upper left', frameon=False, handletextpad=0.4, borderaxespad=0.1, ncol=2,
               columnspacing=1.0, handlelength=1.6)
    axb.set_xticks(xg)
    axb.set_xticklabels([g[1] for g in groups])
    axb.set_xlim(-0.55, 2.6)
    axb.set_ylim(0.78, 1.0)
    axb.set_ylabel('Coverage of 90% interval (95% CI)')
    axb.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    clean(axb)

    # (c) 复测分流：复测一部分猪（改用 B 超）后，其余猪的 MAE
    tc = read_csv('v6_reliability_anatomy/triage_curve.csv')
    fr = np.array([float(r['frac']) for r in tc]) * 100
    series = [('random', 'Random', INK2, 1.2, (0, (3, 2))), ('sigma_hat', 'Fitted error scale', BLUE, 2, '-'),
              ('frame_sd', 'Frame or seed SD', MUTED, 0.8, '-'), ('seed_sd', None, MUTED, 0.8, '-'),
              ('oracle', 'Oracle (true error)', ORANGE, 2, '-')]
    for key, name, col, lw, ls in series:
        v = np.array([float(r[f'{key}_mae_kept']) for r in tc])
        axc.plot(fr, v, color=col, lw=lw, ls=ls, zorder=3 if col != MUTED else 2, label=name or '_')
    axc.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), frameon=False, handlelength=2.2, fontsize=8)
    o30 = [float(r['oracle_mae_kept']) for r in tc if abs(float(r['frac']) - 0.3) < 1e-9][0]
    axc.plot([30], [o30], 'o', ms=4.5, mfc=ORANGE, mec='white', mew=0.8, zorder=4)
    axc.text(31.5, o30 + 0.06, f'{o30:.2f} mm', ha='left', va='bottom', fontsize=8, color=INK)
    axc.set_xlim(0, 52)
    axc.set_ylim(0, 2.8)
    axc.set_xticks([0, 10, 20, 30, 40, 50])
    axc.set_xlabel('Sows re-measured by ultrasound (%)')
    axc.set_ylabel('MAE of remaining sows (mm)')
    axc.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    clean(axc)

    # (d) 按 B 超背膘分段的误差（主模型 5 种子，inner）
    rows = read_csv('v6_anatomy/prof/test_preds.csv')
    y = np.array([float(r['y']) for r in rows])
    e = np.array([float(r['pred_inner']) for r in rows]) - y
    data, labs, maes, ns = [], [], [], []
    for lo, hi, lab in BINS:
        m = np.ones(len(y), bool)
        if lo is not None:
            m &= y >= lo
        if hi is not None:
            m &= y <= hi
        data.append(e[m])
        labs.append(lab)
        maes.append(np.abs(e[m]).mean())
        ns.append(int(m.sum()))
    xb = np.arange(len(data))
    bp = axd.boxplot(data, positions=xb, widths=0.5, patch_artist=True, showfliers=True,
                     medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=BLUE, lw=0.9),
                     capprops=dict(color=BLUE, lw=0.9),
                     flierprops=dict(marker='o', ms=2.5, mfc=BLUE, mec='none', alpha=0.5))
    for b in bp['boxes']:
        b.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE, lw=0.9)
    axd.axhline(0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for x_, mae, n in zip(xb, maes, ns):
        axd.text(x_, 14.4, f'{mae:.1f}', ha='center', va='top', fontsize=8, color=INK)
        axd.text(x_, 11.6, f'n = {n}', ha='center', va='top', fontsize=8, color=INK2)
    axd.text(-0.6, 16.4, 'MAE (mm)', ha='left', va='top', fontsize=8, color=INK2)
    axd.set_xticks(xb)
    axd.set_xticklabels(labs)
    axd.set_xlim(-0.65, len(data) - 0.35)
    axd.set_ylim(min(-10.5, min(d.min() for d in data) - 0.8), 16.4)
    axd.set_yticks([-10, -5, 0, 5, 10])
    axd.set_xlabel('Backfat, ultrasound (mm)')
    axd.set_ylabel('Prediction − ultrasound (mm)')
    axd.grid(axis='y', color=GRID, lw=0.5, zorder=0)
    clean(axd)
    log('   Fig 4d bins: ' + ', '.join(f'{l} n={n} MAE={m:.2f} mean err={d.mean():+.2f}'
                                     for l, n, m, d in zip(labs, ns, maes, data)))

    for ax, s in zip((axa, axb, axc, axd), 'abcd'):
        panel_label(fig, ax, s, dx=0.07)
    save(fig, 'fig4')


# ---------------------------------------------------------------- Fig 1
def box(ax, x, y, w, h, text, fc='#f4f3f0', ec=INK2, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.012', fc=fc, ec=ec,
                                lw=0.7, transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            transform=ax.transAxes, zorder=3, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=7,
                                 lw=0.8, color=INK2, zorder=1, shrinkA=0, shrinkB=0))


def example_raw(uid, frame_idx):
    """示例帧对应的原始彩色帧与 YOLO + FastSAM 掩膜（同 generate_depthmap_dataset 的取帧与分割）。"""
    sys.path.insert(0, SEG_DIR)
    import cv2
    import generate_depthmap_dataset as g
    from ultralytics import YOLO, FastSAM
    idx = {r['uid']: r for r in read_csv(INDEX)}[uid]
    md = idx['master_dir'].replace('G:\\耿来硕数据', 'H:\\耿来硕数据')
    pairs = g.get_frame_pairs(os.path.join(md, 'color'), os.path.join(md, 'depth'))
    cp = pairs[int(frame_idx)][0]
    img = cv2.imread(cp)
    mask = g.get_pig_mask(img, YOLO(os.path.join(SEG_DIR, 'yolo11n.pt')), FastSAM(os.path.join(SEG_DIR, 'FastSAM-s.pt')))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask, cp


def fig1():
    rows, meta = setup_rows()
    att = seed_attention()
    i, uid, fr, h = example_sow(rows, meta, att)
    ho, c0, c1, good = orient(h)
    rgb, mask, cp = example_raw(uid, fr['frame_idx'])
    log(f'   Fig 1 example: {uid} frame {fr["frame_idx"]} ({cp})')

    fig = plt.figure(figsize=(174 * MM, 112 * MM))
    # 上排：数据流
    ax1 = fig.add_axes([0.01, 0.53, 0.27, 0.38])
    ax1.imshow(rgb)
    if mask is not None:
        ax1.contour(mask.astype(float), levels=[0.5], colors=[ORANGE], linewidths=0.9)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.set_title('Top-view colour frame\nand segmentation mask', fontsize=8.5, color=INK, pad=3)
    if mask is not None:   # 该示例帧中猪头朝右（目视确认）；高度图已按 v6f 规则翻转为头在左
        cols, rws = np.where(mask.any(0))[0], np.where(mask.any(1))[0]
        ax1.text(cols[-1], rws[0] - 8, 'Head', ha='right', va='bottom', fontsize=8, color='white')

    ax2 = fig.add_axes([0.32, 0.55, 0.36, 0.34])
    show_hmap(ax2, ho, c0, c1, weights=None)
    ax2.set_title('Height map (7.8 mm per pixel)\nwith 16 transverse profile positions', fontsize=8.5, color=INK,
                  pad=3)
    ax2.text(ax2.get_xlim()[0] + 1, ax2.get_ylim()[1] - 1, 'Head', ha='left', va='top', fontsize=8, color=INK)

    ax3 = fig.add_axes([0.825, 0.58, 0.165, 0.31])
    L = c1 - c0
    for f, lab, ls in ((0.27, 'Shoulder (27%)', (0, (3, 2))), (0.61, 'Last rib (61%)', '-'),
                       (0.89, 'Hip (89%)', (0, (1, 1.2)))):
        x, p = transverse(ho, int(round(c0 + f * L)), good)
        ok = (np.abs(x) <= 375) & ~np.isnan(p)
        ax3.plot(x[ok], p[ok], color=BLUE if f == 0.61 else INK2, lw=1.6 if f == 0.61 else 1, ls=ls, label=lab)
    ax3.legend(loc='lower right', frameon=False, fontsize=8, handlelength=1.8, borderaxespad=0)
    ax3.set_xlim(-280, 280)
    ax3.set_ylim(-400, 20)
    ax3.set_xlabel('Distance from spine (mm)')
    ax3.set_ylabel('Height relative to spine (mm)', fontsize=8.5)
    ax3.set_title('Transverse profiles', fontsize=8.5, color=INK, pad=3)
    ax3.grid(color=GRID, lw=0.5)
    clean(ax3)
    fig.add_artist(FancyArrowPatch((0.282, 0.72), (0.315, 0.72), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=8, lw=0.9, color=INK2))
    fig.add_artist(FancyArrowPatch((0.685, 0.72), (0.73, 0.72), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=8, lw=0.9, color=INK2))

    # 下排：网络结构
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.44])
    ax.set_axis_off()
    box(ax, 0.015, 0.58, 0.17, 0.22, 'Height map\n1 × 128 × 256')
    box(ax, 0.015, 0.14, 0.17, 0.26, '16 transverse profiles\n3 channels × 97 points\n(10%–95% of length)')
    box(ax, 0.22, 0.58, 0.16, 0.22, 'ResNet-18\n(global branch)', fc='#e6effa', ec=BLUE)
    box(ax, 0.22, 0.14, 0.12, 0.26, '1-D CNN\nper profile\n→ 16 × 64', fc='#e6effa', ec=BLUE)
    box(ax, 0.37, 0.14, 0.13, 0.26, '+ position\nembedding\nTransformer × 2', fc='#e6effa', ec=BLUE)
    box(ax, 0.53, 0.14, 0.13, 0.26, 'Position\nattention pooling\n→ 64', fc='#fdeee6', ec=ORANGE)
    box(ax, 0.53, 0.58, 0.13, 0.22, '512 features')
    box(ax, 0.70, 0.36, 0.09, 0.26, 'Concatenate\n576 → linear')
    box(ax, 0.82, 0.36, 0.16, 0.26, 'Frame backfat\n→ mean over frames\n→ sow prediction\n+ 90% interval')
    arrow(ax, 0.185, 0.69, 0.22, 0.69)
    arrow(ax, 0.185, 0.27, 0.22, 0.27)
    arrow(ax, 0.38, 0.69, 0.53, 0.69)
    arrow(ax, 0.34, 0.27, 0.37, 0.27)
    arrow(ax, 0.50, 0.27, 0.53, 0.27)
    arrow(ax, 0.66, 0.69, 0.745, 0.62)
    arrow(ax, 0.66, 0.27, 0.745, 0.36)
    arrow(ax, 0.79, 0.49, 0.82, 0.49)
    ax.text(0.595, 0.06, 'Attention weights show which part of the body is used (Fig. 2)', ha='center',
            va='center', fontsize=8, color=INK2, transform=ax.transAxes, style='italic')
    ax.text(0.02, 0.92, 'Anatomy-guided height-map network', ha='left', va='center', fontsize=9,
            color=INK, transform=ax.transAxes)

    fig.text(0.005, 0.95, 'a', fontsize=10, color=INK)
    fig.text(0.005, 0.425, 'b', fontsize=10, color=INK)
    save(fig, 'fig1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', nargs='+', type=int, default=[2, 3, 4])
    a = ap.parse_args()
    for n in a.fig:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4}[n]()


if __name__ == '__main__':
    main()
