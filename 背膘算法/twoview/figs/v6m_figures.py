# -*- coding: utf-8 -*-
"""
v6m_figures.py — gComm 稿件图 1–4 统一重画（2026-10-19；数字与结论不变，只改呈现）

  Fig 1 (a) 彩色帧 + 掩膜 → 高度图（16 条横截面位置，栏杆遮挡列淡化标注）→ 3 条横截面；(b) 网络结构示意图（重画）
  Fig 2 (a) 横截面位置注意力（5 种子均值 + 种子间范围）；(b) 示例猪高度图叠加注意力；
        (c) 改为 ΔR² 森林图（每个比较相对其参照；全体 / 丹系头胎 / 新美系经产并排；group-bootstrap 95% CI）
  Fig 3 (a)(b) 按卡尺读数分组的均值 ± 95% CI + 淡色散点 + 品系内回归线；(c) ΔR² 森林图（统一风格，数值列对齐）
  Fig 4 (a) 信度与 R² 拆成上下两个子轴；(b) 保形覆盖率；(c) 复测分流；(d) 按 B 超背膘分段的误差

来源 / 背景
  替代 figs/v6i_figures.py 的呈现（v6i 与旧图 figures/ 保留不动）。数据读取与示例猪 / 示例帧的选取规则直接 import v6i，
  保证同一示例。图 2c 的 ΔR² 全部用正则从 runs/* 日志读取（不重算），日志行号写入 runs/v6m_figures/log.txt。
期刊要求（gComm 模板）：Times New Roman 非粗体 8–10 pt（面板标签 9 pt 粗体，图注说明），线宽 ≥ 0.5 pt，RGB，600 dpi。
配色（dataviz 参考调色板前三槽，validate_palette.js --pairs all 浅色全部 PASS；aqua 对比度 < 3:1 → 只作线 / 填充，配直接标注）：
  丹系 / 丹系头胎 = 蓝 #2a78d6；新美系 / 新美系经产 = 橙 #eb6834；全体 = 深墨灰；
  主模型（解剖引导网络）及其横截面 / 注意力 = aqua #1baf7a；边际校准 = 浅灰空心，按品系 × 胎次校准 = 墨色实心；
  参照线 = 中灰；文字一律墨色。
输出：F:/Postgraduate/Lab/十月论文/figures_v2/fig{1,2,3,4}.{tif,pdf,svg,jpg,png}
用法：D:\\Anaconda\\envs\\pytorch\\python.exe figs/v6m_figures.py --fig 1 2 3 4
      （Fig 1 首次运行需要 ultralytics + 原始录像盘 H:，分割结果缓存到 runs/v6m_figures/fig1_example_seg.npz）
"""
import argparse
import io
import os
import re
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v6i_figures as v6i  # noqa: E402  数据读取、示例猪选取、头尾判定、横截面提取沿用 v6i

RUNS = v6i.RUNS
OUT = r'F:/Postgraduate/Lab/十月论文/figures_v2'
LOG = os.path.join(RUNS, 'v6m_figures', 'log.txt')
SEG_CACHE = os.path.join(RUNS, 'v6m_figures', 'fig1_example_seg.npz')
F_POS, CELL = v6i.F_POS, v6i.CELL
MM = 1 / 25.4
W_FIG = 174.0
DPI = 600

INK, AXC, REF, LIGHT, GRID = '#0b0b0b', '#2b2a28', '#8a8882', '#c9c7c0', '#ecebe7'
C_DAN, C_NAM, C_ALL, C_MAIN = '#2a78d6', '#eb6834', '#3a3936', '#1baf7a'
C_MARG, C_GRP = '#a3a19a', INK
MAIN_FILL, MAIN_FILL2, NEUTRAL_FILL, INPUT_FILL = '#e4f5ee', '#b9e6d1', '#eef0f2', '#f4f3ef'
LINE_COL = {'DAN': C_DAN, 'NAM': C_NAM}
LINE_NAME = {'DAN': 'Danish line', 'NAM': 'American line'}
SUBS = [('all', 'All sows', C_ALL, 'o'),
        ('DAN_parity1', 'Danish line, parity 1', C_DAN, 's'),
        ('NAM_multiparous', 'American line, parity ≥ 2', C_NAM, '^')]


def sfmt(v):
    """带符号三位小数，负号用 U+2212。"""
    return f'{v:+.3f}'.replace('-', '−')

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'text.color': INK,
    'axes.edgecolor': AXC, 'axes.labelcolor': INK, 'xtick.color': AXC, 'ytick.color': AXC,
    'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2,
    'ytick.major.pad': 2, 'axes.labelpad': 3, 'lines.linewidth': 1.2, 'legend.frameon': False,
    'legend.handletextpad': 0.5, 'legend.borderaxespad': 0.2, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'mathtext.fontset': 'stix', 'axes.unicode_minus': True,
})


# ---------------------------------------------------------------- 通用
def log(s):
    """打印并追加到 runs/v6m_figures/log.txt（图中每个数字的来源）。"""
    print(s, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f'[{time.strftime("%F %T")}] {s.rstrip()}' + chr(10))


def new_fig(h_mm):
    fig = plt.figure(figsize=(W_FIG * MM, h_mm * MM))
    fig._h_mm = h_mm
    return fig


def axmm(fig, x, y, w, h, **kw):
    """以 mm 为单位、原点在左上角放置坐标轴。"""
    return fig.add_axes([x / W_FIG, 1 - (y + h) / fig._h_mm, w / W_FIG, h / fig._h_mm], **kw)


def plabel(fig, x, y, s):
    """面板标签：9 pt 粗体小写，放在面板单元格左上角（mm）。"""
    fig.text(x / W_FIG, 1 - y / fig._h_mm, s, fontsize=9, fontweight='bold', ha='left', va='top', color=INK)


def ftext(fig, x, y, s, **kw):
    kw.setdefault('fontsize', 8)
    kw.setdefault('color', INK)
    fig.text(x / W_FIG, 1 - y / fig._h_mm, s, **kw)


def clean(ax, grid='y'):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, name)
    fig.savefig(base + '.pdf', facecolor='white')
    fig.savefig(base + '.svg', facecolor='white')
    fig.savefig(base + '.png', dpi=300, facecolor='white')           # 预览
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor='white')
    im = Image.open(buf).convert('RGB')                                # 期刊要求 RGB（去 alpha）
    im.save(base + '.tif', compression='tiff_lzw', dpi=(DPI, DPI))
    im.save(base + '.jpg', quality=95, dpi=(DPI, DPI))
    plt.close(fig)
    log(f'saved -> {base}.{{tif,pdf,svg,jpg,png}}  ({im.size[0]} x {im.size[1]} px, {DPI} dpi RGB)')


# ---------------------------------------------------------------- 日志读取（ΔR²）
DR_RE = re.compile(r'^\s*(?P<name>.+?)\s*\[(?P<sub>[A-Za-z0-9_]+)\s*\]\s*ΔR²\s*(?P<d>[+-]\d+\.\d+)\s*'
                   r'\((?P<lo>[+-]\d+\.\d+),\s*(?P<hi>[+-]\d+\.\d+)\)')


def read_dr(rel, name, start_line=1, start_marker=None):
    """从日志读取某比较在三个子集上的 ΔR² (lo, d, hi, 行号)；同名重复出现则报错。"""
    path = os.path.join(RUNS, rel)
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    if start_marker is not None:
        start_line = [i for i, ln in enumerate(lines, 1) if start_marker in ln][0]
    out = {}
    for i, ln in enumerate(lines[start_line - 1:], start_line):
        m = DR_RE.match(ln)
        if not m or ' '.join(m['name'].split()) != name:
            continue
        sub = m['sub']
        if sub not in [s[0] for s in SUBS]:
            continue                              # no_prev_season / has_prev_season 不进本文
        assert sub not in out, f'{rel}: {name} [{sub}] 重复出现（行 {out.get(sub)}、{i}）'
        out[sub] = (float(m['lo']), float(m['d']), float(m['hi']), i)
    assert len(out) == 3, f'{rel}: {name} 只找到 {list(out)}'
    return out


# ---------------------------------------------------------------- 高度图
def occluded_runs(c0, c1, good, merge=4):
    """体长范围内被栏杆遮挡的列段（v6i.orient 的 ~good，去掉贴着头尾两端的窄端段，相距 ≤ merge 列的段合并）。"""
    bad = [c for c in range(c0, c1 + 1) if not good[c]]
    runs = []
    for c in bad:
        if runs and c - runs[-1][1] <= merge:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return [(a, b) for a, b in runs if a > c0 + 1 and b < c1 - 1]


def body_rows(h, c0, c1):
    """体长范围内至少 3 个有效像素的行（排除掩膜外零星像素，用于画位置线的上下端与裁剪）。"""
    return np.where((h[:, c0:c1 + 1] > 0).sum(1) >= 3)[0]


def show_hmap(ax, h, c0, c1, good, weights=None, highlight=None, scalebar=True):
    """高度图（灰度，越深越高）+ 16 条横截面位置线；weights 给出时线宽 / 不透明度 ∝ 注意力。
    遮挡列用半透明白色覆盖淡化并标注 'Occluded'（不改动、不插补数据）。"""
    hm = np.ma.masked_where(h <= 0, h)
    rows_on = body_rows(h, c0, c1)
    r0, r1 = max(rows_on[0] - 3, 0), min(rows_on[-1] + 3, h.shape[0] - 1)
    im = ax.imshow(hm, cmap='Greys', vmin=300, vmax=1100, interpolation='nearest', aspect='equal', zorder=1)
    ax.set_ylim(r1 + 6, r0 - 7)                            # 上方留 'Occluded' 标注、下方留比例尺
    ax.set_xlim(c0 - 4.5, c1 + 4.5)
    for a, b in occluded_runs(c0, c1, good):
        ax.add_patch(Rectangle((a - 0.5, rows_on[0] - 1), b - a + 1, rows_on[-1] - rows_on[0] + 2, fc='white',
                               ec='none', alpha=0.72, zorder=2))
        ax.text((a + b) / 2, rows_on[0] - 2, 'Occluded', ha='center', va='bottom', fontsize=7, color=INK, zorder=5)
    L = c1 - c0
    for k, f in enumerate(F_POS):
        c = c0 + f * L
        if weights is not None:
            w = weights[k] / np.nanmax(weights)
            ax.plot([c, c], [rows_on[0] - 1, rows_on[-1] + 1], color=C_MAIN, lw=0.5 + 2.5 * w, alpha=0.35 + 0.65 * w,
                    solid_capstyle='butt', zorder=3)
        elif highlight is not None and k in highlight:
            ax.plot([c, c], [rows_on[0] - 1, rows_on[-1] + 1], color=INK, lw=1.0, ls=highlight[k],
                    dash_capstyle='butt', zorder=4)
        else:
            ax.plot([c, c], [rows_on[0] - 1, rows_on[-1] + 1], color=C_MAIN, lw=0.6, alpha=0.75,
                    solid_capstyle='butt', zorder=3)
    if scalebar:
        xb = c1 + 4 - 200 / CELL
        yb = r1 + 0.5
        ax.plot([xb, xb + 200 / CELL], [yb, yb], color=INK, lw=1.2, solid_capstyle='butt', zorder=5)
        ax.text(xb + 100 / CELL, yb + 1.2, '200 mm', ha='center', va='top', fontsize=7, color=INK, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im, rows_on


def example():
    rows, meta = v6i.setup_rows()
    att = v6i.seed_attention()
    i, uid, fr, h = v6i.example_sow(rows, meta, att)
    ho, c0, c1, good = v6i.orient(h)
    return rows, att, i, uid, fr, ho, c0, c1, good


# ---------------------------------------------------------------- Fig 1
def rbox(ax, x, y, w, h, text='', fc=INPUT_FILL, ec=AXC, lw=0.6, fs=8, ty=None, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.3', fc=fc, ec=ec, lw=lw,
                                zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2 if ty is None else ty, text, ha='center', va='center', fontsize=fs, color=INK,
                zorder=4, linespacing=1.2, **kw)


def harrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=6, lw=0.7, color=AXC,
                                 shrinkA=0, shrinkB=0, zorder=1))


def seg_example(uid, frame_idx):
    if os.path.exists(SEG_CACHE):
        z = np.load(SEG_CACHE, allow_pickle=True)
        if str(z['uid']) == uid and int(z['frame_idx']) == int(frame_idx):
            return z['rgb'], z['mask'], str(z['path'])
    rgb, mask, cp = v6i.example_raw(uid, frame_idx)
    os.makedirs(os.path.dirname(SEG_CACHE), exist_ok=True)
    np.savez_compressed(SEG_CACHE, rgb=rgb, mask=mask, path=cp, uid=uid, frame_idx=int(frame_idx))
    return rgb, mask, cp


def fig1():
    rows, att, i, uid, fr, ho, c0, c1, good = example()
    rgb, mask, cp = seg_example(uid, fr['frame_idx'])
    log(f'Fig 1 example: {uid} frame {fr["frame_idx"]} backfat {rows[i]["y"]} mm ({cp}); '
        f'occluded column runs {occluded_runs(c0, c1, good)}')

    # ---- (a) 数据流：左列彩色帧（上）→ 高度图（下），同宽；右侧横截面
    rr, cc = np.where(mask)
    pr, pc = 25, 25
    rlo, rhi = max(rr.min() - pr, 0), min(rr.max() + pr, rgb.shape[0])
    clo, chi = max(cc.min() - pc, 0), min(cc.max() + pc, rgb.shape[1])
    crop, mcrop = rgb[rlo:rhi, clo:chi], mask[rlo:rhi, clo:chi]
    xL, wL = 4.0, 98.0
    y1 = 7.0
    h1 = wL * crop.shape[0] / crop.shape[1]
    rows_on = body_rows(ho, c0, c1)
    nr = rows_on[-1] - rows_on[0] + 19          # 与 show_hmap 的 ylim 一致
    nc = (c1 - c0) + 9
    y2 = y1 + h1 + 7.5
    h2 = wL * nr / nc
    x3, w3 = 128.0, 43.0
    h3 = 50.0
    y3 = y2 + h2 - h3 - 10.0
    yb0, hb = max(y2 + h2, y3 + h3 + 9.0) + 5.0, 64.0       # (b) 的起点与高度（mm）
    fig = new_fig(yb0 + hb)
    ax1 = axmm(fig, xL, y1, wL, h1)
    ax1.imshow(crop, interpolation='lanczos')
    ax1.contour(mcrop.astype(float), levels=[0.5], colors=[C_MAIN], linewidths=0.9)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for sp in ax1.spines.values():
        sp.set_visible(False)
    hcols = np.where(mcrop.any(0))[0]
    hrows = np.where(mcrop[:, hcols[-1] - 10:hcols[-1] + 1].any(1))[0]
    ax1.text(crop.shape[1] + 12, hrows.mean(), 'Head', ha='left', va='center', fontsize=7, color=INK, clip_on=False)

    ax2 = axmm(fig, xL, y2, wL, h2)
    hl = {3: (0, (3, 1.6)), 9: '-', 14: (0, (1, 1.2))}          # F_POS[3]=0.27, [9]=0.61, [14]=0.89
    show_hmap(ax2, ho, c0, c1, good, highlight=hl)
    ax2.text(c0 - 3, rows_on[0] - 2, 'Head', ha='left', va='bottom', fontsize=7, color=INK, zorder=6)

    ax3 = axmm(fig, x3, y3, w3, h3)
    L = c1 - c0
    for k, (f, lab) in enumerate(((0.27, 'Shoulder, 27%'), (0.61, 'Last rib, 61%'), (0.89, 'Hip, 89%'))):
        assert abs(F_POS[[3, 9, 14][k]] - f) < 0.006
        x, p = v6i.transverse(ho, int(round(c0 + f * L)), good)
        ok = (np.abs(x) <= 375) & ~np.isnan(p)
        ax3.plot(x[ok], p[ok], color=INK, lw=1.0, ls=hl[[3, 9, 14][k]], dash_capstyle='butt', label=lab)
    ax3.legend(loc='lower center', handlelength=2.2, fontsize=7, borderaxespad=0, labelspacing=0.25)
    ax3.set_xlim(-280, 280)
    ax3.set_ylim(-340, 20)
    ax3.set_xticks([-200, 0, 200])
    ax3.set_yticks([-300, -200, -100, 0])
    ax3.set_xlabel('Distance from spine (mm)')
    ax3.set_ylabel('Height relative to spine (mm)')
    clean(ax3, grid='y')

    ftext(fig, xL, y1 - 1.2, 'Color frame and segmentation mask', ha='left', va='bottom')
    ftext(fig, xL, y2 - 1.2, 'Height map with 16 profile positions (7.8 mm per pixel)', ha='left', va='bottom')
    ftext(fig, x3 + w3 / 2, y3 - 1.5, 'Transverse profiles', ha='center', va='bottom')

    def farrow(p0, p1):
        fig.add_artist(FancyArrowPatch((p0[0] / W_FIG, 1 - p0[1] / fig._h_mm), (p1[0] / W_FIG, 1 - p1[1] / fig._h_mm),
                                       transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=0.8,
                                       color=AXC))
    farrow((xL + wL * 0.75, y1 + h1 + 1.0), (xL + wL * 0.75, y2 - 0.5))
    farrow((xL + wL + 2.0, y2 + h2 / 2), (x3 - 13.0, y2 + h2 / 2))

    # ---- (b) 网络结构示意图（axes 坐标即 mm）
    ax = axmm(fig, 0, yb0, W_FIG, hb)
    ax.set_xlim(0, W_FIG)
    ax.set_ylim(0, hb)
    ax.set_axis_off()
    BH = 20.0
    yt, yb = 36.0, 6.0                       # 上下两行盒子的底边
    ct, cb = yt + BH / 2, yb + BH / 2
    X = {'in': (3, 27), 'enc': (36, 20), 'tr': (62, 22), 'pool': (90, 21), 'cat': (117, 10), 'lin': (132, 15),
         'agg': (152, 20)}
    # 输入：高度图缩略图 / 横截面示意
    x, w = X['in']
    rbox(ax, x, yt, w, BH, 'Height map\n1 × 128 × 256', ty=yt + 5.0)
    th = ax.inset_axes([x + 2.0, yt + 9.5, w - 4.0, 9.0], transform=ax.transData)
    th.imshow(np.ma.masked_where(ho <= 0, ho)[rows_on[0]:rows_on[-1] + 1, c0:c1 + 1], cmap='Greys', vmin=300,
              vmax=1100, aspect='auto', interpolation='nearest')
    th.set_axis_off()
    rbox(ax, x, yb, w, BH, '16 transverse profiles\n3 × 97 points each', ty=yb + 5.0)
    pr_ax = ax.inset_axes([x + 3.0, yb + 10.0, w - 6.0, 8.5], transform=ax.transData)
    for k, f in enumerate(F_POS[::3]):
        xx, p = v6i.transverse(ho, int(round(c0 + f * L)), good)
        ok = (np.abs(xx) <= 300) & ~np.isnan(p)
        pr_ax.plot(xx[ok] + k * 140, p[ok], color=C_MAIN, lw=0.7)
    pr_ax.set_axis_off()
    # 全局分支
    x, w = X['enc']
    rbox(ax, x, yt, X['tr'][0] + X['tr'][1] - x, BH, 'ResNet-18', fc=NEUTRAL_FILL)
    x, w = X['pool']
    rbox(ax, x, yt, w, BH, 'Global feature\n512', fc=NEUTRAL_FILL)
    # 横截面分支（解剖引导）
    rbox(ax, *X['enc'][:1], yb, X['enc'][1], BH, '1-D CNN\nper profile\n16 × 64', fc=MAIN_FILL, ec=C_MAIN)
    rbox(ax, X['tr'][0], yb, X['tr'][1], BH, 'Position\nembedding +\nTransformer × 2', fc=MAIN_FILL, ec=C_MAIN)
    rbox(ax, X['pool'][0], yb, X['pool'][1], BH, 'Attention\npooling over\npositions, 64', fc=MAIN_FILL2, ec=C_MAIN)
    # 汇合与输出
    x, w = X['cat']
    rbox(ax, x, yb, w, yt + BH - yb, 'Concatenate, 576', rotation=90)
    rbox(ax, X['lin'][0], (ct + cb) / 2 - BH / 2, X['lin'][1], BH, 'Linear\n→ frame\nbackfat')
    rbox(ax, X['agg'][0], (ct + cb) / 2 - BH / 2, X['agg'][1], BH, 'Mean over\nframes → sow\nbackfat and\n90% interval')
    gap = 0.0
    for cy, seq in ((ct, ('in', 'enc')), (ct, ('tr', 'pool')), (cb, ('in', 'enc')), (cb, ('enc', 'tr')),
                    (cb, ('tr', 'pool')), (ct, ('pool', 'cat')), (cb, ('pool', 'cat'))):
        a, b = seq
        harrow(ax, X[a][0] + X[a][1] + gap, X[b][0] - gap, cy)
    cm = (ct + cb) / 2
    harrow(ax, X['cat'][0] + X['cat'][1], X['lin'][0], cm)
    harrow(ax, X['lin'][0] + X['lin'][1], X['agg'][0], cm)
    ax.text(X['enc'][0], yt + BH + 1.6, 'Global branch', ha='left', va='bottom', fontsize=8, color=INK)
    ax.text(X['enc'][0], yb - 1.6, 'Profile branch', ha='left', va='top', fontsize=8, color=INK)

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 0.5, yb0 + 1.0, 'b')
    save(fig, 'fig1')


# ---------------------------------------------------------------- Fig 2
FIG2C = [
    ('Image representation (top view)', None),
    ('Height map vs cropped depth map', ('v4_hmap/compare/log.txt', 'hmap', 1, None)),
    ('Anatomy-guided vs ResNet-18 (five seeds)',
     ('v6_anatomy/compare_log.txt', 'prof vs base', 1, '[2026-10-15 12:19:18]')),
    ('Camera views, cropped depth maps', None),
    ('Rear-oblique vs top view', ('v3_views/compare/log.txt', 'sub', 1, None)),
    ('Late fusion vs top view', ('v3_views/compare/log.txt', 'late', 1, None)),
    ('Dual-view vs top view', ('v3_views/compare/log.txt', 'dual', 1, None)),
    ('Camera views, top-view height map', None),
    ('Rear-oblique vs top height map', ('v6_dualview_hmap/log.txt', 'rear-oblique cropped (v3b sub)', 277, None)),
    ('Late fusion vs top height map', ('v6_dualview_hmap/log.txt', 'late fusion (mean of the two)', 277, None)),
    ('Dual-view vs top height map', ('v6_dualview_hmap/log.txt', 'dual-branch (this script)', 277, None)),
]


def forest_rows(ax, items, xr, label_fn=None):
    """通用森林图行布局：组标题（斜体）+ 比较行；返回 (y 位置列表, 标签列表)。"""
    y, yt, yl = 0.0, [], []
    for lab, d in items:
        if d is None:
            ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8,
                    style='italic', color=INK)
        else:
            yt.append(y)
            yl.append(lab)
            if label_fn:
                label_fn(y, lab, d)
        y -= 1
    ax.axvline(0, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_xlim(*xr)
    ax.set_ylim(y + 0.45, 0.55)
    ax.spines['left'].set_visible(False)
    clean(ax, grid='x')
    return yt


def fig2():
    rows, att, i, uid, fr, ho, c0, c1, good = example()
    seed_curves = np.nanmean(att, 1)
    mean_curve = np.nanmean(att, (0, 1))
    res = v6i.load('v6_anatomy/prof/result.json')
    assert np.allclose(mean_curve, res['position_attention_mean'], atol=1e-6), '与 result.json 不一致'

    fig = new_fig(141)
    # (a)
    axa = axmm(fig, 13, 5, 68, 44)
    x = F_POS * 100
    axa.fill_between(x, seed_curves.min(0), seed_curves.max(0), color=C_MAIN, alpha=0.22, lw=0, zorder=2)
    axa.plot(x, mean_curve, color=C_MAIN, lw=1.5, zorder=3)
    axa.plot(x, mean_curve, 'o', ms=3, mfc=C_MAIN, mec='white', mew=0.6, zorder=4)
    axa.axhline(1 / 16, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axa.text(98, 1 / 16 + 0.004, '1/16', ha='right', va='bottom', fontsize=7, color=INK)
    axa.axvline(60, color=REF, lw=0.7, ls=(0, (1, 1.3)), zorder=1)
    axa.text(58.5, 0.245, 'Virtual\ncaliper', ha='right', va='top', fontsize=7, color=INK, linespacing=1.1)
    k = int(np.argmax(mean_curve))
    axa.text(x[k] + 2.5, mean_curve[k] + 0.003, f'{mean_curve[k]:.2f}', ha='left', va='bottom', fontsize=7,
             color=INK)
    axa.set_xlim(5, 100)
    axa.set_ylim(0, 0.25)
    axa.set_xticks([10, 30, 50, 70, 90])
    axa.set_xlabel('Position along the body (% of length from head)')
    axa.set_ylabel('Attention weight')
    clean(axa)
    log(f'Fig 2a: mean attention peak {mean_curve[k]:.4f} at {x[k]:.1f}% (runs/v6_anatomy/prof/result.json '
        f'position_attention_mean; seed range from _seed_ckpt/seed42–46.npz)')

    # (b)
    w = np.nanmean(att[:, i], 0)
    rows_on = body_rows(ho, c0, c1)
    nr = rows_on[-1] - rows_on[0] + 19          # 与 show_hmap 的 ylim 一致
    nc = (c1 - c0) + 9
    wb = 82.0
    hb = wb * nr / nc
    axb = axmm(fig, 90, 5, wb, hb)
    im, _ = show_hmap(axb, ho, c0, c1, good, weights=w)
    axb.text(c0 - 3, rows_on[0] - 2, 'Head', ha='left', va='bottom', fontsize=7, color=INK, zorder=6)
    cax = axmm(fig, 90 + wb * 0.25, 5 + hb + 3.0, wb * 0.5, 1.8)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[400, 600, 800, 1000])
    cb.set_label('Height above floor (mm)', fontsize=8, labelpad=1.5)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.5)
    log(f'Fig 2b: example {uid} frame {fr["frame_idx"]}; attention peak at {F_POS[int(np.argmax(w))]:.2f}')

    # (c) ΔR² 森林图
    data = []
    for lab, src in FIG2C:
        if src is None:
            data.append((lab, None))
            continue
        rel, name, start, marker = src
        d = read_dr(rel, name, start, marker)
        data.append((lab, d))
        for sub, _, _, _ in SUBS:
            lo, mid, hi, ln = d[sub]
            log(f'Fig 2c: {lab:<42s} [{sub:<15s}] ΔR² {mid:+.3f} ({lo:+.3f}, {hi:+.3f})  <- runs/{rel}:{ln}')
    # 交叉核对：解剖引导 vs 基线 与 v6h result.json 一致
    v6h = v6i.load('v6_main_vs_caliper/result.json')['comparisons']
    for sub, _, _, _ in SUBS:
        lo, mid, hi = v6h[f'main vs ResNet-18 baseline (5 seeds) [{sub}]']['dR2']
        lo2, mid2, hi2, _ = data[2][1][sub]
        assert abs(lo - lo2) < 6e-4 and abs(mid - mid2) < 6e-4 and abs(hi - hi2) < 6e-4, (sub, lo, mid, hi)
    log('Fig 2c: anatomy-guided vs baseline cross-checked with runs/v6_main_vs_caliper/result.json (|diff| < 0.0006)')

    axc = axmm(fig, 64, 72, 106, 59)
    off = {'all': -0.24, 'DAN_parity1': 0.0, 'NAM_multiparous': 0.24}

    def draw(y, lab, d):
        for sub, _, col, mk in SUBS:
            lo, mid, hi, _ = d[sub]
            yy = y + off[sub] * -1
            axc.plot([lo, hi], [yy, yy], color=col, lw=1.0, solid_capstyle='butt', zorder=2)
            axc.plot(mid, yy, mk, ms=3.6 if mk != '^' else 4.0, mfc=col, mec='white', mew=0.5, zorder=3)

    forest_rows(axc, data, (-0.21, 0.18), draw)
    axc.set_xticks([-0.2, -0.1, 0, 0.1])
    axc.set_xlabel('Difference in R² (95% CI)')
    handles = [Line2D([], [], color=col, marker=mk, ms=3.6, mec='white', mew=0.5, lw=1.0, label=name)
               for _, name, col, mk in SUBS]
    axc.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.4,
               handlelength=1.6, borderaxespad=0.3)

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 86.5, 1.0, 'b')
    plabel(fig, 0.5, 63.0, 'c')
    save(fig, 'fig2')


# ---------------------------------------------------------------- Fig 3
def tcrit(n):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1))
    except ImportError:
        return 1.96


def by_level(ax, xs, ys, lines, rng, tag, min_n=5):
    """按卡尺读数（0.2 一档）分组：淡色散点 + 每档均值 ± 95% CI（n ≥ min_n）+ 品系内最小二乘线。"""
    out = []
    dodge = {'DAN': -0.035, 'NAM': 0.035}
    for ln in ('DAN', 'NAM'):
        m = (lines == ln) & ~np.isnan(xs) & ~np.isnan(ys)
        col = LINE_COL[ln]
        xj = xs[m] + dodge[ln] + rng.uniform(-0.022, 0.022, m.sum())
        ax.scatter(xj, ys[m], s=3, color=col, alpha=0.22, linewidths=0, zorder=2, rasterized=True)
        shown = []
        for lv in np.unique(np.round(xs[m], 1)):
            yv = ys[m][np.isclose(xs[m], lv)]
            if len(yv) < min_n:
                continue
            mu, se = yv.mean(), yv.std(ddof=1) / np.sqrt(len(yv))
            ci = tcrit(len(yv)) * se
            ax.plot([lv + dodge[ln]] * 2, [mu - ci, mu + ci], color=col, lw=1.0, solid_capstyle='butt', zorder=4)
            ax.plot(lv + dodge[ln], mu, 's' if ln == 'DAN' else '^', ms=3.4 if ln == 'DAN' else 3.8, mfc=col,
                    mec='white', mew=0.5, zorder=5)
            shown.append(f'{lv:.1f}(n={len(yv)})')
        b1, b0 = np.polyfit(xs[m], ys[m], 1)
        xl = np.array([np.percentile(xs[m], 2), np.percentile(xs[m], 98)])
        ax.plot(xl, b0 + b1 * xl, color=col, lw=1.3, zorder=3, solid_capstyle='round')
        r = np.corrcoef(xs[m], ys[m])[0, 1]
        out.append((ln, r, int(m.sum())))
        log(f'Fig 3{tag}: {ln} n={m.sum()} r={r:.4f}; levels with mean ± CI: {", ".join(shown)}')
    return out


def line_legend(ax, stats, extra=None):
    hs = [Line2D([], [], color=LINE_COL[ln], marker='s' if ln == 'DAN' else '^', ms=3.6, mec='white', mew=0.5,
                 lw=1.3, label=f'{LINE_NAME[ln]} (n = {n}), r = {r:.2f}') for ln, r, n in stats]
    if extra:
        hs.append(Line2D([], [], color='none', label=extra))
    ax.legend(handles=hs, loc='upper left', handlelength=1.8, borderaxespad=0.1, labelspacing=0.3)


def fig3():
    fig = new_fig(140)
    # (a) 虚拟 vs 人工卡尺（663 头分析集）
    axa = axmm(fig, 13, 5, 70, 50)
    keep = {r['uid'] for r in v6i.read_csv('v1_tabular/preds.csv')}
    rows = [r for r in v6i.read_csv('v5_caliper/features.csv') if r['farm'] == 'FMH' and r['uid'] in keep]
    cal = np.array([v6i.fnum(r['caliper']) for r in rows])
    ang = np.array([v6i.fnum(r['angle_f0.60_D150']) for r in rows])
    line = np.array([r['line'] for r in rows])
    miss = [(r['uid'], r['line'], r['caliper'], r['angle_f0.60_D150']) for r in rows
            if np.isnan(v6i.fnum(r['caliper'])) or np.isnan(v6i.fnum(r['angle_f0.60_D150']))]
    log(f'Fig 3a: {len(rows)} records (runs/v5_caliper/features.csv, farm FMH ∩ v1_tabular/preds.csv); '
        f'excluded for missing values: {miss}')
    log('Fig 3a n vs Table 1: Danish 446 here vs 447 caliper scores in Table 1 (386 + 61) because the virtual '
        'caliper angle is missing for fmh_8.27_740A311; American 215 = 216 − 1 implausible caliper entry '
        '(fmh_8.24_559LY0289), same as Table 1 (27 + 188).')
    st = by_level(axa, cal, ang, line, np.random.default_rng(1), 'a')
    ok = ~np.isnan(cal) & ~np.isnan(ang)
    r_all = np.corrcoef(cal[ok], ang[ok])[0, 1]
    log(f'Fig 3a: all sows n={ok.sum()} r={r_all:.4f}')
    line_legend(axa, st, f'All sows (n = {ok.sum()}), r = {r_all:.2f}')
    axa.set_xlabel('Manual caliper score')
    axa.set_ylabel('Virtual caliper angle (°)')
    axa.set_xticks(np.arange(2.2, 5.01, 0.4))
    axa.set_xlim(2.05, 5.15)
    axa.set_ylim(98, 152)
    clean(axa)

    # (b) 卡尺读数–背膘
    axb = axmm(fig, 100, 5, 70, 50)
    rows = v6i.read_csv('v1b_caliper/preds.csv')
    cal = np.array([v6i.fnum(r['caliper']) if r['caliper_flag'] != 'implausible' else np.nan for r in rows])
    y = np.array([float(r['y']) for r in rows])
    line = np.array([r['line'] for r in rows])
    log(f'Fig 3b: {len(rows)} records (runs/v1b_caliper/preds.csv); implausible caliper excluded: '
        f'{[r["uid"] for r in rows if r["caliper_flag"] == "implausible"]}')
    st = by_level(axb, cal, y, line, np.random.default_rng(0), 'b')
    line_legend(axb, st)
    axb.set_xlabel('Manual caliper score')
    axb.set_ylabel('Backfat, ultrasound (mm)')
    axb.set_xticks(np.arange(2.2, 5.01, 0.4))
    axb.set_xlim(2.05, 5.15)
    axb.set_ylim(7, 35)
    clean(axb)

    # (c) ΔR² 森林图（全体；v6h 五种子 + v6a 嵌套图像分数）
    v6a = v6i.load('v6_camera_caliper_anatomy/result.json')['comparisons']
    v6h = v6i.load('v6_main_vs_caliper/result.json')['comparisons']
    items = [
        ('Camera versus other single sources', None),
        ('Main network vs caliper score', ('v6h', 'main vs caliper (C) [all]')),
        ('Image score vs caliper score', ('v6a', 'camera vs manual caliper (stacked image score)')),
        ('Image score vs production records', ('v6a', 'camera vs production records (reference)')),
        ('Image geometry vs caliper score', ('v6a', 'image geometry vs manual caliper')),
        ('Increment from adding a source', None),
        ('Image score added to caliper score', ('v6a', 'camera over manual caliper')),
        ('Caliper score added to image score', ('v6a', 'manual caliper over camera')),
        ('Image geometry added to image score', ('v6a', 'image geometry over CNN score')),
        ('Body dimensions added to image score', ('v6a', 'whole-body dims over CNN score')),
        ('Image geometry added to caliper + image score', ('v6a', 'image geometry over caliper+camera')),
    ]
    data = []
    for lab, src in items:
        if src is None:
            data.append((lab, None))
            continue
        d = v6h[src[1]]['dR2'] if src[0] == 'v6h' else v6a[src[1]]['all']['dR2']
        data.append((lab, (d, src[0] == 'v6h')))
        js = ('runs/v6_main_vs_caliper/result.json' if src[0] == 'v6h'
              else 'runs/v6_camera_caliper_anatomy/result.json [all]')
        log(f'Fig 3c: {lab:<46s} ΔR² {d[1]:+.3f} ({d[0]:+.3f}, {d[2]:+.3f})  <- {js} "{src[1]}"')

    axc = axmm(fig, 64, 74, 72, 58)

    def draw(y, lab, dd):
        (lo, mid, hi), main = dd
        col = C_MAIN if main else C_ALL
        axc.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle='butt', zorder=2)
        axc.plot(mid, y, 'o', ms=4, mfc=col, mec='white', mew=0.6, zorder=3)
        axc.text(1.02, y, sfmt(mid), transform=axc.get_yaxis_transform(), ha='left', va='center', fontsize=8)
        axc.text(1.155, y, f'({sfmt(lo)}, {sfmt(hi)})', transform=axc.get_yaxis_transform(), ha='left', va='center',
                 fontsize=8)

    forest_rows(axc, data, (-0.05, 0.30), draw)
    axc.text(1.02, 0.55 - 0.0, 'ΔR²', transform=axc.get_yaxis_transform(), ha='left', va='bottom', fontsize=8)
    axc.text(1.155, 0.55, '95% CI', transform=axc.get_yaxis_transform(), ha='left', va='bottom', fontsize=8)
    axc.set_xticks([0, 0.1, 0.2, 0.3])
    axc.set_xlabel('Difference in R² (95% CI)')

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 87.5, 1.0, 'b')
    plabel(fig, 0.5, 66.0, 'c')
    save(fig, 'fig3')


# ---------------------------------------------------------------- Fig 4
BINS = v6i.BINS


def fig4():
    rel = v6i.load('v6_reliability_anatomy/result.json')
    fig = new_fig(132)

    # (a) 上：Spearman–Brown 信度；下：k 帧均值的 R²（同一 x）
    icc = rel['icc']['ICC1']
    k90 = rel['icc']['frames_for_090']
    kk = np.linspace(1, 20, 400)
    sb = kk * icc / (1 + (kk - 1) * icc)
    kc = v6i.read_csv('v6_reliability_anatomy/a2_k_curve.csv')
    k = np.array([float(r['k']) for r in kc])
    r2m = np.array([float(r['R2_mean']) for r in kc])
    r2s = np.array([float(r['R2_sd']) for r in kc])
    log(f'Fig 4a: ICC1 {icc:.4f}, frames for 0.90 {k90:.3f} (runs/v6_reliability_anatomy/result.json icc); '
        f'R² k-curve {dict(zip(k.astype(int).tolist(), np.round(r2m, 3).tolist()))} '
        f'(runs/v6_reliability_anatomy/a2_k_curve.csv)')
    ax_top = axmm(fig, 13, 5, 68, 23)
    ax_bot = axmm(fig, 13, 31, 68, 21)
    ax_top.plot(kk, sb, color=INK, lw=1.2, zorder=3)
    ax_top.plot([1], [icc], 'o', ms=3.5, mfc=INK, mec='white', mew=0.5, zorder=4)
    ax_top.text(1.6, icc, f'Single frame, ICC {icc:.2f}', ha='left', va='center', fontsize=7)
    ax_top.axhline(0.90, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax_top.plot([k90], [0.90], 'o', ms=3.5, mfc=INK, mec='white', mew=0.5, zorder=4)
    ax_top.text(k90 + 0.6, 0.885, f'0.90 at {k90:.1f} frames', ha='left', va='top', fontsize=7)
    ax_top.set_ylim(0.6, 1.02)
    ax_top.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_top.set_ylabel('Reliability')
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_bot.plot(k, r2m, color=C_MAIN, lw=1.3, zorder=3)
    ax_bot.errorbar(k, r2m, yerr=r2s, fmt='o', ms=3.2, mfc=C_MAIN, mec='white', mew=0.5, ecolor=C_MAIN,
                    elinewidth=0.8, capsize=0, zorder=4)
    ax_bot.set_ylim(0, 0.42)
    ax_bot.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax_bot.set_ylabel('R²')
    ax_bot.set_xlabel('Frames averaged per sow')
    for ax_ in (ax_top, ax_bot):
        ax_.set_xlim(0, 21)
        ax_.set_xticks([1, 5, 10, 15, 20])
        clean(ax_)
    fig.align_ylabels([ax_top, ax_bot])

    # (b) 90% 保形区间覆盖率
    axb = axmm(fig, 100, 5, 72, 47)
    cf = rel['conformal']['alpha=0.1']
    groups = [('all', 'All sows'), ('DAN_parity1', 'Danish line,\nparity 1'),
              ('NAM_multiparous', 'American line,\nparity ≥ 2')]
    meths = [('marginal', 'Marginal calibration', C_MARG, 'none'), ('mondrian', 'By line and parity', C_GRP, None)]
    for j, (mk, mn, col, mfc) in enumerate(meths):
        for g, (gk, _) in enumerate(groups):
            d = cf[mk][gk]
            xx = g + (j - 0.5) * 0.3
            lo, hi = d['cov_ci']
            axb.plot([xx, xx], [lo, hi], color=col, lw=1.2, solid_capstyle='butt', zorder=2)
            axb.plot(xx, d['coverage'], 'o', ms=4, mfc='white' if mfc else col, mec=col, mew=1.0, zorder=3)
            axb.text(xx, lo - 0.005, f'{d["width_mean"]:.1f}', ha='center', va='top', fontsize=7)
            log(f'Fig 4b: {mk:<8s} {gk:<15s} coverage {d["coverage"]:.4f} CI ({lo:.4f}, {hi:.4f}) width '
                f'{d["width_mean"]:.3f} mm (runs/v6_reliability_anatomy/result.json conformal alpha=0.1)')
    hs = [Line2D([], [], color=col, marker='o', ms=4, mfc='white' if mfc else col, mec=col, mew=1.0, lw=1.2,
                 label=mn) for _, mn, col, mfc in meths]
    axb.axhline(0.90, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    axb.legend(handles=hs, loc='upper left', ncol=2, handlelength=1.4, columnspacing=1.2, borderaxespad=0.1)
    axb.set_xticks(range(3))
    axb.set_xticklabels([g[1] for g in groups])
    axb.set_xlim(-0.5, 2.5)
    axb.set_ylim(0.78, 1.0)
    axb.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    axb.set_ylabel('Coverage of 90% interval')
    clean(axb)

    # (c) 复测分流
    axc = axmm(fig, 13, 76, 68, 44)
    tc = v6i.read_csv('v6_reliability_anatomy/triage_curve.csv')
    fr = np.array([float(r['frac']) for r in tc]) * 100
    series = [('random', 'Random', REF, 1.0, (0, (3, 2))), ('frame_sd', 'Frame or seed SD', LIGHT, 0.8, '-'),
              ('seed_sd', None, LIGHT, 0.8, '-'), ('sigma_hat', 'Fitted error scale', C_MAIN, 1.5, '-'),
              ('oracle', 'Oracle (true error)', INK, 1.3, '-')]
    for key, name, col, lw, ls in series:
        v = np.array([float(r[f'{key}_mae_kept']) for r in tc])
        axc.plot(fr, v, color=col, lw=lw, ls=ls, zorder=3, label=name or '_')
    axc.legend(loc='lower left', handlelength=2.0, labelspacing=0.3)
    o30 = [float(r['oracle_mae_kept']) for r in tc if abs(float(r['frac']) - 0.3) < 1e-9][0]
    axc.plot([30], [o30], 'o', ms=3.5, mfc=INK, mec='white', mew=0.5, zorder=4)
    axc.text(31.5, o30 + 0.05, f'{o30:.2f} mm', ha='left', va='bottom', fontsize=7)
    log(f'Fig 4c: oracle MAE kept at 30% = {o30:.4f} mm; start MAE {float(tc[0]["random_mae_kept"]):.4f} '
        f'(runs/v6_reliability_anatomy/triage_curve.csv, model {tc[0]["model"]})')
    axc.set_xlim(0, 52)
    axc.set_ylim(0, 2.8)
    axc.set_xticks([0, 10, 20, 30, 40, 50])
    axc.set_xlabel('Sows re-measured by ultrasound (%)')
    axc.set_ylabel('MAE of remaining sows (mm)')
    clean(axc)

    # (d) 按 B 超背膘分段的误差
    axd = axmm(fig, 100, 76, 72, 44)
    rows = v6i.read_csv('v6_anatomy/prof/test_preds.csv')
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
    bp = axd.boxplot(data, positions=xb, widths=0.52, patch_artist=True, showfliers=True,
                     medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=AXC, lw=0.6),
                     capprops=dict(color=AXC, lw=0.6),
                     flierprops=dict(marker='o', ms=2.0, mfc=C_MAIN, mec='none', alpha=0.7))
    for b in bp['boxes']:
        b.set(facecolor=MAIN_FILL2, edgecolor=AXC, lw=0.6)
    axd.axhline(0, color=REF, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for x_, mae in zip(xb, maes):
        axd.text(x_, 12.8, f'{mae:.1f}', ha='center', va='center', fontsize=7)
    axd.text(-0.62, 12.8, 'MAE', ha='right', va='center', fontsize=7)
    axd.set_xticks(xb)
    axd.set_xticklabels([f'{l}\n({n})' for l, n in zip(labs, ns)])
    axd.set_xlim(-0.6, len(data) - 0.4)
    axd.set_ylim(min(-10.5, min(d.min() for d in data) - 0.8), 14.5)
    axd.set_yticks([-10, -5, 0, 5, 10])
    axd.set_xlabel('Backfat, ultrasound (mm)')
    axd.set_ylabel('Prediction − ultrasound (mm)')
    clean(axd)
    log('Fig 4d bins (runs/v6_anatomy/prof/test_preds.csv pred_inner): ' + ', '.join(
        f'{l} n={n} MAE={m:.2f} mean err={d.mean():+.2f}' for l, n, m, d in zip(labs, ns, maes, data)))

    plabel(fig, 0.5, 1.0, 'a')
    plabel(fig, 87.5, 1.0, 'b')
    plabel(fig, 0.5, 70.0, 'c')
    plabel(fig, 87.5, 70.0, 'd')
    save(fig, 'fig4')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', nargs='+', type=int, default=[2, 3, 4])
    a = ap.parse_args()
    log(f'== v6m_figures.py --fig {" ".join(map(str, a.fig))}')
    for n in a.fig:
        {1: fig1, 2: fig2, 3: fig3, 4: fig4}[n]()


if __name__ == '__main__':
    main()
