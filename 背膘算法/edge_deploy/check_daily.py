# -*- coding: utf-8 -*-
"""
每日采集数据校验脚本
================================================================
用途
    每天收工前跑一次，把"今天到底采到了什么"变成一页纸的确定结论。
    现场问题必须当天发现 —— 隔一夜猪群与设备状态就变了，补不回来。

校验内容
    1) 目录规范：{root}/{YYYYMMDD}/{sta_id}/{earid5}_{visit_seq}/
    2) 帧完整性：color/ 与 depth/ 数量一致、不低于阈值、无 0 字节文件
    3) meta.json 存在且字段齐全，且与目录名自洽
    4) 耳标撞号：同一天不同饲喂站出现相同 earid5
    5) 磁盘占用与按当前速率的剩余可用天数
    6) （可选）与服务器 JSON 导出比对，找出"有上报无图像"和"有图像无上报"

退出码
    0 = 全部通过；1 = 有告警；2 = 有致命问题（当天必须处理）

用法
    python check_daily.py --root D:/nky_feeder_2026 --date 20260819
    python check_daily.py --root D:/nky_feeder_2026 --date 20260819 \
        --server-json server_20260819.json

Author: 耿来硕 + Claude
Date:   2026-08-18
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict

DIR_RE = re.compile(r'^(\d{5})_(\d+)$')
META_KEYS = ['earid', 'sta_id', 'visit_seq', 'date', 'n_frames']
IMG_EXT = ('.png', '.jpg', '.jpeg')


class Report:
    def __init__(self):
        self.lines = []
        self.warn = 0
        self.fatal = 0

    def add(self, s=''):
        self.lines.append(s)

    def w(self, s):
        self.warn += 1
        self.lines.append('  [告警] ' + s)

    def f(self, s):
        self.fatal += 1
        self.lines.append('  [致命] ' + s)

    def text(self):
        return '\n'.join(self.lines)


def count_imgs(d):
    if not os.path.isdir(d):
        return None, []
    files = [f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
    empty = [f for f in files if os.path.getsize(os.path.join(d, f)) == 0]
    return len(files), empty


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser(description='每日采集数据校验')
    ap.add_argument('--root', required=True, help='数据根目录，如 D:/nky_feeder_2026')
    ap.add_argument('--date', default=time.strftime('%Y%m%d'),
                    help='YYYYMMDD，默认今天')
    ap.add_argument('--min-frames', type=int, default=100,
                    help='每次访问应有的帧数下限（方案定为 100）')
    ap.add_argument('--server-json', default=None,
                    help='服务器导出的当日 JSON（数组，或每行一条 JSON）')
    ap.add_argument('--out', default=None, help='报告输出路径，默认写到 root 下')
    a = ap.parse_args()

    day_dir = os.path.join(a.root, a.date)
    r = Report()
    r.add('=' * 68)
    r.add('每日采集校验报告   日期 %s   生成于 %s'
          % (a.date, time.strftime('%Y-%m-%d %H:%M:%S')))
    r.add('根目录 %s' % a.root)
    r.add('=' * 68)

    if not os.path.isdir(day_dir):
        r.add()
        r.f('当日目录不存在：%s —— 今天很可能一帧都没采到。' % day_dir)
        print(r.text())
        return 2

    # ---------------- 遍历 ----------------
    visits = []                       # (sta, earid5, seq, path)
    r.add()
    r.add('【1】目录结构与帧完整性')
    for sta in sorted(os.listdir(day_dir)):
        sta_dir = os.path.join(day_dir, sta)
        if not os.path.isdir(sta_dir):
            continue
        for name in sorted(os.listdir(sta_dir)):
            vdir = os.path.join(sta_dir, name)
            if not os.path.isdir(vdir):
                continue
            m = DIR_RE.match(name)
            if not m:
                r.w('目录名不符合 {earid5}_{visit_seq} 规范：%s'
                    % os.path.join(sta, name))
                continue
            earid5, seq = m.group(1), int(m.group(2))
            visits.append((sta, earid5, seq, vdir))

            nc, empty_c = count_imgs(os.path.join(vdir, 'color'))
            nd, empty_d = count_imgs(os.path.join(vdir, 'depth'))
            tag = '%s/%s' % (sta, name)
            if nc is None:
                r.f('%s 缺少 color/ 目录' % tag)
                continue
            if nd is None:
                r.f('%s 缺少 depth/ 目录 —— 无深度则该次访问不可用于建模' % tag)
                continue
            if nc == 0 or nd == 0:
                r.f('%s 空目录（color=%d, depth=%d）' % (tag, nc, nd))
                continue
            if nc != nd:
                r.w('%s color/depth 帧数不一致：%d vs %d' % (tag, nc, nd))
            if min(nc, nd) < a.min_frames:
                r.w('%s 帧数不足：%d < %d' % (tag, min(nc, nd), a.min_frames))
            if empty_c or empty_d:
                r.w('%s 存在 0 字节文件：color %d 个, depth %d 个'
                    % (tag, len(empty_c), len(empty_d)))

            mp = os.path.join(vdir, 'meta.json')
            if not os.path.exists(mp):
                r.w('%s 缺少 meta.json' % tag)
            else:
                try:
                    meta = json.load(open(mp, encoding='utf-8'))
                    miss = [k for k in META_KEYS if k not in meta]
                    if miss:
                        r.w('%s meta.json 缺字段：%s' % (tag, ','.join(miss)))
                    if str(meta.get('earid', '')).zfill(5) != earid5:
                        r.f('%s meta.json 的 earid=%s 与目录名 %s 不一致'
                            % (tag, meta.get('earid'), earid5))
                    if str(meta.get('date', '')) != a.date:
                        r.w('%s meta.json 的 date=%s 与所在日期目录不一致'
                            % (tag, meta.get('date')))
                except Exception as e:
                    r.w('%s meta.json 解析失败：%s' % (tag, e))

            if not os.path.exists(os.path.join(vdir, 'intrinsic.json')):
                r.w('%s 缺少 intrinsic.json —— 无内参无法反投影' % tag)

    r.add('  当日访问目录共 %d 个，涉及饲喂站 %d 台'
          % (len(visits), len(set(v[0] for v in visits))))

    # ---------------- 撞号 ----------------
    r.add()
    r.add('【2】耳标撞号检查（同一 earid5 出现在多台饲喂站）')
    by_ear = defaultdict(set)
    for sta, earid5, _, _ in visits:
        by_ear[earid5].add(sta)
    clash = {e: s for e, s in by_ear.items() if len(s) > 1}
    if clash:
        for e, s in sorted(clash.items()):
            r.f('耳标 %s 同时出现在饲喂站 %s —— 低5位撞号或猪只串栏，'
                '必须人工确认后才能用于建模' % (e, '、'.join(sorted(s))))
    else:
        r.add('  未发现撞号。')

    # ---------------- 访问次数分布 ----------------
    r.add()
    r.add('【3】每头猪当日访问次数')
    per_pig = defaultdict(list)
    for sta, earid5, seq, _ in visits:
        per_pig[(sta, earid5)].append(seq)
    for (sta, e), seqs in sorted(per_pig.items()):
        if sorted(seqs) != list(range(1, len(seqs) + 1)):
            r.w('站%s 耳标%s 的 visit_seq 不连续：%s' % (sta, e, sorted(seqs)))
    r.add('  当日有图像的猪只 %d 头，访问 %d 次（均值 %.2f 次/头）'
          % (len(per_pig), len(visits), len(visits) / max(1, len(per_pig))))

    # ---------------- 与服务器 JSON 比对 ----------------
    r.add()
    if a.server_json:
        r.add('【4】与服务器上报记录比对')
        try:
            raw = open(a.server_json, encoding='utf-8').read().strip()
            try:
                recs = json.loads(raw)
                if isinstance(recs, dict):
                    recs = [recs]
            except json.JSONDecodeError:
                recs = [json.loads(x) for x in raw.splitlines() if x.strip()]

            srv = set()
            for rec in recs:
                if str(rec.get('flag')) != '1':
                    continue          # 0=心跳 2=状态，不是测定数据
                e = str(rec.get('earid', '')).zfill(5)[-5:]
                srv.add((str(rec.get('sta_id')), e, int(rec.get('visit_seq', 0))))
            img = set((s, e, q) for s, e, q, _ in visits)

            only_srv = sorted(srv - img)
            only_img = sorted(img - srv)
            r.add('  服务器有效上报 %d 条；本地图像会话 %d 个' % (len(srv), len(img)))
            if only_srv:
                r.w('有上报但无图像 %d 条（前10）：%s'
                    % (len(only_srv), only_srv[:10]))
                r.add('        → 优先怀疑：上位机未收到 11 指令 / 相机掉线 / 落盘失败')
            if only_img:
                r.w('有图像但无上报 %d 条（前10）：%s'
                    % (len(only_img), only_img[:10]))
                r.add('        → 优先怀疑：4G 未上报成功 / 服务器导出不全')
            if not only_srv and not only_img:
                r.add('  两侧完全对齐。')
        except Exception as e:
            r.w('服务器 JSON 读取失败：%s' % e)
    else:
        r.add('【4】与服务器上报记录比对：未提供 --server-json，跳过')

    # ---------------- 磁盘 ----------------
    r.add()
    r.add('【5】磁盘占用与余量')
    gb = dir_size(day_dir) / 1024 ** 3
    try:
        free_gb = shutil.disk_usage(a.root).free / 1024 ** 3
        r.add('  当日新增 %.2f GB；磁盘剩余 %.1f GB' % (gb, free_gb))
        if gb > 0.01:
            days = free_gb / gb
            r.add('  按当日速率估算，还可采集约 %.0f 天' % days)
            if days < 10:
                r.f('剩余不足 10 天，立即准备换盘')
            elif days < 25:
                r.w('剩余不足 25 天，提前安排换盘')
    except Exception as e:
        r.w('磁盘信息读取失败：%s' % e)

    # ---------------- 结论 ----------------
    r.add()
    r.add('=' * 68)
    if r.fatal:
        verdict, code = '不通过 —— 有 %d 项致命问题，今天必须处理' % r.fatal, 2
    elif r.warn:
        verdict, code = '基本通过 —— 有 %d 项告警，建议今天处理' % r.warn, 1
    else:
        verdict, code = '全部通过', 0
    r.add('结论：%s' % verdict)
    r.add('=' * 68)

    out = a.out or os.path.join(a.root, 'check_%s.txt' % a.date)
    txt = r.text()
    try:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(txt + '\n')
    except Exception as e:
        print('[warn] 报告写入失败：%s' % e, file=sys.stderr)
    print(txt)
    print('\n报告已保存：%s' % out)
    return code


if __name__ == '__main__':
    sys.exit(main())
