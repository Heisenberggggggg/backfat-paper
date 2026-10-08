# -*- coding: utf-8 -*-
"""
从原始数据盘抽取 Jetson 演示子集
================================================================
为什么需要这个脚本
    原始数据盘（H:）里每头猪除 color/ depth/ 外还有十几个点云派生目录
    （clustered/ jl/ tj/ zt/ frame_pointclouds/ ...），体积是有用数据的几十倍。
    Jetson 上跑 demo 只需要 color + depth + intrinsic.json。

    更重要的是：**不要把原始数据盘直接接到 Jetson 上。**
    那是 916 头标签数据的唯一副本，接到 Linux 上误写/误删的代价无法承受，
    而且演示实际只需要几百 MB。抽子集拷到 U 盘或 Jetson 本地盘即可。

输出目录直接采用 D1 方案 §3.3 的规范布局，因此它同时也是一份布局样例。

用法
    python pack_demo_subset.py --src "H:/耿来硕数据/DK/nkyfirst" \
        --dst D:/jetson_demo --pigs 3 --stride 3

Author: 耿来硕 + Claude
Date:   2026-08-18
"""

import argparse
import json
import os
import shutil
import sys

IMG_EXT = ('.png', '.jpg', '.jpeg')


def find_master(pig_dir):
    """在一头猪的目录里找出含 color/ 与 depth/ 的那一层。

    盘上有两种布局：
      nkyfirst/second: {pig}/depth_cz_r_d_p/master/{color,depth}
      丰美禾:          {date}/{pig}/master/{color,depth}
    直接搜比写死路径稳。
    """
    for root, dirs, _ in os.walk(pig_dir):
        if 'color' in dirs and 'depth' in dirs:
            return root
        if root.count(os.sep) - pig_dir.count(os.sep) > 3:
            dirs[:] = []
    return None


def listimgs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def main():
    ap = argparse.ArgumentParser(description='抽取 Jetson 演示子集')
    ap.add_argument('--src', required=True, help='源目录，如 H:/耿来硕数据/DK/nkyfirst')
    ap.add_argument('--dst', required=True, help='目标目录，如 D:/jetson_demo')
    ap.add_argument('--pigs', type=int, default=3, help='抽几头猪')
    ap.add_argument('--stride', type=int, default=1,
                    help='帧抽样步长，3 表示每 3 帧取 1 帧')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='每头猪最多保留多少帧，0=不限')
    ap.add_argument('--date', default='20260819', help='输出目录用的日期')
    ap.add_argument('--sta-id', default='1')
    a = ap.parse_args()

    pigs = sorted(x for x in os.listdir(a.src)
                  if os.path.isdir(os.path.join(a.src, x)))[:a.pigs]
    if not pigs:
        raise SystemExit('源目录下没有猪只目录：%s' % a.src)

    total = 0
    made = []
    for i, pig in enumerate(pigs):
        m = find_master(os.path.join(a.src, pig))
        if m is None:
            print('[skip] %s 未找到 color/depth' % pig, file=sys.stderr)
            continue
        cs, ds = listimgs(os.path.join(m, 'color')), listimgs(os.path.join(m, 'depth'))
        n = min(len(cs), len(ds))
        idx = list(range(0, n, max(1, a.stride)))
        if a.max_frames:
            idx = idx[:a.max_frames]

        earid5 = str(10000 + i)          # 演示用占位耳标，非真实耳标
        out = os.path.join(a.dst, a.date, a.sta_id, '%s_1' % earid5)
        os.makedirs(os.path.join(out, 'color'), exist_ok=True)
        os.makedirs(os.path.join(out, 'depth'), exist_ok=True)

        for k, j in enumerate(idx):
            for sub, files in (('color', cs), ('depth', ds)):
                src = os.path.join(m, sub, files[j])
                ext = os.path.splitext(files[j])[1]
                dst = os.path.join(out, sub, 'frame_%03d%s' % (k, ext))
                shutil.copy2(src, dst)
                total += os.path.getsize(dst)

        ip = os.path.join(m, 'intrinsic.json')
        if os.path.exists(ip):
            shutil.copy2(ip, os.path.join(out, 'intrinsic.json'))
            total += os.path.getsize(ip)

        json.dump({'earid': earid5, 'earid_full': '(演示占位)', 'sta_id': int(a.sta_id),
                   'visit_seq': 1, 'date': a.date, 'n_frames': len(idx),
                   'source_pig': pig, 'source_dir': m,
                   'note': '演示子集，耳标为占位值，不可用于建模统计'},
                  open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        made.append((pig, earid5, len(idx)))
        print('  %s -> %s  (%d 帧)' % (pig, os.path.relpath(out, a.dst), len(idx)))

    print('\n共 %d 头，合计 %.1f MB' % (len(made), total / 1024 / 1024))
    print('目标目录：%s' % a.dst)
    if made:
        d = os.path.join(a.dst, a.date, a.sta_id, '%s_1' % made[0][1])
        print('')
        print('Jetson 上跑 demo（下面三行接成一条命令）：')
        print('  python3 backfat_edge_demo.py --source replay')
        print('      --dir %s' % d.replace(chr(92), '/'))
        print('      --weights best_model_depthmap.pth --device-name Jetson')


if __name__ == '__main__':
    main()
