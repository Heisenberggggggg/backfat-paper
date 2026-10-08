# -*- coding: utf-8 -*-
"""
build_index.py — 十月论文（gComm）双视角数据索引 v1

来源 / 背景
  十月论文只用丰美禾 + DK（省农科院）两场数据（见 课题整体规划研究/03_… v3 修订）。
  两场每头猪都有 master（俯视）+ sub*（后方斜拍）两个视角。旧脚本（generate_depthmap_dataset.py）
  只登记了 master，且丰美禾用「标签耳号是目录名子串」匹配，未检查一对多冲突。

本脚本做什么（只读原始数据）
  1. 复用 generate_depthmap_dataset.py 的标签读取函数（docx / xlsx）
  2. 为每头猪登记 master 与 sub 两个视角的 color/depth 目录、帧数、各自相机内参
  3. 丰美禾标签匹配：记录所有命中的耳号；取最长命中；命中多个且背膘值不一致 → label_conflict=1
  4. 生成 group_id（场 + 耳号），同一头猪跨日期重复采集时归入同一组，供后续猪级划分防泄漏
  输出 index_v1.csv（UTF-8 BOM，Excel 可直接打开）；不过滤背膘范围，过滤在训练阶段做。

用法
  D:/Anaconda/envs/pytorch/python.exe build_index.py
"""
import csv
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'yolo+FastSAM'))
from generate_depthmap_dataset import (  # noqa: E402  复用已验证的标签读取逻辑
    load_labels_from_docx_nky1, load_labels_from_docx_nky2, load_labels_from_docx_nky3,
    load_labels_from_xlsx_fmh, load_labels_from_xlsx_fmh_nov)

DATA_ROOT = r'G:\耿来硕数据'
DK_BASE = os.path.join(DATA_ROOT, 'DK')
FMH_BASE = os.path.join(DATA_ROOT, '丰美禾数据')
FMH_DATES = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
             '11.1', '11.2', '11.3', '11.4', '11.5']
OUT_CSV = os.path.join(HERE, 'index_v2.csv')  # v2：丰美禾按日期分节匹配标签（v1 为全表子串匹配，保留作对照）


def read_intrinsic(view_dir):
    p = os.path.join(view_dir, 'intrinsic.json')
    if not os.path.isfile(p):
        return {}
    with open(p, encoding='utf-8') as f:
        j = json.load(f)
    m = j.get('intrinsic_matrix', [])
    if len(m) != 9:
        return {}
    # Open3D / Azure Kinect 导出为列主序：[fx,0,0, 0,fy,0, cx,cy,1]
    return dict(fx=m[0], fy=m[4], cx=m[6], cy=m[7], w=j.get('width'), h=j.get('height'),
                sn=j.get('serial_number', ''), depth_mode=j.get('depth_mode', ''))


def view_info(view_dir):
    c, d = os.path.join(view_dir, 'color'), os.path.join(view_dir, 'depth')
    if not (os.path.isdir(c) and os.path.isdir(d)):
        return None
    cs = {os.path.splitext(f)[0] for f in os.listdir(c) if f.endswith(('.jpg', '.png'))}
    ds = {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith('.png')}
    return dict(dir=view_dir, n=len(cs & ds), intr=read_intrinsic(view_dir))


def find_sub(parent):
    """在 parent 下找 sub* 视角目录（sub / sub2 / sub10 …），返回第一个有效的。"""
    if not os.path.isdir(parent):
        return None, None
    for name in sorted(os.listdir(parent)):
        if name.lower().startswith('sub'):
            v = view_info(os.path.join(parent, name))
            if v and v['n'] > 0:
                return name, v
    return None, None


def row(uid, farm, batch, pig_dir, ear_key, breed, bf, label_src, n_match, conflict, master, sub_name, sub):
    r = dict(uid=uid, farm=farm, batch=batch, pig_dir=pig_dir, ear_key=ear_key,
             group_id=f'{farm}_{ear_key}', breed=breed, backfat=bf, label_src=label_src,
             n_label_matches=n_match, label_conflict=int(conflict))
    for tag, v in (('master', master), ('sub', sub)):
        r[f'{tag}_dir'] = v['dir'] if v else ''
        r[f'n_{tag}'] = v['n'] if v else 0
        it = v['intr'] if v else {}
        for k in ('fx', 'fy', 'cx', 'cy', 'sn', 'depth_mode'):
            r[f'{tag}_{k}'] = it.get(k, '')
    r['sub_name'] = sub_name or ''
    return r


def build_dk():
    rows = []
    specs = [('nkyfirst', load_labels_from_docx_nky1(os.path.join(DK_BASE, '第一次数据7.29.docx'))),
             ('nkysecond', load_labels_from_docx_nky2(os.path.join(DK_BASE, '第二次数据9.6.docx'))),
             ('nkythird', load_labels_from_docx_nky3(os.path.join(DK_BASE, '第三次数据12.9.docx')))]
    for batch, labels in specs:
        bdir = os.path.join(DK_BASE, batch)
        if batch == 'nkythird':
            pig_paths = [(os.path.join(bdir, k, p), p) for k in os.listdir(bdir)
                         if os.path.isdir(os.path.join(bdir, k)) for p in os.listdir(os.path.join(bdir, k))]
        else:
            pig_paths = [(os.path.join(bdir, p), p) for p in os.listdir(bdir)]
        for pig_path, pig in pig_paths:
            if not os.path.isdir(pig_path):
                continue
            if batch == 'nkythird':
                if pig.startswith(('w', 'wc')):
                    continue
                m = re.match(r'^[cwk]*(\d+)-(.+)$', pig)
                key = f'{m.group(1)}-{m.group(2)}' if m else None
            else:
                key = pig
            if key not in labels:
                continue
            base = os.path.join(pig_path, 'depth_cz_r_d_p')
            master = None
            for mname in (['master10', 'master'] if pig.startswith('6-') else ['master', 'master10']):
                master = view_info(os.path.join(base, mname))
                if master:
                    break
            sub_name, sub = find_sub(base)
            rows.append(row(f'dk_{batch}_{pig}', 'DK', batch, pig_path, f'{batch}:{key}', '',
                            labels[key], 'docx', 1, False, master, sub_name, sub))
    return rows


def load_fmh_sections():
    """
    v2：按「记录日期」分节读取丰美禾两份标签表。
    表结构：节标题行含「记录日期：8月24日」或「日期：10月29日」，其后为
      序号 | 耳标号 | 背膘值 | 体况（1-5分） | 胎次 | 配种日期(如 6.17 = 6月17日) | …
    返回 {(月, 日): [dict(tag, backfat, bcs, parity, mating), ...]}
    """
    import openpyxl
    secs = {}
    for fn in ('背膘记录表.xlsx', '11.10处理背膘数据.xlsx'):
        ws = openpyxl.load_workbook(os.path.join(FMH_BASE, fn), read_only=True, data_only=True).worksheets[0]
        cur = None
        for r in ws.iter_rows(values_only=True):
            s = ' '.join(str(c) for c in r[:10] if c is not None)
            m = re.search(r'日期[:：]\s*(\d+)月(\d+)日', s)
            if m:
                cur = (int(m.group(1)), int(m.group(2)))
                secs.setdefault(cur, [])
                continue
            # 11.10 表部分节的「序号」列为空，因此不再要求序号为数字；以「耳标号含数字 + 背膘可解析为数」判定数据行
            if cur is None or r[1] is None or not re.search(r'\d', str(r[1])):
                continue
            tag = str(r[1]).strip()
            try:
                bf = float(r[2])
            except (TypeError, ValueError):
                continue
            if not tag:
                continue
            secs[cur].append(dict(tag=tag, backfat=bf, bcs=r[3], parity=r[4], mating=r[5]))
    return secs


def gest_day(date_str, mating):
    """拍摄日期 '8.24' 与配种日期 6.17（float 或 '6.17'）→ 妊娠日龄；无法解析返回 ''。"""
    from datetime import date
    try:
        cm, cd = map(int, date_str.split('.'))
        mm_s, md_s = str(mating).strip().split('.')
        mm = int(mm_s)
    except (ValueError, AttributeError):
        return ''
    # Excel 把 7.20 存成浮点 7.2 → 一位小数时「2日」与「20日」都可能；两者都合理则视为无法判定
    days = [int(md_s)] if len(md_s) >= 2 else [int(md_s), int(md_s) * 10]
    cands = []
    for md in days:
        try:
            g = (date(2025, cm, cd) - date(2025 if mm <= cm else 2024, mm, md)).days
        except ValueError:
            continue
        if 0 <= g <= 118:
            cands.append(g)
    return cands[0] if len(cands) == 1 else ''


def build_fmh():
    secs = load_fmh_sections()
    rows = []
    for date in FMH_DATES:
        ddir = os.path.join(FMH_BASE, date)
        if not os.path.isdir(ddir):
            continue
        key = tuple(map(int, date.split('.')))
        sec = secs.get(key, [])
        # 采集季：8 月一季（8/13–9/29 各节），10–11 月一季。同一头猪两季各测一次，背膘不同，绝不能跨季匹配
        season = [e for k, v in secs.items() if (k[0] >= 10) == (key[0] >= 10) for e in v]
        for pig in os.listdir(ddir):
            pig_path = os.path.join(ddir, pig)
            if not os.path.isdir(pig_path):
                continue
            # 目录名 = 全局序号 + 耳号 → 耳号应为目录名后缀。先同日期节内匹配，失败再退到同季
            hits = [e for e in sec if pig.endswith(e['tag'])]
            level = 'date'
            if not hits:
                hits = [e for e in season if pig.endswith(e['tag'])]
                level = 'season'
            if not hits:
                continue
            best_len = max(len(e['tag']) for e in hits)
            best = [e for e in hits if len(e['tag']) == best_len]
            conflict = len({e['backfat'] for e in best}) > 1
            e = best[0]
            m = re.match(r'^\d+([A-Za-z]+)', pig)
            breed = m.group(1).upper() if m else ''
            ear_key = re.sub(r'^\d+', '', pig) if m else e['tag']  # 去掉目录名前面的全局序号
            master = view_info(os.path.join(pig_path, 'master'))
            sub_name, sub = find_sub(pig_path)
            r = row(f'fmh_{date}_{pig}', 'FMH', date, pig_path, ear_key, breed, e['backfat'],
                    f'xlsx:{date}:{e["tag"]}', len(hits), conflict, master, sub_name, sub)
            r.update(parity=e['parity'] if e['parity'] is not None else '', bcs=e['bcs'] if e['bcs'] is not None else '',
                     mating=e['mating'] if e['mating'] is not None else '', gest_day=gest_day(date, e['mating']),
                     match_level=level)
            rows.append(r)
    return rows


def main():
    rows = build_dk() + build_fmh()
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader()
        w.writerows(rows)

    # 与旧深度图数据集（generate_depthmap_dataset.py 的子串匹配）标签对比
    old_meta = os.path.join(HERE, '..', 'yolo+FastSAM', 'depthmap_dataset', 'metadata.csv')
    if os.path.isfile(old_meta):
        old = {}
        with open(old_meta, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                old[r['pig_id']] = float(r['backfat'])
        diffs = [(r['uid'], old[r['uid']], r['backfat']) for r in rows
                 if r['farm'] == 'FMH' and r['uid'] in old]
        changed = [d for d in diffs if abs(d[1] - d[2]) > 1e-6]
        print(f'[旧标签对比] 丰美禾可对比 {len(diffs)} 头，标签不同 {len(changed)} 头；'
              f'|差| 均值 {sum(abs(a - b) for _, a, b in changed) / max(len(changed), 1):.2f} mm')
        for u, a, b in changed[:8]:
            print(f'    {u}: 旧 {a} → 新 {b}')

    print(f'写出 {OUT_CSV}，共 {len(rows)} 行')
    for farm in ('DK', 'FMH'):
        rs = [r for r in rows if r['farm'] == farm]
        both = [r for r in rs if r['n_master'] > 0 and r['n_sub'] > 0]
        bfs = [r['backfat'] for r in rs]
        grp = Counter(r['group_id'] for r in rs)
        print(f'\n[{farm}] 有标签 {len(rs)} 头 | 双视角齐全 {len(both)} | 仅 master '
              f'{sum(1 for r in rs if r["n_master"] and not r["n_sub"])} | 仅 sub '
              f'{sum(1 for r in rs if r["n_sub"] and not r["n_master"])}')
        print(f'  背膘 min/max {min(bfs)}/{max(bfs)}；[7,32] 内 {sum(7 <= b <= 32 for b in bfs)} 头')
        print(f'  group 数 {len(grp)}，重复采集的猪（同 group 多行）{sum(1 for c in grp.values() if c > 1)}')
        print(f'  标签一对多命中 {sum(1 for r in rs if r["n_label_matches"] > 1)}，'
              f'其中背膘冲突 {sum(r["label_conflict"] for r in rs)}')
        print(f'  sub 目录名分布 {dict(Counter(r["sub_name"] for r in rs))}')
        print(f'  相机 SN（master）{dict(Counter(r["master_sn"] for r in rs))}')
        print(f'  相机 SN（sub）{dict(Counter(r["sub_sn"] for r in rs))}')
        if farm == 'FMH':
            print(f'  品系 {dict(Counter(r["breed"] for r in rs))}')


if __name__ == '__main__':
    main()
