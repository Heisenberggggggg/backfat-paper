# -*- coding: utf-8 -*-
"""
v6o_rerun_subset.py — 在剔除后的 622 条子集上重跑论文全部分析（总调度脚本，可并行、可断点续跑）

背景：DECISIONS 2026-10-20（二）。先在笔记本上运行 v6o_make_exclusions.py、v6o_make_subset.py 得到 runs_v6o/（种子目录），
再在实验室电脑（RTX 5090）上运行本脚本。所有步骤通过环境变量 BACKFAT_RUNS / BACKFAT_INDEX（见 v6o_paths.py）
读写 runs_v6o/，不碰 runs/ 里的旧结果。

步骤 = 稿件【来源】引用到的全部脚本（共 40 步），按依赖关系调度：
  - gpu 类（网络训练、TabPFN）最多同时运行 --gpu-jobs 个；cpu 类最多 --cpu-jobs 个；就绪步骤中耗时长的先启动；
  - 每步完成后写 _driver/done/<步骤>.ok；重新运行时跳过已完成的步骤（--redo 指定重跑）；
  - 某步失败：它的下游不再启动，其余步骤继续；结束时列出失败步骤，退出码 1；
  - 每步的输出在 _driver/logs/<步骤>.txt；总日志 _driver/driver_log.txt；环境信息 _driver/env.json；汇总 _driver/summary.md。
唯一的内部步骤 crop_frames：把 v3b master（裁剪深度图）种子 42 的逐帧预测写成 framepreds_new_s128/frame_preds.csv，
供 analysis_a1_a2_a6.py 计算裁剪深度图的单帧组内相关（原来用 cv_depthmap_framepreds.py 的旧运行，本次改用同批重跑的 v3b）。
不在本脚本里：v6j 部署权重与 Jetson 测速（与样本集无关，沿用 10-16 实测）；作图（结果拷回笔记本后用 figs/v6m_figures.py）。

用法（在 twoview 目录下）：
  python v6o_rerun_subset.py --list                 # 列出步骤与依赖
  python v6o_rerun_subset.py --dry-run              # 只打印调度顺序
  python v6o_rerun_subset.py --gpu-jobs 3           # 正式运行（5090 32 GB：建议 3，显存充裕可 4）
  python v6o_rerun_subset.py --only v1_tab v1b      # 只跑指定步骤（其依赖须已完成或同在 --only 中）
  python v6o_rerun_subset.py --redo v4b             # 删除该步完成标记后重跑（输出目录由各脚本自行覆盖 / 追加）
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding='utf-8')
SEEDS5 = ['42', '43', '44', '45', '46']
SMOKE_EPOCHS = {'v3b_view_models.py', 'v4b_hmap_models.py', 'v6b_dualview_hmap.py', 'v6f_anatomy_net.py',
                'v6g_anatomy_nested.py', 'v4h_hmap_nested_fusion.py', 'v5a_multitask_hmap.py'}
SMOKE_SEEDS = {'v3b_view_models.py', 'v4b_hmap_models.py', 'v6b_dualview_hmap.py', 'v6f_anatomy_net.py',
               'v5a_multitask_hmap.py'}
# 需要 statsmodels 的步骤：笔记本上 pytorch 环境没有 statsmodels，这几步历来用基础环境跑（--python-stats）
STATS_STEPS = {'a1a2a6', 'v4g', 'an_cal', 'v6c_anat', 'v6c_resnet'}


def steps(R):
    """(名称, 命令参数或 None = 内部步骤, 依赖, 类型, 笔记本 RTX 4060 上的大致分钟数)"""
    j = lambda *p: os.path.join(R, *p)  # noqa: E731
    S = [
        ('v1_tab', ['v1_tabular_baseline.py'], [], 'gpu', 1.5),
        ('v1b', ['v1b_caliper_vs_vision.py'], ['v1_tab'], 'cpu', 0.7),
        # 表示与视角（ResNet-18 基线）
        ('v3b_master', ['v3b_view_models.py', '--view', 'master'], ['v1_tab'], 'gpu', 15),
        ('v3b_sub', ['v3b_view_models.py', '--view', 'sub'], ['v1_tab'], 'gpu', 14),
        ('v3b_dual', ['v3b_view_models.py', '--view', 'dual'], ['v1_tab'], 'gpu', 33),
        ('v4b', ['v4b_hmap_models.py'], ['v1_tab'], 'gpu', 34),
        ('v6b', ['v6b_dualview_hmap.py'], ['v4b', 'v3b_sub'], 'gpu', 42),
        # 解剖引导网络（主模型与同代码基线 5 种子；多示例变体 3 种子）
        ('v6f_prof', ['v6f_anatomy_net.py', '--variant', 'prof', '--seeds'] + SEEDS5, ['v1_tab'], 'gpu', 57),
        ('v6f_base', ['v6f_anatomy_net.py', '--variant', 'base', '--seeds'] + SEEDS5, ['v1_tab'], 'gpu', 55),
        ('v6f_mil', ['v6f_anatomy_net.py', '--variant', 'mil'], ['v1_tab'], 'gpu', 32),
        ('v6f_profmil', ['v6f_anatomy_net.py', '--variant', 'prof_mil'], ['v1_tab'], 'gpu', 35),
        ('v6g', ['v6g_anatomy_nested.py'], ['v1_tab'], 'gpu', 63),
        ('v4h', ['v4h_hmap_nested_fusion.py'], ['v4b', 'v1b'], 'gpu', 49),
        # 训练目标变体
        ('v4e', ['v4e_body_dims_fusion.py'], ['v1b'], 'gpu', 1.5),
        ('v5a_cal', ['v5a_multitask_hmap.py', '--mode', 'cal'], ['v1b', 'v4e'], 'gpu', 28),
        ('v5a_caldims', ['v5a_multitask_hmap.py', '--mode', 'caldims'], ['v1b', 'v4e'], 'gpu', 28),
        ('v5a_lds', ['v5a_multitask_hmap.py', '--mode', 'lds'], ['v1b', 'v4e'], 'gpu', 27),
        ('v5a_nll', ['v5a_multitask_hmap.py', '--mode', 'nll'], ['v1b', 'v4e'], 'gpu', 27),
        # 冻结基础模型特征
        ('a3_dinov3_relh', ['a3_foundation_regression.py', '--model', 'dinov3-vitb16', '--enc', 'relh', '--vis', 'stack'],
         ['v1b'], 'gpu', 3),
        ('a3_dinov3_hmap', ['a3_foundation_regression.py', '--model', 'dinov3-vitb16', '--enc', 'hmap', '--vis', 'stack'],
         ['v1b'], 'gpu', 3),
        ('a3_dinov2_relh', ['a3_foundation_regression.py', '--model', 'dinov2-base', '--enc', 'relh', '--vis', 'stack'],
         ['v1b'], 'gpu', 3),
        # 虚拟卡尺、体尺、相机 + 卡尺组合（TabPFN）
        ('v5b', ['v5b_virtual_caliper.py'], ['v1b'], 'cpu', 3.3),
        ('v5c', ['v5c_virtual_caliper_fusion.py', '--device', 'auto'], ['v5b', 'v4h', 'v4e'], 'gpu', 2),
        ('v5e', ['v5e_tabpfn_seed_robustness.py'], ['v5b', 'v4h', 'v4e'], 'gpu', 2.3),
        ('v6a_anat', ['v6a_camera_caliper_fusion.py', '--nest', j('v6_anatomy_nested'),
                      '--out', j('v6_camera_caliper_anatomy')], ['v6g', 'v5b', 'v4e'], 'gpu', 1.8),
        ('v6a_resnet', ['v6a_camera_caliper_fusion.py'], ['v4h', 'v5b', 'v4e'], 'gpu', 1.8),
        ('v5f', ['v5f_table3.py'], ['v4h', 'v4b', 'v3b_master', 'v4e', 'v5c'], 'gpu', 2.7),
        # 汇总与比较（CPU）
        ('v6f_compare', ['v6f_anatomy_net.py', '--compare'], ['v6f_prof', 'v6f_base', 'v6f_mil', 'v6f_profmil', 'v4b'],
         'cpu', 1),
        ('v3c', ['v3c_view_compare.py'], ['v3b_master', 'v3b_sub', 'v3b_dual'], 'cpu', 1),
        ('v4c', ['v4c_hmap_compare.py'], ['v3b_master', 'v4b'], 'cpu', 1),
        ('crop_frames', None, ['v3b_master'], 'cpu', 0.1),
        ('a1a2a6', ['analysis_a1_a2_a6.py'], ['crop_frames', 'v1_tab'], 'cpu', 1),
        ('v4g', ['v4g_hmap_reliability.py'], ['v4b'], 'cpu', 1),
        ('v5d', ['v5d_multitask_compare.py'], ['v4b', 'v5a_cal', 'v5a_caldims', 'v5a_lds', 'v5a_nll', 'v5b'], 'cpu', 1),
        ('v5g', ['v5g_line_from_shape.py'], ['v4e', 'v5b'], 'cpu', 1),
        ('an_cal', ['analysis_caliper_backfat.py'], ['v1b'], 'cpu', 0.5),
        ('v6d', ['v6d_table1.py'], ['v1b'], 'cpu', 0.2),
        ('v6h', ['v6h_main_vs_caliper.py'], ['v6f_prof', 'v6f_base', 'v6a_anat'], 'cpu', 1),
        ('v6c_anat', ['v6c_reliability_single_dataset.py', '--pred-csv', j('v6_anatomy', 'prof', 'test_preds.csv'),
                      '--seed-cols'] + [f'pred_inner_s{s}' for s in SEEDS5] +
         ['--mean-col', 'pred_inner', '--frame-csv', j('v6_anatomy_nested', 'frame_preds.csv'),
          '--out', j('v6_reliability_anatomy'), '--icc'], ['v6f_prof', 'v6g'], 'cpu', 0.5),
        ('v6c_resnet', ['v6c_reliability_single_dataset.py'], ['v4b'], 'cpu', 0.5),
    ]
    return [dict(name=n, cmd=c, deps=d, kind=k, est=e) for n, c, d, k, e in S]


def crop_frames(R):
    """v3b master 种子 42 的逐帧预测 → framepreds_new_s128/frame_preds.csv（analysis_a1_a2_a6.py 的输入格式）。"""
    src = os.path.join(R, 'v3_views', 'master_s128', 'frame_preds.csv')
    dst = os.path.join(R, 'framepreds_new_s128', 'frame_preds.csv')
    with open(src, encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if r['seed'] == '42']
    assert rows, 'v3b master frame_preds 中没有种子 42'
    with open(dst, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'group', 'fold', 'frame_idx', 'y', 'pred'])
        for r in rows:
            w.writerow([r['uid'], r['group'], r['fold'], r['frame_idx'], r['y'], r['pred']])
    return f'{len(rows)} frames, {len({r["uid"] for r in rows})} pigs → {dst}'


def env_info():
    info = dict(time=time.strftime('%F %T'), host=platform.node(), python=sys.version.split()[0],
                platform=platform.platform())
    for mod in ('numpy', 'sklearn', 'torch', 'tabpfn', 'pandas', 'scipy'):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, '__version__', '?')
        except Exception as e:  # noqa: BLE001
            info[mod] = f'not available ({type(e).__name__})'
    try:
        import torch
        info['cuda'] = torch.version.cuda
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
        info['cudnn'] = torch.backends.cudnn.version()
    except Exception as e:  # noqa: BLE001
        info['gpu'] = f'? ({e})'
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs_v6o'))
    ap.add_argument('--gpu-jobs', type=int, default=3)
    ap.add_argument('--cpu-jobs', type=int, default=3)
    ap.add_argument('--threads', type=int, default=4, help='每个子进程的 OMP / MKL 线程数')
    ap.add_argument('--only', nargs='+')
    ap.add_argument('--redo', nargs='+', default=[])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='流程冒烟测试：网络只训 1 epoch、1 个种子（须配合临时 --runs 副本）')
    ap.add_argument('--python-stats', default=sys.executable,
                    help='运行 statsmodels 步骤的解释器（实验室单一环境不用设；笔记本上为 D:\\Anaconda\\python.exe）')
    a = ap.parse_args()
    R = os.path.abspath(a.runs)
    if a.smoke and R == os.path.abspath(os.path.join(HERE, 'runs_v6o')):
        sys.exit('--smoke 会写入无意义的结果，请用 --runs 指向 runs_v6o 的临时副本')
    S = steps(R)
    byname = {s['name']: s for s in S}
    for s in S:
        for d in s['deps']:
            assert d in byname, f'{s["name"]}: 未知依赖 {d}'
    # 优先级 = 关键路径长度（本步 + 最长下游链），让长依赖链先启动
    children = {s['name']: [c['name'] for c in S if s['name'] in c['deps']] for s in S}
    rank = {}

    def crit(n):
        if n not in rank:
            rank[n] = byname[n]['est'] + max([crit(c) for c in children[n]], default=0)
        return rank[n]

    for s in S:
        s['rank'] = crit(s['name'])
    if a.smoke:                                                 # 只检验流程能否走通，数字无意义
        for s in S:
            c = s['cmd']
            if c and c[0] in SMOKE_EPOCHS:
                c += ['--epochs', '1']
            if c and c[0] in SMOKE_SEEDS:                       # 2 个种子：v6c 的种子间 SD 至少要 2 个种子
                c += ['--seeds', '42', '43']                    # argparse：后出现的同名参数覆盖前面的
            if s['name'] == 'v6c_anat':
                c += ['--seed-cols', 'pred_inner_s42', 'pred_inner_s43']
            if s['name'] == 'v6c_resnet':
                c += ['--seed-cols', 'pred_s42', 'pred_s43']
    if a.list:
        for s in S:
            print(f'{s["name"]:16s} {s["kind"]}  ~{s["est"]:>4g} min  deps={",".join(s["deps"]) or "-"}')
        print(f'共 {len(S)} 步；笔记本 4060 串行合计约 {sum(s["est"] for s in S) / 60:.1f} h')
        return
    for p in ('index_v2_v6o.csv', os.path.join('framepreds_new_s128', 'test_preds.csv'),
              os.path.join('v6_anatomy', '_cache', 'profiles_v2.npz'), os.path.join('_seed', 'log.txt')):
        if not os.path.exists(os.path.join(R, p)):
            sys.exit(f'种子目录不完整，缺 {os.path.join(R, p)}：请先在笔记本上运行 v6o_make_subset.py 并把 runs_v6o 拷过来')
    D = os.path.join(R, '_driver')
    mainlog = None
    if not a.dry_run:                                           # dry-run 不在种子目录里留任何文件
        for sub in ('done', 'logs'):
            os.makedirs(os.path.join(D, sub), exist_ok=True)
        mainlog = open(os.path.join(D, 'driver_log.txt'), 'a', encoding='utf-8')

    def log(s):
        line = f'[{time.strftime("%F %T")}] {s}'
        print(line, flush=True)
        if mainlog:
            mainlog.write(line + '\n')
            mainlog.flush()

    def done_mark(n):
        return os.path.join(D, 'done', f'{n}.ok')

    for n in a.redo:
        if os.path.exists(done_mark(n)):
            os.remove(done_mark(n))
    want = set(a.only) if a.only else set(byname)
    done = {s['name'] for s in S if os.path.exists(done_mark(s['name']))}
    pending = [s for s in S if s['name'] in want and s['name'] not in done]
    missing = {d for s in pending for d in s['deps'] if d not in done and d not in want}
    if missing:
        sys.exit(f'--only 的依赖尚未完成：{sorted(missing)}')

    env = dict(os.environ, BACKFAT_RUNS=R, BACKFAT_INDEX=os.path.join(R, 'index_v2_v6o.csv'),
               PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_DISABLE_XET='1',
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    log(f'v6o rerun | runs {R} | gpu-jobs {a.gpu_jobs} cpu-jobs {a.cpu_jobs} threads {a.threads} | '
        f'{len(pending)} steps to run, {len(done)} already done')
    if a.dry_run:
        fin, rest, wave = set(done), list(pending), 0
        while rest:
            ready = sorted([s for s in rest if all(d in fin for d in s['deps'])], key=lambda s: -s['rank'])
            if not ready:
                sys.exit(f'依赖无法满足：{[s["name"] for s in rest]}')
            wave += 1
            print(f'wave {wave}: ' + ', '.join(f'{s["name"]}({s["kind"]})' for s in ready))
            fin |= {s['name'] for s in ready}
            rest = [s for s in rest if s['name'] not in fin]
        return
    info = env_info()
    with open(os.path.join(D, 'env.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    log(f'env: {info}')

    running, failed, timing = {}, set(), {}
    t_all = time.time()
    while pending or running:
        for n, (p, fh, t0) in list(running.items()):          # 回收已结束的进程
            if p.poll() is None:
                continue
            fh.close()
            timing[n] = (time.time() - t0) / 60
            del running[n]
            if p.returncode == 0:
                with open(done_mark(n), 'w', encoding='utf-8') as f:
                    f.write(f'{time.strftime("%F %T")} {timing[n]:.1f} min\n')
                done.add(n)
                log(f'DONE  {n:16s} {timing[n]:6.1f} min')
            else:
                failed.add(n)
                log(f'FAIL  {n:16s} exit {p.returncode} after {timing[n]:.1f} min → 见 _driver/logs/{n}.txt')
        for s in list(pending):                                 # 依赖失败的步骤不再运行
            if any(d in failed for d in s['deps']):
                failed.add(s['name'])
                pending.remove(s)
                log(f'SKIP  {s["name"]:16s} 依赖失败')
        ready = sorted([s for s in pending if all(d in done for d in s['deps'])], key=lambda s: -s['rank'])
        for s in ready:                                         # 启动就绪步骤（耗时长的先跑）
            busy = sum(1 for n in running if byname[n]['kind'] == s['kind'])
            if busy >= (a.gpu_jobs if s['kind'] == 'gpu' else a.cpu_jobs):
                continue
            pending.remove(s)
            n = s['name']
            if s['cmd'] is None:
                t0 = time.time()
                try:
                    msg = crop_frames(R)
                    with open(done_mark(n), 'w', encoding='utf-8') as f:
                        f.write(f'{time.strftime("%F %T")} {msg}\n')
                    done.add(n)
                    timing[n] = (time.time() - t0) / 60
                    log(f'DONE  {n:16s} {msg}')
                except Exception as e:  # noqa: BLE001
                    failed.add(n)
                    log(f'FAIL  {n:16s} {type(e).__name__}: {e}')
                continue
            fh = open(os.path.join(D, 'logs', f'{n}.txt'), 'a', encoding='utf-8')
            fh.write(f'\n===== {time.strftime("%F %T")} {" ".join(s["cmd"])}\n')
            fh.flush()
            py = a.python_stats if n in STATS_STEPS else sys.executable
            p = subprocess.Popen([py, '-u'] + s['cmd'], cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[n] = (p, fh, time.time())
            log(f'START {n:16s} ({s["kind"]}, ~{s["est"]:g} min on 4060) | running: {", ".join(running)}')
        if pending and not running and not any(all(d in done for d in s['deps']) for s in pending):
            log(f'无法继续：{[s["name"] for s in pending]}（依赖未完成）')
            failed |= {s['name'] for s in pending}
            break
        time.sleep(5)

    total = (time.time() - t_all) / 60
    with open(os.path.join(D, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(f'# v6o rerun summary\n\n- runs: `{R}`\n- finished: {time.strftime("%F %T")}，本次用时 {total:.1f} min\n'
                f'- GPU: {info.get("gpu")}；torch {info.get("torch")}；CUDA {info.get("cuda")}；tabpfn {info.get("tabpfn")}\n\n'
                '| 步骤 | 类型 | 状态 | 用时 (min) |\n|---|---|---|---|\n')
        for s in S:
            n = s['name']
            st = 'FAILED' if n in failed else ('done' if n in done else 'not run')
            f.write(f'| {n} | {s["kind"]} | {st} | {timing.get(n, float("nan")):.1f} |\n')
    log(f'finished in {total:.1f} min | done {len(done)}/{len(S)} | failed {sorted(failed) or "none"}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
