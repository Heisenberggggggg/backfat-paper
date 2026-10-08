# -*- coding: utf-8 -*-
"""
边缘部署性能实测脚本
================================================================
用途
    产出论文"边缘部署性能"一节所需的实测表（对标王起繁论文表 4，
    以及 02_前沿方法调研 §2.7 / AgriJetsonBench 的方法学）。

为什么这个脚本明天就能跑
    单帧时延 / FPS / 显存 / 功耗的实测结果，与模型精度好坏**无关**。
    即使当前回归模型 R² 只有 0.211，它的推理耗时、显存占用、功耗依然是
    真实且可报道的。因此 P0-7「边缘部署实测」不必等算法改进。

必报指标（按阶段拆解）
    采集+对齐 / 分割 / 深度图生成 / 回归 / 端到端 的时延，
    外加 FPS、显存峰值、功耗，以及量化前后的输出扰动。

用法
    # PC 基线
    python edge_profile.py --dir <帧目录> --weights <pth> --modes pt-fp32,pt-fp16
    # Jetson（先导出 engine，再测）
    python edge_profile.py --dir <帧目录> --weights <pth> \
        --modes pt-fp32,pt-fp16,trt-fp16 --export --platform "Jetson Orin NX 8GB"

不可用的模式会明确跳过并说明原因，不会静默产生假数字。

Author: 耿来硕 + Claude
Date:   2026-08-18
"""

import argparse
import csv
import json
import os
import platform
import re
import shutil
import statistics as st
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backfat_edge_demo import BackfatPipeline, ReplaySource  # noqa: E402

STAGES = ['grab', 'detect', 'segment', 'depthmap', 'regress', 'total']
STAGE_ZH = {
    'grab': '采集+对齐', 'detect': '检测(YOLO11n)', 'segment': '分割(FastSAM-s)',
    'depthmap': '深度图生成', 'regress': '回归(ResNet-18)', 'total': '端到端',
}


# ==================================================================
# 功耗采样
# ==================================================================
class PowerSampler(threading.Thread):
    """后台采样板级/GPU 功率。

    Jetson 走 tegrastats（板级输入功率，AgriJetsonBench 的推荐口径）；
    PC 走 nvidia-smi（仅 GPU 功率，口径不同，报告时必须注明）。
    两者都拿不到时返回 None —— 宁可空着，也不编数字。
    """

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        if shutil.which('tegrastats'):
            self.backend = 'tegrastats'
        elif shutil.which('nvidia-smi'):
            self.backend = 'nvidia-smi'
        else:
            self.backend = None

    def _read_nvidia_smi(self):
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=power.draw',
                 '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL, timeout=5).decode()
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def _run_tegrastats(self):
        """tegrastats 输出形如 VDD_IN 5321mW/5100mW，取瞬时值。"""
        try:
            p = subprocess.Popen(['tegrastats', '--interval',
                                  str(int(self.interval * 1000))],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        except Exception:
            return
        pat = re.compile(r'(VDD_IN|POM_5V_IN)\s+(\d+)mW')
        try:
            for line in p.stdout:
                if self._stop.is_set():
                    break
                m = pat.search(line.decode('utf-8', 'ignore'))
                if m:
                    self.samples.append(int(m.group(2)) / 1000.0)
        finally:
            try:
                p.terminate()
            except Exception:
                pass

    def run(self):
        if self.backend == 'tegrastats':
            self._run_tegrastats()
        elif self.backend == 'nvidia-smi':
            while not self._stop.is_set():
                v = self._read_nvidia_smi()
                if v is not None:
                    self.samples.append(v)
                time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def summary(self):
        if not self.samples:
            return None
        return {'mean_w': st.mean(self.samples), 'max_w': max(self.samples),
                'n': len(self.samples), 'backend': self.backend}


# ==================================================================
def gpu_mem_peak_mb(torch):
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return None


def bench_one(mode, args, frames):
    """跑一种模式，返回统计所需的原始数据。模式不可用则返回 None。"""
    import cv2
    import torch

    half = mode.endswith('fp16')
    use_trt = mode.startswith('trt')

    yolo_p, sam_p = args.yolo, args.fastsam
    if use_trt:
        eng_y = os.path.splitext(args.yolo)[0] + '.engine'
        eng_s = os.path.splitext(args.fastsam)[0] + '.engine'
        if not (os.path.exists(eng_y) and os.path.exists(eng_s)):
            print('[skip] %s：未找到 engine 文件（%s / %s）。'
                  '先加 --export 生成，或在 Jetson 上运行。'
                  % (mode, os.path.basename(eng_y), os.path.basename(eng_s)))
            return None
        yolo_p, sam_p = eng_y, eng_s

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pipe = BackfatPipeline(yolo_p, sam_p, args.weights, args.device)

    if half and pipe.device.type == 'cuda':
        pipe.model = pipe.model.half()

        def regress_half(dm):
            x = np.clip((dm - 300.0) / (2500.0 - 300.0), 0, 1)
            x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).half().to(pipe.device)
            with torch.no_grad():
                return float(pipe.model(t).squeeze().float().cpu().numpy())
        pipe.regress = regress_half
    elif half:
        print('[skip] %s：CPU 上不做 FP16。' % mode)
        return None

    # ---- warm-up：第一帧含 CUDA/TRT 初始化，必须排除 ----
    for i in range(args.warmup):
        rgb, depth, _, _ = frames[i % len(frames)]
        pipe.run(rgb, depth, 0.0)

    # grab 一律记 0：帧已预加载到内存，循环里并没有真正发生采集。
    # 若沿用 ReplaySource 初次读盘的耗时，total 会凭空多出十几毫秒，
    # 与实测 wall-clock 推出的 FPS 对不上（曾出现 54ms/帧却报 30 FPS）。
    # 真实的相机采集耗时必须用 --source live 单独测，见报告脚注。
    acc = {k: [] for k in STAGES}
    preds = []
    pw = PowerSampler()
    if pw.backend:
        pw.start()
    t_start = time.perf_counter()
    for i in range(args.iters):
        rgb, depth, _, _ = frames[i % len(frames)]
        r = pipe.run(rgb, depth, 0.0)
        preds.append(r['backfat'])
        for k in STAGES:
            acc[k].append(pipe.timing[k])
    wall = time.perf_counter() - t_start
    pw.stop()
    time.sleep(0.3)

    return {'stages': acc, 'preds': preds, 'wall_s': wall,
            'fps': args.iters / wall,
            'mem_mb': gpu_mem_peak_mb(torch), 'power': pw.summary(),
            'device': str(pipe.device)}


def stats(v):
    v = sorted(v)
    return {'mean': st.mean(v), 'std': st.pstdev(v) if len(v) > 1 else 0.0,
            'p50': v[len(v) // 2],
            'p95': v[min(len(v) - 1, int(len(v) * 0.95))]}


def main():
    ap = argparse.ArgumentParser(description='边缘部署性能实测（时延/FPS/显存/功耗）')
    ap.add_argument('--dir', required=True, help='测试帧目录（含 color/ 与 depth/）')
    ap.add_argument('--weights', default='', help='背膘回归权重 .pth')
    ap.add_argument('--yolo', default='yolo11n.pt')
    ap.add_argument('--fastsam', default='FastSAM-s.pt')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--modes', default='pt-fp32',
                    help='逗号分隔：pt-fp32,pt-fp16,trt-fp32,trt-fp16,trt-int8')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--platform', default=None,
                    help='报告里显示的平台名，默认自动探测')
    ap.add_argument('--export', action='store_true',
                    help='先把 YOLO / FastSAM 导出为 TensorRT engine（需在目标机上执行）')
    ap.add_argument('--out', default='bench_out')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    plat = a.platform or ('Jetson' if os.path.exists('/etc/nv_tegra_release')
                          else platform.node())

    if a.export:
        from ultralytics import YOLO, FastSAM
        for p, cls in ((a.yolo, YOLO), (a.fastsam, FastSAM)):
            try:
                print('[export] %s -> engine ...' % p)
                cls(p).export(format='engine', half=True)
            except Exception as e:
                print('[warn] 导出 %s 失败：%s' % (p, e), file=sys.stderr)

    # 预读全部帧：使"采集"阶段只计内存拷贝，不混入磁盘 IO 抖动
    src = ReplaySource(a.dir, loop=False)
    frames = []
    while True:
        g = src.read()
        if g is None:
            break
        frames.append(g)
    if not frames:
        raise SystemExit('测试帧目录为空：%s' % a.dir)
    print('[info] 载入 %d 帧，平台=%s，warmup=%d，iters=%d'
          % (len(frames), plat, a.warmup, a.iters))

    results = {}
    for mode in [m.strip() for m in a.modes.split(',') if m.strip()]:
        print('\n=== %s ===' % mode)
        r = bench_one(mode, a, frames)
        if r is None:
            continue
        results[mode] = r
        print('  端到端 %.2f ms | %.1f FPS | 显存峰值 %s | 功耗 %s'
              % (st.mean(r['stages']['total']), r['fps'],
                 ('%.0f MB' % r['mem_mb']) if r['mem_mb'] else 'n/a',
                 ('%.1f W (%s)' % (r['power']['mean_w'], r['power']['backend']))
                 if r['power'] else 'n/a'))

    if not results:
        raise SystemExit('没有任何模式跑成功，未生成报告。')

    # ---------------- CSV ----------------
    csv_p = os.path.join(a.out, 'bench_result.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['platform', 'mode', 'stage', 'mean_ms', 'std_ms',
                    'p50_ms', 'p95_ms'])
        for mode, r in results.items():
            for s in STAGES:
                d = stats(r['stages'][s])
                w.writerow([plat, mode, s, '%.3f' % d['mean'], '%.3f' % d['std'],
                            '%.3f' % d['p50'], '%.3f' % d['p95']])

    # ---------------- Markdown 表（可直接贴进论文） ----------------
    md_p = os.path.join(a.out, 'bench_table.md')
    modes = list(results)
    with open(md_p, 'w', encoding='utf-8') as f:
        f.write('# 边缘部署性能实测\n\n')
        f.write('- 平台：%s\n- 测试帧数：%d（warmup %d 帧已排除）\n'
                '- 生成时间：%s\n\n'
                % (plat, a.iters, a.warmup, time.strftime('%Y-%m-%d %H:%M:%S')))

        f.write('## 表 1  分阶段单帧时延 (ms)\n\n')
        f.write('| 阶段 | ' + ' | '.join(modes) + ' |\n')
        f.write('|---|' + '---|' * len(modes) + '\n')
        for s in STAGES:
            row = ['**%s**' % STAGE_ZH[s] if s == 'total' else STAGE_ZH[s]]
            for m in modes:
                d = stats(results[m]['stages'][s])
                row.append('%.2f ± %.2f' % (d['mean'], d['std']))
            f.write('| ' + ' | '.join(row) + ' |\n')
        f.write('\n> 注：回放模式下帧已预加载到内存，"采集+对齐"计为 0，\n'
                '> 表中"端到端"= 检测 + 分割 + 深度图 + 回归，与 FPS 口径一致。\n'
                '> **相机实际采集耗时须用 `backfat_edge_demo.py --source live` 单独实测后另行计入。**\n')

        f.write('\n## 表 2  吞吐、显存与功耗\n\n')
        f.write('| 模式 | FPS | 显存峰值 (MB) | 平均功耗 (W) | 峰值功耗 (W) | 功率口径 |\n')
        f.write('|---|---|---|---|---|---|\n')
        for m in modes:
            r = results[m]
            p = r['power']
            f.write('| %s | %.1f | %s | %s | %s | %s |\n' % (
                m, r['fps'],
                '%.0f' % r['mem_mb'] if r['mem_mb'] else 'n/a',
                '%.1f' % p['mean_w'] if p else 'n/a',
                '%.1f' % p['max_w'] if p else 'n/a',
                (p['backend'] + ('（板级输入）' if p['backend'] == 'tegrastats'
                                 else '（仅GPU）')) if p else 'n/a'))

        base = 'pt-fp32' if 'pt-fp32' in results else modes[0]
        f.write('\n## 表 3  量化前后预测一致性（基准：%s）\n\n' % base)
        f.write('| 模式 | 与基准的平均绝对偏差 (mm) | 最大绝对偏差 (mm) |\n|---|---|---|\n')
        b = np.array(results[base]['preds'])
        for m in modes:
            v = np.array(results[m]['preds'])
            n = min(len(b), len(v))
            f.write('| %s | %.4f | %.4f |\n'
                    % (m, float(np.mean(np.abs(v[:n] - b[:n]))) if n else 0.0,
                       float(np.max(np.abs(v[:n] - b[:n]))) if n else 0.0))
        f.write('\n> 注：表 3 反映的是**量化对同一批输入的输出扰动**，'
                '不是模型相对真值的精度。\n'
                '> 论文中"INT8 前后 MAE 变化"须在带标签的测试集上离线评测得到。\n')

        if not any(results[m]['power'] for m in modes):
            f.write('\n> ⚠️ 本次未采集到功耗数据（既无 tegrastats 也无 nvidia-smi），'
                    '功耗列为 n/a，**不得据此填写论文**。\n')

    with open(os.path.join(a.out, 'bench_raw.json'), 'w', encoding='utf-8') as f:
        json.dump({m: {'stages': r['stages'], 'fps': r['fps'],
                       'mem_mb': r['mem_mb'], 'power': r['power'],
                       'device': r['device']}
                   for m, r in results.items()}, f, ensure_ascii=False, indent=2)

    print('\n[done] %s\n       %s' % (csv_p, md_p))


if __name__ == '__main__':
    main()
