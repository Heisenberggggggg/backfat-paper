# -*- coding: utf-8 -*-
"""
双视角流水线自动运行脚本
========================
按顺序运行4个步骤，每步完成后自动进入下一步：
1. generate_pointcloud_v3_rear.py  — 臀部掩膜点云生成（如已完成则跳过）
2. postprocess_pointcloud_v3_rear.py — 臀部点云后处理
3. extract_features_v3_rear.py — 臀部特征提取
4. train_traditional_v3_dualview.py — 双视角融合训练

Author: 耿来硕 + Claude
Date: 2026-03-31
"""
import subprocess
import sys
import time
import os

PYTHON = r"D:\Anaconda\envs\pytorch\python.exe"
BASE_DIR = r"F:\Postgraduate\Lab\饲喂站\背膘算法"

steps = [
    ("Step 1/4: 臀部掩膜点云生成", os.path.join(BASE_DIR, "yolo+FastSAM", "generate_pointcloud_v3_rear.py")),
    ("Step 2/4: 臀部点云后处理",   os.path.join(BASE_DIR, "yolo+FastSAM", "postprocess_pointcloud_v3_rear.py")),
    ("Step 3/4: 臀部特征提取",     os.path.join(BASE_DIR, "网络", "pointmlp_backfat", "extract_features_v3_rear.py")),
    ("Step 4/4: 双视角融合训练",   os.path.join(BASE_DIR, "网络", "pointmlp_backfat", "train_traditional_v3_dualview.py")),
]

log_path = os.path.join(BASE_DIR, "yolo+FastSAM", "dualview_pipeline_log.txt")

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

if __name__ == '__main__':
    log("=" * 60)
    log("双视角流水线启动")
    log("=" * 60)

    pipeline_start = time.time()

    for step_name, script_path in steps:
        log(f"\n{'='*60}")
        log(f"开始 {step_name}")
        log(f"脚本: {script_path}")
        log(f"{'='*60}")

        step_start = time.time()
        ret = subprocess.run(
            [PYTHON, script_path],
            cwd=os.path.dirname(script_path),
        )
        elapsed = time.time() - step_start

        if ret.returncode != 0:
            log(f"[错误] {step_name} 失败! 返回码={ret.returncode}, 耗时={elapsed:.0f}秒")
            log("流水线中止")
            sys.exit(1)
        else:
            log(f"[完成] {step_name} 成功, 耗时={elapsed:.0f}秒")

    total = time.time() - pipeline_start
    log(f"\n{'='*60}")
    log(f"全部流水线完成! 总耗时={total:.0f}秒 ({total/3600:.1f}小时)")
    log(f"{'='*60}")
