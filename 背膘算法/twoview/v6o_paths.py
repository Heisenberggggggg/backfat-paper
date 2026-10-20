# -*- coding: utf-8 -*-
"""
v6o_paths.py — 分析脚本的 runs 根目录与标签索引路径（2026-10-20 起统一从这里取）

背景：2026-10-20 导师决定剔除背膘 > 25 mm 的记录，并按吕年珅的名单剔除流产母猪（DECISIONS 2026-10-20（二））。
要在剔除后的子集上重跑全部分析，又不能覆盖 runs/ 里的旧结果，所以各脚本的输入输出根目录改为可切换：
  - 不设环境变量：与原来完全相同（twoview/runs/、twoview/index_v2.csv）；
  - 设 BACKFAT_RUNS=<目录>：所有 runs/... 的读写改到该目录（如 twoview/runs_v6o/）；
  - 设 BACKFAT_INDEX=<文件>：标签索引改用该文件（如 runs_v6o/index_v2_v6o.csv，含两处标签修正）。
由 v6o_rerun_subset.py 统一设置；单独运行某个脚本时也可手动设置。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.abspath(os.environ.get('BACKFAT_RUNS') or os.path.join(HERE, 'runs'))
INDEX_PATH = os.path.abspath(os.environ.get('BACKFAT_INDEX') or os.path.join(HERE, 'index_v2.csv'))
