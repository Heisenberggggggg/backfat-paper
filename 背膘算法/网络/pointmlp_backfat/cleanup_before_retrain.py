"""
重新训练前清理脚本
====================
删除上次只跑了 fold_1 的不完整 breed_* 目录，
保留脚本/日志以便对比。
"""

import os
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(script_dir, 'checkpoints_pointmlp_breed')

# Step 1: 备份旧日志（不覆盖，保留历史参考）
print("Step 1: 备份旧训练日志")
print("-" * 60)
for breed in ['A', 'LY', 'T', 'ALL']:
    log_src = os.path.join(script_dir, f'training_log_v6_{breed}.txt')
    log_bak = os.path.join(script_dir, f'training_log_v6_{breed}_fold1only_backup.txt')
    if os.path.exists(log_src):
        if not os.path.exists(log_bak):
            shutil.copy2(log_src, log_bak)
            print(f"  ✓ 备份 {log_src} -> {log_bak}")
        else:
            print(f"  - 已存在备份 {log_bak}，跳过")
    else:
        print(f"  - 无日志 {log_src}，跳过")

# Step 2: 删除不完整的 breed_* 目录
print()
print("Step 2: 删除不完整的 checkpoint")
print("-" * 60)
if not os.path.exists(base):
    print(f"  目录不存在: {base}")
else:
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and name.startswith('breed_'):
            print(f"  删除 {full}")
            try:
                shutil.rmtree(full)
                print(f"    ✓ 已删除")
            except Exception as e:
                print(f"    ✗ 删除失败: {e}")

print()
print("-" * 60)
print("清理完成。现在可以双击 run_breed_training.bat 重新运行。")
