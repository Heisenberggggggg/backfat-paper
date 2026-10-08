"""
CUDA / PyTorch 缓存清理 + 重定向
=====================================
1. 清理 C 盘 CUDA / PyTorch 训练产生的缓存
2. 把以后的缓存重定向到 F 盘（不再占 C 盘）

可重复运行，安全。
"""

import os
import shutil
from pathlib import Path


def fmt(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_dir_size(entry.path)
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        return 0
    return total


def safe_clean(path, label):
    """清理目录里所有内容，但保留目录本身"""
    if not os.path.exists(path):
        print(f"  [{label}] 不存在，跳过: {path}")
        return 0

    before = get_dir_size(path)
    if before == 0:
        print(f"  [{label}] 已为空: {path}")
        return 0

    print(f"  [{label}] 清理中: {path} ({fmt(before)})")
    cleaned = 0
    failed = 0
    for entry in os.scandir(path):
        try:
            if entry.is_file(follow_symlinks=False):
                size = entry.stat(follow_symlinks=False).st_size
                os.remove(entry.path)
                cleaned += size
            elif entry.is_dir(follow_symlinks=False):
                size = get_dir_size(entry.path)
                shutil.rmtree(entry.path)
                cleaned += size
        except (PermissionError, OSError) as e:
            failed += 1

    print(f"    -> 释放 {fmt(cleaned)}" + (f"，{failed}个文件无法删除" if failed else ""))
    return cleaned


# ==================== 第一步：清理 ====================
print("=" * 70)
print("第一步：清理 CUDA / PyTorch 缓存")
print("=" * 70)

caches_to_clean = [
    (r'C:\Users\Gengls\AppData\Local\NVIDIA\ComputeCache', 'NVIDIA Compute缓存'),
    (r'C:\Users\Gengls\AppData\Local\NVIDIA Corporation\NV_Cache', 'NVIDIA NV_Cache'),
    (r'C:\ProgramData\NVIDIA Corporation\NV_Cache', 'NVIDIA系统缓存'),
    (r'C:\Users\Gengls\.cache\torch', 'PyTorch缓存(.cache)'),
    (r'C:\Users\Gengls\AppData\Local\torch_extensions', 'PyTorch扩展缓存'),
    (r'C:\Users\Gengls\AppData\Local\Temp\torch_extensions', 'PyTorch临时扩展'),
    (r'C:\Users\Gengls\.nv', 'NVIDIA用户缓存(.nv)'),
    (r'C:\Users\Gengls\.cache\huggingface', 'HuggingFace缓存'),
    (r'C:\Users\Gengls\.cache\pip', 'pip缓存'),
]

total_cleaned = 0
for path, label in caches_to_clean:
    total_cleaned += safe_clean(path, label)

print()
print(f"小计释放: {fmt(total_cleaned)}")

# ==================== 第二步：重定向缓存到 F 盘 ====================
print()
print("=" * 70)
print("第二步：把未来的缓存重定向到 F 盘")
print("=" * 70)

new_cache_root = r'F:\cuda_pytorch_cache'
os.makedirs(new_cache_root, exist_ok=True)
os.makedirs(os.path.join(new_cache_root, 'cuda_compute'), exist_ok=True)
os.makedirs(os.path.join(new_cache_root, 'torch_extensions'), exist_ok=True)
os.makedirs(os.path.join(new_cache_root, 'torch_hub'), exist_ok=True)

# 设置用户级环境变量（永久生效，对所有未来的 cmd / Python 都有效）
import subprocess

env_vars = {
    'CUDA_CACHE_PATH': os.path.join(new_cache_root, 'cuda_compute'),
    'TORCH_EXTENSIONS_DIR': os.path.join(new_cache_root, 'torch_extensions'),
    'TORCH_HOME': os.path.join(new_cache_root, 'torch_hub'),
}

print(f"\n设置环境变量（用户级，重启cmd后生效）：")
for var, val in env_vars.items():
    print(f"  {var} = {val}")
    try:
        subprocess.run(['setx', var, val], capture_output=True, check=True, text=True)
        print(f"    ✓ 已设置")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ 设置失败: {e}")

print()
print("=" * 70)
print("完成！")
print("=" * 70)
print()
print(f"释放C盘: {fmt(total_cleaned)}")
print(f"未来缓存目录: {new_cache_root}")
print()
print("⚠️  重要：重启所有cmd窗口后环境变量才生效！")
print("    然后再双击 run_breed_training.bat 重新训练，")
print("    新缓存就会写到 F 盘而不是 C 盘。")
