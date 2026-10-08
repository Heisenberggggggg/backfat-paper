"""
C盘空间诊断脚本
==================
扫描常见占用空间的目录，找出占用最大的文件夹和文件。
"""

import os
from pathlib import Path


def get_dir_size(path):
    """递归计算目录大小（字节）"""
    total = 0
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


def fmt(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def scan_directory(path, min_mb=100):
    """扫描目录，列出所有>=min_mb的子目录"""
    if not os.path.exists(path):
        return []
    results = []
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_dir(follow_symlinks=False):
                    size = get_dir_size(entry.path)
                    if size >= min_mb * 1024 * 1024:
                        results.append((size, entry.path))
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass
    return sorted(results, reverse=True)


if __name__ == '__main__':
    print("=" * 70)
    print("C盘空间诊断 - 扫描常见占用目录")
    print("=" * 70)

    user = os.environ.get('USERPROFILE', r'C:\Users\Gengls')

    # 关键目录列表（按嫌疑程度排序）
    key_dirs = [
        (r'C:\Users\Gengls\AppData\Local\Temp', 'Temp临时文件'),
        (r'C:\Users\Gengls\AppData\Local', 'AppData\\Local'),
        (r'C:\Users\Gengls\AppData\Roaming', 'AppData\\Roaming'),
        (r'C:\Users\Gengls\.cache', '.cache缓存'),
        (r'C:\Users\Gengls\.claude', 'Claude Code配置'),
        (r'C:\ProgramData', 'ProgramData'),
        (r'C:\Windows\SoftwareDistribution', 'Windows更新缓存'),
        (r'C:\Windows\Temp', 'Windows Temp'),
        (r'C:\Windows\Installer', 'Windows Installer'),
        (r'C:\Users\Gengls\Desktop', '桌面'),
        (r'C:\Users\Gengls\Downloads', '下载'),
        (r'C:\Users\Gengls\Documents', '文档'),
        (r'C:\Users\Gengls\Videos', '视频'),
    ]

    print("\n【第一轮】关键目录总体积：")
    print("-" * 70)
    for dir_path, desc in key_dirs:
        if os.path.exists(dir_path):
            size = get_dir_size(dir_path)
            print(f"  {fmt(size):>12}  {desc:<20}  {dir_path}")
        else:
            print(f"  {'N/A':>12}  {desc:<20}  {dir_path} (不存在)")

    print("\n" + "=" * 70)
    print("【第二轮】细分：各目录里 >= 100MB 的大户")
    print("=" * 70)

    big_dirs = [
        r'C:\Users\Gengls\AppData\Local',
        r'C:\Users\Gengls\AppData\Local\Temp',
        r'C:\Users\Gengls\AppData\Roaming',
        r'C:\Users\Gengls\.cache',
        r'C:\ProgramData',
    ]

    for parent in big_dirs:
        if not os.path.exists(parent):
            continue
        print(f"\n>>> {parent}")
        results = scan_directory(parent, min_mb=100)
        if not results:
            print("    (所有子目录均 < 100MB)")
            continue
        for size, path in results[:15]:
            print(f"    {fmt(size):>12}  {path}")

    print("\n" + "=" * 70)
    print("【诊断完成】")
    print("=" * 70)
    print("可清理建议：")
    print("  - AppData\\Local\\Temp: 可以清空")
    print("  - Windows\\SoftwareDistribution\\Download: Windows更新后可删")
    print("  - .cache/pip, .cache/torch, .cache/huggingface: 可清理")
    print("  - Claude task output: C:\\Users\\Gengls\\AppData\\Local\\Temp\\claude\\")
