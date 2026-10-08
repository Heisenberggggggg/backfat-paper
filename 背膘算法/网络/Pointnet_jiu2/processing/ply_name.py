import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = r'E:\first_pcl\orgin\4-61\fragments'
target_folder = r'E:\first_pcl\erbiao\4-61'

# 确保目标文件夹存在，如果不存在则创建
os.makedirs(target_folder, exist_ok=True)

# 获取源文件夹中的所有.ply文件列表
ply_files = [f for f in os.listdir(source_folder) if f.endswith('.ply')]

# 遍历文件列表，并复制到目标文件夹并重命名
for idx, ply_file in enumerate(ply_files):
    source_file_path = os.path.join(source_folder, ply_file)
    # 构造目标文件名，形如 '4-3_1.ply', '4-3_2.ply', ...
    target_file_name = f'4-61_{idx+1}.ply'
    target_file_path = os.path.join(target_folder, target_file_name)
    # 复制文件并重命名
    shutil.copy(source_file_path, target_file_path)

print("复制并重命名完成。")
