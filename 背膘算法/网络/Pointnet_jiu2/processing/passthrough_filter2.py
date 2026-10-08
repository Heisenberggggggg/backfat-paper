'''
这是处理txt文件的，直通滤波
'''

import os
import numpy as np



def passthrough_filter(point_cloud, min_values, max_values):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    mask = (x >= min_values[0]) & (x <= max_values[0]) & \
           (y >= min_values[1]) & (y <= max_values[1]) & \
           (z >= min_values[2]) & (z <= max_values[2])

    filtered_cloud = point_cloud[mask]

    return filtered_cloud


# 定义原始文件夹和处理后文件夹路径
orgin_path = r'E:\ear_txt\4-9'
output_path = r'E:\txt_filter\4-9'
# orgin_path = r'E:\ear_txt\1'
# output_path = r'E:\ear_txt\2'
np.set_printoptions(suppress=True)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# 获取原始文件夹下所有点云文件
file_list = os.listdir(orgin_path)

# 遍历每个点云文件进行直通滤波并保存处理后的文件
for file_name in file_list:
    file_path = os.path.join(orgin_path, file_name)

    # 读取点云数据
    point_cloud = np.loadtxt(file_path)

    # 设置直通滤波的最小和最大值
    min_values = [-0.9, -0.16, 1]
    max_values = [1.145, 0.44, 1.8]

    filtered_cloud = passthrough_filter(point_cloud, min_values, max_values)

    # 保存处理后的文件到处理后文件夹
    output_file_path = os.path.join(output_path, file_name)
    np.savetxt(output_file_path, filtered_cloud, fmt='%f')