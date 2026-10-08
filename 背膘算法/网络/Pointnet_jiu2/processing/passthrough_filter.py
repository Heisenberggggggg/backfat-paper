'''
这是处理ply文件的，直通滤波
'''
import os
import numpy as np
from plyfile import PlyData, PlyElement


def passthrough_filter(point_cloud, min_values, max_values):
    x = point_cloud['x']
    y = point_cloud['y']
    z = point_cloud['z']

    mask = (x >= min_values[0]) & (x <= max_values[0]) & \
           (y >= min_values[1]) & (y <= max_values[1]) & \
           (z >= min_values[2]) & (z <= max_values[2])

    filtered_cloud = point_cloud[mask]

    return filtered_cloud


# 定义原始文件夹和处理后文件夹路径
orgin_path = r'E:\first_pcl\zhitong\orgin'
output_path = r'E:\first_pcl\zhitong\chulihou'

# 获取原始文件夹下所有点云文件
file_list = os.listdir(orgin_path)

# 遍历每个点云文件进行直通滤波并保存处理后的文件
for file_name in file_list:
    file_path = os.path.join(orgin_path, file_name)

    # 读取PLY文件
    plydata = PlyData.read(file_path)
    data = plydata.elements[0].data

    # 转换为numpy数组
    point_cloud = np.array([(x, y, z) for x, y, z in zip(data['x'], data['y'], data['z'])],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # 设置直通滤波的最小和最大值
    min_values = [-0.6, -0.5, -0.1]
    max_values = [1.2, 0.4, 1.6]

    filtered_cloud = passthrough_filter(point_cloud, min_values, max_values)

    # 创建新的PlyData对象
    new_plydata = PlyData([PlyElement.describe(filtered_cloud, 'vertex')])

    # 保存处理后的PLY文件到处理后文件夹
    output_file_path = os.path.join(output_path, file_name)
    new_plydata.write(output_file_path)