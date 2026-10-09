import torch
import numpy as np
from torch.utils.data import Dataset
import os

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


# -------------------- 粘贴到你的 utils.py 文件中 --------------------

class MyDataset(Dataset):
    def __init__(self, root, args):
        """
        root: 数据集的根目录，例如 'F:/.../data/train'
        args: 从 hydra 传入的配置参数
        """
        super(MyDataset, self).__init__()

        # self.root 是数据文件夹的完整、绝对路径, e.g., 'F:/.../data/train'
        self.root = root
        self.uniform = args.use_uniform_sample
        self.npoints = args.num_point
        self.use_normals = args.use_normals

        # --- 路径修正逻辑 ---
        # 1. 从数据文件夹的绝对路径中，获取它的基本名称 ('train' or 'test')
        dir_basename = os.path.basename(root)  # 结果是 'train'

        # 2. 构造索引文件的正确名称，并假定它在项目根目录下
        #    os.getcwd() 在 hydra 运行时会是输出目录，所以我们用 hydra.utils.get_original_cwd()
        #    但为了简单起见，我们直接假定它在当前目录，因为你从根目录运行脚本。
        index_file_name = f"{dir_basename}.txt" # 结果是 'train.txt'

        # 3. 打开正确的索引文件
        try:
            with open(index_file_name, 'r') as f:
                # data 是一个列表，每项是 "文件名 标签\n"
                data_lines = f.readlines()
        except FileNotFoundError:
            print(f"致命错误：找不到索引文件！")
            print(f"程序期望在你的项目根目录下找到名为 '{index_file_name}' 的文件。")
            print(f"请确认该文件存在且位置正确。")
            raise

        # --- 文件列表和标签列表的构建 ---
        self.datapath = []
        for line in data_lines:
            line = line.strip()
            if not line: continue  # 跳过空行

            parts = line.split()
            filename = parts[0]
            label = parts[1]

            # 这是最关键的一步：使用完整的 self.root 路径和文件名拼接
            # 结果将是 'F:/.../data/train/11-6_48.txt'
            full_path = os.path.join(self.root, filename)
            self.datapath.append((full_path, label))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        # 从 self.datapath 获取完整路径和标签
        full_path, label_str = self.datapath[index]

        try:
            point_set = np.loadtxt(full_path, delimiter=' ').astype(np.float32)
        except FileNotFoundError:
            # 这个报错现在会打印出绝对路径，让我们能清楚地看到问题
            print(f"致命错误：在数据加载过程中找不到点云文件！")
            print(f"程序尝试加载的完整路径是: {full_path}")
            print(f"请检查：")
            print(f"  1. 你的索引文件 '{os.path.basename(full_path)}.txt' 中的文件名 '{os.path.basename(full_path)}' 是否正确无误？")
            print(f"  2. 该文件是否真的存在于 '{os.path.dirname(full_path)}' 文件夹中？")
            raise

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            total_points = point_set.shape[0]
            indices = np.linspace(0, total_points - 1, self.npoints, dtype=int)
            point_set = point_set[indices]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        label = np.array(float(label_str)) # 将字符串标签转换为浮点数
        return point_set, torch.from_numpy(label).float()

# -------------------- 替换到这里结束 --------------------

# class MyDatasetUI(Dataset):
#     def __init__(self,root):
#
#         super(MyDatasetUI, self).__init__()
#         self.root = root
#         self.txt_root = self.root + '.txt'
#         self.uniform = True
#         self.npoints = 2048
#         self.use_normals = False
#
#         # f = open(self.txt_root, 'r')
#         # data = f.readlines()
#
#         imgs = []
#         labels = [0]
#         # for line in data:
#         #     line = line.rstrip()
#         #     word = line.split()
#         #     imgs.append(os.path.join(self.root, word[0]))
#         #
#         #     labels.append(word[1])
#         self.img = imgs
#         self.label = labels
#
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, item):
#         # print(f"Item: {item}, Data length: {len(self.img)}")
#         img = self.img[item]
#
#         label = self.label[item]
#
#         point_set = np.loadtxt(img, delimiter=' ').astype(np.float32)
#
#         if self.uniform:  # 若在预处理部分已经采样过，这里注释
#             point_set = farthest_point_sample(point_set, self.npoints)  # 或者只能用这个
#         else:
#             total_points = point_set.shape[0]
#
#             # 计算采样间隔
#             indices = np.linspace(0, total_points - 1, self.npoints, dtype=int)
#
#             # 均匀采样点
#             point_set = point_set[indices]
#
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#
#
#         # point_set = np.loadtxt(img, delimiter=' ').astype(np.float32) # 改
#         # point_set = np.loadtxt(img, delimiter=' ', max_rows=self.npoints).astype(np.float32)
#
#
#         if not self.use_normals:
#             point_set = point_set[:, 0:3]
#
#         label = np.array(label).astype(np.float32)
#         label = torch.from_numpy(label)
#         # print(point_set.shape)
#         return point_set, label           #img dtype torch.float32  label dtype torch.int32


class MyDatasetUI(Dataset):
    def __init__(self, point_cloud):
        super(MyDatasetUI, self).__init__()
        self.point_cloud_file = point_cloud  # 直接接收单个点云文件
        self.uniform = True
        self.npoints = 2048
        self.use_normals = False

        # 直接指定一个文件路径
        self.img = [self.point_cloud_file]
        self.label = [0]  # 假设只有一个标签

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        # 读取点云文件数据
        point_set = np.loadtxt(img, delimiter=' ').astype(np.float32)

        # 进行采样处理（根据self.uniform的值来决定）
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)  # 使用最远点采样
        else:
            total_points = point_set.shape[0]
            indices = np.linspace(0, total_points - 1, self.npoints, dtype=int)
            point_set = point_set[indices]  # 均匀采样

        # 归一化处理
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # 如果不使用法向量，保留前三个维度
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        # 将标签转换为float32类型，并转为torch tensor
        label = np.array(label).astype(np.float32)
        label = torch.from_numpy(label)

        return point_set, label


if __name__ == '__main__':
    import torch
    import argparse


    def parse_args():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('training')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
        parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
        parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
        parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
        parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
        parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
        parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
        parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
        parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
        parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
        parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
        parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
        return parser.parse_args()

    train_path = r'E:\71data\1\train'
    train_dataset = MyDataset(train_path, args=parse_args())
    DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)  # torch.Size([8, 1024, 3])
        print(label.shape)  # torch.Size([8])
        print(point)
        print(label)