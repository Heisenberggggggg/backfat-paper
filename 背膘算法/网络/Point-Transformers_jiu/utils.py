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


class MyDataset(Dataset):
    def __init__(self,root,args,):

        super(MyDataset, self).__init__()
        self.root = root
        self.txt_root = self.root + '.txt'
        self.uniform = args.use_uniform_sample
        self.npoints = args.num_point
        self.use_normals = args.use_normals

        f = open(self.txt_root, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            imgs.append(os.path.join(self.root, word[0]))

            labels.append(word[1])
        self.img = imgs
        self.label = labels


    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]

        point_set = np.loadtxt(img, delimiter=' ').astype(np.float32)

        if self.uniform:  # 若在预处理部分已经采样过，这里注释
            point_set = farthest_point_sample(point_set, self.npoints)  # 或者只能用这个
        else:
            total_points = point_set.shape[0]

            # 计算采样间隔
            indices = np.linspace(0, total_points - 1, self.npoints, dtype=int)

            # 均匀采样点
            point_set = point_set[indices]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])


        # point_set = np.loadtxt(img, delimiter=' ').astype(np.float32) # 改
        # point_set = np.loadtxt(img, delimiter=' ', max_rows=self.npoints).astype(np.float32)


        if not self.use_normals:
            point_set = point_set[:, 0:3]


        # label = np.array(label).astype(np.int32)
        label = np.array(label).astype(np.float32)
        label = torch.from_numpy(label)
        # print(point_set.shape)
        return point_set, label           #img dtype torch.float32  label dtype torch.int32

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