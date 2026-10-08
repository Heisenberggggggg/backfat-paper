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


#-------------------------------------这个dataset是之前自己写的----------------------------------------------------#

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
            # print(word[0], word[1]
            #       #, word[2]
            #       )
            # word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
            # imgs.append(os.path.join(self.root, word[2], word[0]))
            imgs.append(os.path.join(self.root, word[0]))

            labels.append(word[1])
        self.img = imgs
        self.label = labels


    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]


        #img = Image.open(img)   #这里应该是加载点云文件
        point_set = np.loadtxt(img, delimiter=' ').astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]


        label = np.array(label).astype(np.int32)
        label = torch.from_numpy(label)


        return point_set, label           #img dtype torch.float32  label dtype torch.int32

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

    train_path = r'E:\first_pcl\dataloader\train'
    train_dataset = MyDataset(train_path, args=parse_args())
    DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)  # torch.Size([8, 1024, 3])
        print(label.shape)  # torch.Size([8])
        print(point)
        print(label)
