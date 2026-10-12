'''
版本二：使用mixed_pooling，fc2更改,再加入CBAM模块
'''
import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from models.Hengshuang.transformer import TransformerBlock

import torch
from torch import nn

#-----------------------------------CBAM---------------------------------------------------------------------------#
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图
#----------------------------------------CBAM------------------------------------------------------------------------#



class MixedPooling(nn.Module):
    def __init__(self, alpha=0.5):
        super(MixedPooling, self).__init__()
        self.alpha = alpha
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x shape: (batch_size, num_points, feature_dim)
        x = x.transpose(1, 2)  # Change shape to (batch_size, feature_dim, num_points)
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        mixed_pooled = self.alpha * avg_pooled + (1 - self.alpha) * max_pooled
        return mixed_pooled.view(x.size(0), -1)


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()



        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        # print(f"Backbone input x: {x.shape}, dtype: {x.dtype}")
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim

        self.mixed_pooling = MixedPooling(alpha=0.5)
        self.cbam = CBAM(64)
        # self.fc2 = nn.Sequential(
        #     nn.Linear(32 * 2 ** nblocks, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_c)
        # )
        self.fc2 = nn.Sequential(  # 1
            nn.Linear(32 * 2 ** nblocks, 64),  # 1
            nn.ReLU(),  # 1
            nn.Linear(64, 32),  # 1
            nn.ReLU(),  # 1
            nn.Linear(32, n_c)  # 1
        )
        self.nblocks = nblocks

    def forward(self, x):
        # print(f"x形状: {x.shape}")       #torch.Size([16, 1024, 3])
        points, _ = self.backbone(x)  # 关心第一个值，特征向量

        points_1 = points.unsqueeze(2)
        cbam_features = self.cbam(points_1)
        points_2 = cbam_features.squeeze(2)
        # print(f"points形状: {points.shape}")    #[16, 4, 512])
        # res = self.fc2(points.mean(1))
        pooled_features = self.mixed_pooling(points_2)
        res = self.fc2(pooled_features)
        # print(f"res形状: {res.shape}")        #[16, 10]
        return res


class PointTransformerSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, cfg.model.transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = 32 * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))

        self.fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )

    def forward(self, x):
        points, xyz_and_feats = self.backbone(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]

        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]

        return self.fc3(points)


# --------------------------------------------------------------------------------------------------#
# class Config:
#     class Model:
#         def __init__(self):
#             self.nblocks = 2
#             self.nneighbor = 16
#             self.transformer_dim = 64
#
#     def __init__(self):
#         self.num_point = 1024
#         self.num_class = 1
#         self.input_dim = 3
#         self.model = self.Model()
#
#
# cfg = Config()
# cls_model = PointTransformerCls(cfg)
#
# # 创建一个随机输入张量，形状为 (batch_size, num_points, input_dim)
# batch_size = 16
# num_points = cfg.num_point
# input_dim = cfg.input_dim
# inputs = torch.randn(batch_size, num_points, input_dim)
#
# # 通过分类模型前向传播
# cls_output = cls_model(inputs)
#
# # 打印分类模型的输入和输出形状
# print(f"模型输入形状: {inputs.shape}")
# print(f"模型输出形状: {cls_output.shape}")

# --------------------------------------------------------------------------------------------------#

# 创建 Backbone 实例
# model = Backbone(cfg)
#
# # 创建一个随机输入张量，形状为 (batch_size, num_points, input_dim)
# batch_size = 16
# num_points = cfg.num_point
# input_dim = cfg.input_dim
# inputs = torch.randn(batch_size, num_points, input_dim)
#
# # 通过模型前向传播
# points, xyz_and_feats = model(inputs)
#
# # 打印输入和输出的形状
# print(f"backbone输入形状: {inputs.shape}")
# print(f"backbone输出 points 形状: {points.shape}")
#
# print(f"输出 xyz_and_feats 长度: {len(xyz_and_feats)}")
# for i, (xyz, feats) in enumerate(xyz_and_feats):
#     print(f"阶段 {i} - xyz 形状: {xyz.shape}, feats 形状: {feats.shape}")

'''
backbone输入形状: torch.Size([16, 1024, 3])
backbone输出 points 形状: torch.Size([16, 4, 512])
输出 xyz_and_feats 长度: 5
阶段 0 - xyz 形状: torch.Size([16, 1024, 3]), feats 形状: torch.Size([16, 1024, 32])
阶段 1 - xyz 形状: torch.Size([16, 256, 3]), feats 形状: torch.Size([16, 256, 64])
阶段 2 - xyz 形状: torch.Size([16, 64, 3]), feats 形状: torch.Size([16, 64, 128])
阶段 3 - xyz 形状: torch.Size([16, 16, 3]), feats 形状: torch.Size([16, 16, 256])
阶段 4 - xyz 形状: torch.Size([16, 4, 3]), feats 形状: torch.Size([16, 4, 512])
'''

# -----------------------------------------------------------------------------------------------------------#