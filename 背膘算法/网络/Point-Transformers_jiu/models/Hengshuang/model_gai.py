'''
加入PnP
'''
import torch
import torch.nn as nn
from pointnet_util import  PointNetSetAbstraction
from models.Hengshuang.transformer_gai import TransformerBlock
from module_jiu.PnP import *  # 确保导入 PnP3D 模块
import random
class MixedPooling(nn.Module):
    def __init__(self, alpha):
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



# class MixedPooling(nn.Module):
#     def __init__(self):
#         super(MixedPooling, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#
#     def forward(self, x):
#         # x shape: (batch_size, num_points, feature_dim)
#         x = x.transpose(1, 2)  # Change shape to (batch_size, feature_dim, num_points)
#         self.alpha = random.choice([0, 1])  # Randomly choose between 0 and 1
#         avg_pooled = self.avg_pool(x)
#         max_pooled = self.max_pool(x)
#         mixed_pooled = self.alpha * avg_pooled + (1 - self.alpha) * max_pooled
#         return mixed_pooled.view(x.size(0), -1)

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim

        # 第一层全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),  # 输入维度为 d_points，输出维度为 32
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(32, 32)  # 线性变换到 32 维
        )

        # 第一个 Transformer Block
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.pnp3d1 = PnP3D(32)  # 添加 PnP3D 模块

        # 第一个 Transition Down 和 Transformer Block
        channel_0 = 32 * 2 ** (0 + 1)  # channel = 64
        self.transition_down_0 = TransitionDown(npoints // 4, nneighbor, [channel_0 // 2 + 3, channel_0, channel_0])
        self.transformer_0 = TransformerBlock(channel_0, cfg.model.transformer_dim, nneighbor)
        self.pnp3d_0 = PnP3D(channel_0)  # 添加 PnP3D 模块

        # 第二个 Transition Down 和 Transformer Block
        channel_1 = 32 * 2 ** (1 + 1)  # channel = 128
        self.transition_down_1 = TransitionDown(npoints // 16, nneighbor, [channel_1 // 2 + 3, channel_1, channel_1])
        self.transformer_1 = TransformerBlock(channel_1, cfg.model.transformer_dim, nneighbor)
        self.pnp3d_1 = PnP3D(channel_1)  # 添加 PnP3D 模块

        self.nblocks = nblocks

    def forward(self, x):
        # 输入 x 的形状为 [B, N, d_points]
        xyz = x[..., :3]  # 提取点的坐标
        points = self.transformer1(xyz, self.fc1(x))[0]

        points = self.pnp3d1(xyz, points, k=20)  # 调用 PnP3D 模块


        xyz_and_feats = [(xyz, points)]

        xyz, points = self.transition_down_0(xyz, points)
        points = self.transformer_0(xyz, points)[0]

        points = self.pnp3d_0(xyz, points, k=20)  # 调用 PnP3D 模块
        xyz_and_feats.append((xyz, points))

        # 第二个 Transition Down 和 Transformer Block
        xyz, points = self.transition_down_1(xyz, points)
        points = self.transformer_1(xyz, points)[0]

        points = self.pnp3d_1(xyz, points, k=20)  # 调用 PnP3D 模块

        xyz_and_feats.append((xyz, points))

        return points, xyz_and_feats

class PointTransformer_jiu(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.mixed_pooling = MixedPooling(alpha=0.5)
        # self.mixed_pooling = MixedPooling()
        self.fc2 = nn.Sequential(  # 1

            nn.Linear(32 * 2 ** nblocks, 256),  # 1
            nn.ReLU(),  # 1
            nn.Linear(256, 128),  # 1
            nn.ReLU(),  # 1
            nn.Linear(128, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):

        points, _ = self.backbone(x)  # 关心第一个值，特征向量
        pooled_features = self.mixed_pooling(points)
        res = self.fc2(pooled_features)
        return res

class BackboneUI(nn.Module):
    def __init__(self):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = 2048, 2, 16, 1,3

        # 第一层全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),  # 输入维度为 d_points，输出维度为 32
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(32, 32)  # 线性变换到 32 维
        )

        # 第一个 Transformer Block
        self.transformer1 = TransformerBlock(32, 512, nneighbor)
        self.pnp3d1 = PnP3D(32)  # 添加 PnP3D 模块

        # 第一个 Transition Down 和 Transformer Block
        channel_0 = 32 * 2 ** (0 + 1)  # channel = 64
        self.transition_down_0 = TransitionDown(npoints // 4, nneighbor, [channel_0 // 2 + 3, channel_0, channel_0])
        self.transformer_0 = TransformerBlock(channel_0, 512, nneighbor)
        self.pnp3d_0 = PnP3D(channel_0)  # 添加 PnP3D 模块

        # 第二个 Transition Down 和 Transformer Block
        channel_1 = 32 * 2 ** (1 + 1)  # channel = 128
        self.transition_down_1 = TransitionDown(npoints // 16, nneighbor, [channel_1 // 2 + 3, channel_1, channel_1])
        self.transformer_1 = TransformerBlock(channel_1, 512, nneighbor)
        self.pnp3d_1 = PnP3D(channel_1)  # 添加 PnP3D 模块

        self.nblocks = nblocks

    def forward(self, x):
        # 输入 x 的形状为 [B, N, d_points]
        xyz = x[..., :3]  # 提取点的坐标
        points = self.transformer1(xyz, self.fc1(x))[0]

        points = self.pnp3d1(xyz, points, k=20)  # 调用 PnP3D 模块

        xyz_and_feats = [(xyz, points)]

        # 第一个 Transition Down 和 Transformer Block
        xyz, points = self.transition_down_0(xyz, points)
        points = self.transformer_0(xyz, points)[0]

        points = self.pnp3d_0(xyz, points, k=20)  # 调用 PnP3D 模块

        xyz_and_feats.append((xyz, points))

        # 第二个 Transition Down 和 Transformer Block
        xyz, points = self.transition_down_1(xyz, points)
        points = self.transformer_1(xyz, points)[0]
        points = self.pnp3d_1(xyz, points, k=20)  # 调用 PnP3D 模块
        xyz_and_feats.append((xyz, points))

        return points, xyz_and_feats
class PointTransformer_UI(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneUI()
        npoints, nblocks, nneighbor, n_c, d_points = 2048, 2, 16, 1, 3
        self.mixed_pooling = MixedPooling(alpha=0.5)
        # self.mixed_pooling = MixedPooling()
        self.fc2 = nn.Sequential(  # 1

            nn.Linear(32 * 2 ** nblocks, 256),  # 1
            nn.ReLU(),  # 1
            nn.Linear(256, 128),  # 1
            nn.ReLU(),  # 1
            nn.Linear(128, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):
        # print(f"x形状: {x.shape}")       #torch.Size([16, 1024, 3])
        points, _ = self.backbone(x)  # 关心第一个值，特征向量
        # print(f"points形状: {points.shape}")    #[16, 4, 512])
        # res = self.fc2(points.mean(1))
        pooled_features = self.mixed_pooling(points)
        res = self.fc2(pooled_features)
        # print(f"res形状: {res.shape}")        #[16, 10]
        # print(res)
        return res


# --------------------------------------------------------------------------------------------------#
# class Config:
#     class Model:
#         def __init__(self):
#             self.nblocks = 2
#             self.nneighbor = 16
#             self.transformer_dim = 512
#
#     def __init__(self):
#         self.num_point = 1024
#         self.num_class = 1
#         self.input_dim = 3
#         self.model = self.Model()
# #
# #
# # #
# cfg = Config()
# cls_model = PointTransformer_jiu(cfg)
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