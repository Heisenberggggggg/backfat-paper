'''
版本一：使用mixed_pooling，fc2更改
'''
import torch
import torch.nn as nn
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction
from models.Hengshuang.transformer import TransformerBlock
# from kan_1 import *
from models.Hengshuang.kan_1 import *
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
        # self.fc1 = nn.Sequential(
        #     nn.Linear(d_points, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32)
        # )
        self.kan = KAN([d_points, 32, 32, 32])
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        # print(f"Backbone input x: {x.shape}, dtype: {x.dtype}")
        xyz = x[..., :3]
        # points = self.transformer1(xyz, self.fc1(x))[0]
        points = self.transformer1(xyz, self.kan(x))[0]

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
        # self.fc2 = nn.Sequential(
        #     nn.Linear(32 * 2 ** nblocks, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, n_c)
        # )
        # self.fc2 = nn.Sequential(                #1
        #     nn.Linear(32 * 2 ** nblocks, 64),   #1
        #     nn.ReLU(),                 #1
        #     nn.Linear(64, 32),          #1
        #     nn.ReLU(),                #1
        #     nn.Linear(32, n_c)            #1
        # )
        self.nblocks = nblocks
        self.kan2 =  KAN([32 * 2 ** nblocks, 128, 64, n_c])
    def forward(self, x):
        # print(f"x形状: {x.shape}")       #torch.Size([16, 1024, 3])
        points, _ = self.backbone(x)    #关心第一个值，特征向量
        # print(f"points形状: {points.shape}")    #[16, 4, 512])
        # res = self.fc2(points.mean(1))
        pooled_features = self.mixed_pooling(points)
        # print(pooled_features.shape)
        # res = self.fc2(pooled_features)
        res = self.kan2(pooled_features)

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



#--------------------------------------------------------------------------------------------------#
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
# cfg = Config()
# cls_model = PointTransformerCls(cfg).cuda()
#
# total_params = sum(p.numel() for p in cls_model.parameters())
# total_params_mb = total_params * 4 / (1024 ** 2)
# print(f"模型参数量: {total_params_mb:.2f} MB")
#
# # 创建一个随机输入张量，形状为 (batch_size, num_points, input_dim)
# batch_size = 16
# num_points = cfg.num_point
# input_dim = cfg.input_dim
# inputs = torch.randn(batch_size, num_points, input_dim).cuda()
#
# # 通过分类模型前向传播
# cls_output = cls_model(inputs)
#
# # 打印分类模型的输入和输出形状
# print(f"模型输入形状: {inputs.shape}")
# print(f"模型输出形状: {cls_output.shape}")

#--------------------------------------------------------------------------------------------------#

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

#-----------------------------------------------------------------------------------------------------------#