"""
PointMLP-Elite 背膘回归训练脚本
================================
基于官方 PointMLP 论文架构（Ma et al., 2022），改造为回归任务。
- 纯 PyTorch 实现 FPS/KNN，无需编译 pointnet2_ops CUDA 扩展
- 自动从 docx/xlsx 读取标签，匹配处理后的点云
- 按猪 ID 分割数据集，防止数据泄露
- 完整的训练、验证、测试流程

Author: 耿来硕 + Claude
Date: 2026-03-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import re
import json
import random
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


# ==============================================================================
# 1. 纯 PyTorch 的点云操作（替代 pointnet2_ops）
# ==============================================================================

def square_distance(src, dst):
    """计算两组点之间的欧氏距离平方。
    Input:  src [B, N, C], dst [B, M, C]
    Output: dist [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS) — 纯 PyTorch 实现。
    Input:  xyz [B, N, 3], npoint: 采样点数
    Output: centroids [B, npoint] (索引)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """根据索引提取点。
    Input:  points [B, N, C], idx [B, S] 或 [B, S, K]
    Output: new_points [B, S, C] 或 [B, S, K, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device) \
        .view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_point(nsample, xyz, new_xyz):
    """KNN 近邻搜索。
    Input:  nsample: 近邻数, xyz [B, N, C], new_xyz [B, S, C]
    Output: group_idx [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


# ==============================================================================
# 2. PointMLP-Elite 模型（官方架构，改造为回归）
# ==============================================================================

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class LocalGrouper(nn.Module):
    """局部分组模块：FPS 下采样 + KNN 分组 + 几何仿射归一化。
    这是 PointMLP 的核心创新：通过可学习的仿射参数 α, β 对局部邻域
    进行归一化，使模型能捕捉细微的局部几何变化。
    """
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor"):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize else None
        if self.normalize not in ["center", "anchor"]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        # FPS 下采样
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)       # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, d]

        # KNN 分组
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)         # [B, S, K, 3]
        grouped_points = index_points(points, idx)   # [B, S, K, d]

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        # 几何仿射归一化
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            elif self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1),
                            dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        # 拼接锚点特征
        new_points = torch.cat([
            grouped_points,
            new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)
        ], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation)
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    """带残差连接的 1D 卷积块。"""
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0,
                 bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size,
                      groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size,
                          groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(channel, channel, kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(int(channel * res_expansion), channel, kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    """预提取模块：对局部邻域特征做 Conv1d + 残差块 + 池化。"""
    def __init__(self, channels, out_channels, blocks=1, groups=1,
                 res_expansion=1, bias=True, activation='relu', use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(
                out_channels, groups=groups, res_expansion=res_expansion,
                bias=bias, activation=activation
            ))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    """后提取模块：对全局特征做位置级残差块。"""
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1,
                 bias=True, activation='relu'):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(
                channels, groups=groups, res_expansion=res_expansion,
                bias=bias, activation=activation
            ))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class PointMLPRegressor(nn.Module):
    """PointMLP-Elite 回归模型。

    与官方分类模型的区别：
    1. 最后的 classifier 改为 regressor（输出维度 1，无 Softmax）
    2. 回归头使用更平滑的过渡（256→128→64→1）
    3. Dropout 稍低（0.3），防止小数据集下欠拟合
    """
    def __init__(self, points=1024, embed_dim=32, groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1],
                 pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2]):
        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == \
               len(pos_blocks) == len(dim_expansion)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points

        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce

            local_grouper = LocalGrouper(
                last_channel, anchor_points, kneighbor, use_xyz, normalize
            )
            self.local_grouper_list.append(local_grouper)

            pre_block_module = PreExtraction(
                last_channel, out_channel, pre_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation,
                use_xyz=use_xyz
            )
            self.pre_blocks_list.append(pre_block_module)

            pos_block_module = PosExtraction(
                out_channel, pos_block_num, groups=groups,
                res_expansion=res_expansion, bias=bias, activation=activation
            )
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)

        # 回归头（与官方分类头的关键区别）
        self.regressor = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.BatchNorm1d(128),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.act,
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 输出 1 个连续值（背膘厚度 mm）
        )

    def forward(self, x):
        """
        Input: x [B, 3, N] — channels-first 格式（与官方一致）
        Output: [B, 1] — 预测的背膘厚度
        """
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        x = self.embedding(x)      # [B, embed_dim, N]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # [B, last_channel]
        x = self.regressor(x)  # [B, 1]
        return x


# ==============================================================================
# 3. 数据集
# ==============================================================================

class BackfatPointCloudDataset(Dataset):
    """背膘点云数据集。

    直接从处理后的点云 .txt 文件读取，支持数据增强。
    每个样本 = 一帧点云 + 对应的背膘真值（mm）。
    """
    def __init__(self, samples, num_points=1024, augment=False):
        """
        Args:
            samples: list of (file_path, backfat_value)
            num_points: 统一采样到的点数
            augment: 是否进行数据增强
        """
        self.samples = samples
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, pc):
        """数据增强：旋转、抖动、缩放、随机丢点。"""
        # Z 轴随机旋转
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0],
                       [sin_t, cos_t, 0],
                       [0, 0, 1]], dtype=np.float32)
        pc = pc @ R.T

        # 随机抖动
        pc += np.random.normal(0, 0.005, size=pc.shape).astype(np.float32)

        # 随机各向同性缩放
        scale = np.random.uniform(0.95, 1.05)
        pc *= scale

        # 随机丢点（模拟遮挡，只在点数充足时执行）
        if np.random.random() < 0.3 and len(pc) > self.num_points:
            keep = max(int(len(pc) * 0.85), self.num_points)
            keep = min(keep, len(pc))  # 确保不超过实际点数
            idx = np.random.choice(len(pc), keep, replace=False)
            pc = pc[idx]

        return pc

    def __getitem__(self, idx):
        file_path, backfat = self.samples[idx]

        # 读取点云
        try:
            pc = np.loadtxt(file_path, dtype=np.float32)
        except Exception as e:
            # 容错：如果文件损坏，返回零点云
            print(f"Warning: failed to load {file_path}: {e}")
            pc = np.zeros((self.num_points, 3), dtype=np.float32)

        if pc.ndim != 2 or pc.shape[1] < 3:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)
        pc = pc[:, :3]  # 只取 xyz

        # 数据增强（在归一化之前）
        if self.augment:
            pc = self._augment(pc)

        # 统一点数：随机采样或补充
        n = len(pc)
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            pc = pc[idx]
        elif n < self.num_points:
            idx = np.random.choice(n, self.num_points, replace=True)
            pc = pc[idx]

        # 仅中心化，保留点云的相对尺度信息（不做单位球归一化）
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid

        # 转为 tensor，channels-first [3, N]
        pc_tensor = torch.from_numpy(pc.T).float()  # [3, N]
        label_tensor = torch.tensor([backfat], dtype=torch.float32)

        return pc_tensor, label_tensor


# ==============================================================================
# 4. 数据准备：从 docx/xlsx 读取标签，匹配点云路径
# ==============================================================================

def load_labels_from_docx_nky1(docx_path):
    """读取第一次数据 docx，返回 {ear_tag: backfat}。"""
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 9:
                ear_tag = cells[1]
                bf_str = cells[8]
                if ear_tag and bf_str and '-' in ear_tag:
                    try:
                        labels[ear_tag] = float(bf_str)
                    except ValueError:
                        pass
    return labels


def load_labels_from_docx_nky2(docx_path):
    """读取第二次数据 docx，返回 {ear_tag: backfat}。"""
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 13:
                ear_tag = cells[1]
                # B超背膘可能在第11或12列
                bf_str = cells[11] if cells[11] else (cells[12] if len(cells) > 12 else '')
                if ear_tag and bf_str and '-' in ear_tag:
                    try:
                        labels[ear_tag] = float(bf_str)
                    except ValueError:
                        pass
    return labels


def load_labels_from_docx_nky3(docx_path):
    """读取第三次数据 docx（按栋分表），返回 {dong-short_id: backfat}。"""
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    dong_num = None
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            header_text = ' '.join(cells)
            for d in [10, 11, 12, 13]:
                if f'第{d}栋' in header_text or f'{d}栋' in header_text:
                    dong_num = d
                    break
            if dong_num and len(cells) >= 6:
                ear_short = cells[1]
                bf_str = cells[5]
                if ear_short and bf_str and cells[0].isdigit():
                    try:
                        bf = float(bf_str)
                        full_id = f'{dong_num}-{ear_short}'
                        labels[full_id] = bf
                    except ValueError:
                        pass
    return labels


def load_labels_from_xlsx_fmh(xlsx_path):
    """读取丰美禾背膘记录表，返回 {(seq, tag): backfat}。"""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Sheet1']
    entries = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        cells = [str(c).strip() if c else '' for c in row[:3]]
        seq_str, tag, bf_str = cells
        if seq_str and tag and bf_str:
            try:
                entries.append((int(seq_str), tag, float(bf_str)))
            except ValueError:
                pass
    return entries


def build_dataset_registry(dk_base, fmh_base):
    """扫描所有数据目录，匹配标签，返回 {pig_uid: (data_dir, backfat, n_frames)}。"""
    registry = {}

    # === nkyfirst ===
    labels1 = load_labels_from_docx_nky1(os.path.join(dk_base, '第一次数据7.29.docx'))
    nky1_dir = os.path.join(dk_base, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        for pig_id in os.listdir(nky1_dir):
            if pig_id in labels1:
                for master in ['master', 'master10']:
                    pp = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', master, 'processed_pointclouds')
                    if os.path.isdir(pp):
                        files = [f for f in os.listdir(pp) if f.startswith('pig_') and f.endswith('.txt')]
                        if files:
                            registry[f'nky1_{pig_id}'] = (pp, labels1[pig_id], len(files))
                            break

    # === nkysecond ===
    labels2 = load_labels_from_docx_nky2(os.path.join(dk_base, '第二次数据9.6.docx'))
    nky2_dir = os.path.join(dk_base, 'nkysecond')
    if os.path.isdir(nky2_dir):
        for pig_id in os.listdir(nky2_dir):
            if pig_id in labels2:
                for master in ['master', 'master10']:
                    pp = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', master, 'processed_pointclouds')
                    if os.path.isdir(pp):
                        files = [f for f in os.listdir(pp) if f.startswith('pig_') and f.endswith('.txt')]
                        if files:
                            registry[f'nky2_{pig_id}'] = (pp, labels2[pig_id], len(files))
                            break

    # === nkythird ===
    labels3 = load_labels_from_docx_nky3(os.path.join(dk_base, '第三次数据12.9.docx'))
    nky3_dir = os.path.join(dk_base, 'nkythird')
    if os.path.isdir(nky3_dir):
        for date_dir in os.listdir(nky3_dir):
            date_path = os.path.join(nky3_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
            for pig_dir in os.listdir(date_path):
                if pig_dir in labels3:
                    for master in ['master', 'master10']:
                        pp = os.path.join(date_path, pig_dir, 'depth_cz_r_d_p', master, 'processed_pointclouds')
                        if os.path.isdir(pp):
                            files = [f for f in os.listdir(pp) if f.startswith('pig_') and f.endswith('.txt')]
                            if files:
                                registry[f'nky3_{pig_dir}'] = (pp, labels3[pig_dir], len(files))
                                break

    # === 丰美禾 ===
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    if os.path.isfile(xlsx_path):
        fmh_entries = load_labels_from_xlsx_fmh(xlsx_path)
        fmh_data_dir = os.path.join(fmh_base, '10.29')
        if os.path.isdir(fmh_data_dir):
            # 解析目录名: "10LY1198" -> (10, "LY1198")
            dir_map = {}
            for d in os.listdir(fmh_data_dir):
                m = re.match(r'^(\d+)(.*)', d)
                if m:
                    dir_map[(int(m.group(1)), m.group(2))] = d

            for seq, tag, bf in fmh_entries:
                tag_clean = tag.strip()
                for k, v in dir_map.items():
                    if k[0] == seq and (k[1] == tag_clean or
                                        k[1].lstrip('0') == tag_clean.lstrip('0')):
                        pp = os.path.join(fmh_data_dir, v, 'master', 'processed_pointclouds')
                        if os.path.isdir(pp):
                            files = [f for f in os.listdir(pp) if f.endswith('.txt')]
                            if files:
                                registry[f'fmh_{v}'] = (pp, bf, len(files))
                        break

    return registry


def split_by_pig_id(registry, train_ratio=0.8, val_ratio=0.1, seed=42):
    """按猪 ID 分割数据集（同一头猪的所有帧在同一个集合中）。"""
    pig_ids = sorted(registry.keys())
    rng = random.Random(seed)
    rng.shuffle(pig_ids)

    n = len(pig_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = pig_ids[:n_train]
    val_ids = pig_ids[n_train:n_train + n_val]
    test_ids = pig_ids[n_train + n_val:]

    def make_samples(ids):
        samples = []
        for pid in ids:
            data_dir, bf, n_frames = registry[pid]
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.txt'):
                    samples.append((os.path.join(data_dir, f), bf))
        return samples

    return make_samples(train_ids), make_samples(val_ids), make_samples(test_ids), \
           train_ids, val_ids, test_ids


# ==============================================================================
# 5. 训练和评估
# ==============================================================================

def compute_metrics(preds, targets):
    """计算回归指标：MAE, RMSE, R2, MAPE。"""
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch_idx, (points, labels) in enumerate(loader):
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy().flatten())
        all_targets.extend(labels.detach().cpu().numpy().flatten())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for points, labels in loader:
        points = points.to(device)
        labels = labels.to(device)
        outputs = model(points)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics, all_preds, all_targets


def plot_results(train_losses, val_losses, train_r2s, val_r2s,
                 test_preds, test_targets, save_dir):
    """绘制训练曲线和测试散点图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss 曲线
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # R² 曲线
    axes[1].plot(train_r2s, label='Train R2', color='blue')
    axes[1].plot(val_r2s, label='Val R2', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2')
    axes[1].set_title('Training & Validation R2')
    axes[1].legend()
    axes[1].grid(True)

    # 测试集散点图
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    axes[2].scatter(test_targets, test_preds, alpha=0.3, s=10, color='blue')
    min_val = min(test_targets.min(), test_preds.min()) - 1
    max_val = max(test_targets.max(), test_preds.max()) + 1
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    axes[2].set_xlabel('True Backfat (mm)')
    axes[2].set_ylabel('Predicted Backfat (mm)')
    metrics = compute_metrics(test_preds, test_targets)
    axes[2].set_title(f'Test: R2={metrics["R2"]:.3f}, MAE={metrics["MAE"]:.2f}mm')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'), dpi=150)
    plt.close()
    print(f"Results plot saved to {os.path.join(save_dir, 'training_results.png')}")


# ==============================================================================
# 6. 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser('PointMLP-Elite Backfat Regression')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dk_base', type=str, default=r'G:\耿来硕数据\DK')
    parser.add_argument('--fmh_base', type=str, default=r'G:\耿来硕数据\丰美禾数据')
    parser.add_argument('--save_dir', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\网络\pointmlp_backfat\checkpoints')
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"{'='*60}")
    print(f"PointMLP-Elite Backfat Regression Training")
    print(f"Time: {timestamp}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # --- 构建数据注册表 ---
    print("\n[1/5] Building dataset registry...")
    registry = build_dataset_registry(args.dk_base, args.fmh_base)
    print(f"  Found {len(registry)} pigs with matched labels and point clouds")

    if len(registry) == 0:
        print("ERROR: No data found! Check paths.")
        return

    # --- 分割数据集 ---
    print("\n[2/5] Splitting dataset by pig ID...")
    train_samples, val_samples, test_samples, \
        train_ids, val_ids, test_ids = split_by_pig_id(registry, seed=args.seed)

    print(f"  Train: {len(train_ids)} pigs, {len(train_samples)} samples")
    print(f"  Val:   {len(val_ids)} pigs, {len(val_samples)} samples")
    print(f"  Test:  {len(test_ids)} pigs, {len(test_samples)} samples")

    # 保存分割信息
    split_info = {
        'train_pigs': train_ids,
        'val_pigs': val_ids,
        'test_pigs': test_ids,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
    }
    with open(os.path.join(args.save_dir, 'split_info.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    # 打印标签分布
    train_labels = [s[1] for s in train_samples]
    val_labels = [s[1] for s in val_samples]
    test_labels = [s[1] for s in test_samples]
    print(f"  Train label range: {min(train_labels):.1f} - {max(train_labels):.1f} mm "
          f"(mean={np.mean(train_labels):.1f})")
    print(f"  Val   label range: {min(val_labels):.1f} - {max(val_labels):.1f} mm "
          f"(mean={np.mean(val_labels):.1f})")
    print(f"  Test  label range: {min(test_labels):.1f} - {max(test_labels):.1f} mm "
          f"(mean={np.mean(test_labels):.1f})")

    # --- 创建 DataLoader ---
    print("\n[3/5] Creating data loaders...")
    train_dataset = BackfatPointCloudDataset(train_samples, args.num_points, augment=True)
    val_dataset = BackfatPointCloudDataset(val_samples, args.num_points, augment=False)
    test_dataset = BackfatPointCloudDataset(test_samples, args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- 初始化模型 ---
    print("\n[4/5] Initializing PointMLP-Elite regressor...")
    model = PointMLPRegressor(points=args.num_points).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    # --- 训练循环 ---
    print(f"\n[5/5] Training for {args.epochs} epochs...")
    print(f"{'='*60}")

    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience = 30
    patience_counter = 0

    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.epochs} (lr={lr:.6f})")

        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )

        # 验证
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_metrics['R2'])
        val_r2s.append(val_metrics['R2'])

        print(f"  Train - Loss: {train_loss:.4f} | MAE: {train_metrics['MAE']:.2f}mm | "
              f"RMSE: {train_metrics['RMSE']:.2f}mm | R2: {train_metrics['R2']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | MAE: {val_metrics['MAE']:.2f}mm | "
              f"RMSE: {val_metrics['RMSE']:.2f}mm | R2: {val_metrics['R2']:.4f}")

        # 保存最优模型（按 val loss）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_metrics['R2']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  * Best model saved! (Val Loss: {val_loss:.4f}, R2: {val_metrics['R2']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # 每 20 个 epoch 保存一次 checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    # --- 测试 ---
    print(f"\n{'='*60}")
    print("Final Testing with best model...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded best model from epoch {checkpoint['epoch']}")

    test_loss, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n  Test Results:")
    print(f"    Loss:  {test_loss:.4f}")
    print(f"    MAE:   {test_metrics['MAE']:.2f} mm")
    print(f"    RMSE:  {test_metrics['RMSE']:.2f} mm")
    print(f"    R2:    {test_metrics['R2']:.4f}")
    print(f"    MAPE:  {test_metrics['MAPE']:.2f}%")

    # 按猪计算平均预测（每头猪多帧取均值）
    print(f"\n  Per-pig averaged predictions:")
    pig_preds = defaultdict(list)
    pig_targets = defaultdict(float)
    sample_idx = 0
    for pid in test_ids:
        data_dir, bf, n_frames = registry[pid]
        pig_targets[pid] = bf
        for _ in range(n_frames):
            if sample_idx < len(test_preds):
                pig_preds[pid].append(test_preds[sample_idx])
                sample_idx += 1

    pig_avg_preds = []
    pig_avg_targets = []
    for pid in test_ids:
        if pig_preds[pid]:
            avg_pred = np.mean(pig_preds[pid])
            true_val = pig_targets[pid]
            pig_avg_preds.append(avg_pred)
            pig_avg_targets.append(true_val)
            print(f"    {pid}: True={true_val:.1f}mm, Pred={avg_pred:.1f}mm, "
                  f"Error={abs(avg_pred - true_val):.1f}mm")

    if pig_avg_preds:
        pig_metrics = compute_metrics(pig_avg_preds, pig_avg_targets)
        print(f"\n  Per-pig Metrics:")
        print(f"    MAE:  {pig_metrics['MAE']:.2f} mm")
        print(f"    RMSE: {pig_metrics['RMSE']:.2f} mm")
        print(f"    R2:   {pig_metrics['R2']:.4f}")

    # 绘图
    plot_results(train_losses, val_losses, train_r2s, val_r2s,
                 test_preds, test_targets, args.save_dir)

    # 保存完整结果
    results = {
        'best_epoch': checkpoint['epoch'],
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'pig_metrics': {k: float(v) for k, v in pig_metrics.items()} if pig_avg_preds else {},
        'args': vars(args),
        'total_pigs': len(registry),
        'train_pigs': len(train_ids),
        'val_pigs': len(val_ids),
        'test_pigs': len(test_ids),
    }
    with open(os.path.join(args.save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"All outputs saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
