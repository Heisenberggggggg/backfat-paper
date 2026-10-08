"""
PointMLP-Elite v3 背膘回归训练脚本（方案一+二+四联合改进）
=========================================================
背景：
  v2 训练结果严重过拟合（训练R2=0.69，验证R2~0，测试R2=-0.06），
  方案三（传统ML+手工特征）验证了点云几何特征中背膘信号微弱（XGBoost测试R2=0.15）。

本版改进（三管齐下）：
  方案一 — 强化数据增强：加大旋转/缩放/抖动力度，新增随机平移和镜像翻转，
           迫使模型不能通过"记形状"来预测，必须学习泛化特征。
  方案二 — 缩减模型容量：通道数砍半（32→16 embed, 各stage减半），参数量从48万降到~7万，
           dropout提到0.6，weight_decay提到0.01，限制过拟合能力。
  方案四 — 混合特征：将方案三提取的41维手工统计特征与PointMLP的深度特征拼接，
           让模型同时利用两种信息来源，手工特征提供"保底"背膘信号。

改动范围：仅此脚本，不改 train.py / extract_features.py 等现有代码。
输出保存到 checkpoints_v3/，不覆盖之前任何结果。

Author: 耿来硕 + Claude
Date: 2026-03-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import sys
import re
import json
import csv
import random
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# 复用 train.py 中的数据注册和分割函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (build_dataset_registry, square_distance, farthest_point_sample,
                   index_points, knn_point, get_activation, compute_metrics)


# ==============================================================================
# 1. 模型组件（与 train.py 相同，此处完整复制以保持独立性）
# ==============================================================================

class LocalGrouper(nn.Module):
    """局部分组模块：FPS + KNN + 几何仿射归一化。"""
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
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
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


# ==============================================================================
# 2. 方案二改进：缩小版 PointMLP-Elite + 方案四：混合特征回归头
# ==============================================================================

class PointMLPRegressorV3(nn.Module):
    """PointMLP-Elite v3 — 缩小版 + 混合特征回归头。

    与 v2 的区别：
    方案二 — embed_dim 32->16, 各stage通道砍半, dropout 0.3->0.6
    方案四 — 回归头额外接收41维手工特征，与深度全局特征拼接后一起预测
    """
    def __init__(self, points=1024,
                 embed_dim=16,           # v2是32，砍半
                 groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1],
                 pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
                 handcraft_feat_dim=41,   # 方案四：手工特征维度
                 dropout=0.6):            # v2是0.3，加大
        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.handcraft_feat_dim = handcraft_feat_dim
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

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
        self.deep_feat_dim = last_channel  # PointMLP输出的深度特征维度

        # 方案四：混合回归头
        # 输入 = 深度特征(last_channel) + 手工特征(handcraft_feat_dim)
        combined_dim = last_channel + handcraft_feat_dim
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, handcraft_feat=None):
        """
        Input:
            x: [B, 3, N] — 点云 channels-first
            handcraft_feat: [B, 41] — 手工统计特征（可选，None时用零填充）
        Output: [B, 1] — 预测背膘厚度
        """
        xyz = x.permute(0, 2, 1)  # [B, N, 3]
        x = self.embedding(x)     # [B, embed_dim, N]

        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)

        deep_feat = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # [B, last_channel]

        # 方案四：拼接手工特征
        if handcraft_feat is not None:
            combined = torch.cat([deep_feat, handcraft_feat], dim=1)
        else:
            # 如果没提供手工特征，用零填充
            zeros = torch.zeros(deep_feat.size(0), self.handcraft_feat_dim,
                                device=deep_feat.device)
            combined = torch.cat([deep_feat, zeros], dim=1)

        out = self.regressor(combined)  # [B, 1]
        return out


# ==============================================================================
# 3. 方案四：加载手工特征的数据集 + 方案一：强化数据增强
# ==============================================================================

class BackfatDatasetV3(Dataset):
    """v3数据集：点云 + 手工特征 + 强化增强。

    方案一改进：增强力度全面加大
    方案四改进：每个样本额外返回对应猪的41维手工统计特征
    """
    def __init__(self, samples, pig_features, num_points=1024, augment=False):
        """
        Args:
            samples: list of (file_path, backfat_value, pig_id)
            pig_features: {pig_id: np.array of shape (41,)} — 各猪的手工特征
            num_points: 统一采样点数
            augment: 是否数据增强
        """
        self.samples = samples
        self.pig_features = pig_features
        self.num_points = num_points
        self.augment = augment
        self.feat_dim = next(iter(pig_features.values())).shape[0] if pig_features else 41

    def __len__(self):
        return len(self.samples)

    def _augment(self, pc):
        """方案一：强化数据增强。

        与v2对比：
        - 缩放范围: 0.95-1.05 -> 0.85-1.15（加大）
        - 抖动sigma: 0.005 -> 0.01（翻倍）
        - 丢点概率: 0.3 -> 0.5（提高）
        - 新增: 随机平移（各轴正负5%范围）
        - 新增: X轴镜像翻转（50%概率）
        """
        # 1. Z轴随机旋转（0-360度）
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0],
                       [sin_t, cos_t, 0],
                       [0, 0, 1]], dtype=np.float32)
        pc = pc @ R.T

        # 2. 随机平移（新增！各轴正负5%范围）
        ranges = np.ptp(pc, axis=0)  # 各轴极差
        shift = np.random.uniform(-0.05, 0.05, size=3).astype(np.float32) * ranges
        pc = pc + shift

        # 3. 随机各向同性缩放（范围加大）
        scale = np.random.uniform(0.85, 1.15)
        pc *= scale

        # 4. 随机抖动（sigma翻倍）
        pc += np.random.normal(0, 0.01, size=pc.shape).astype(np.float32)

        # 5. X轴镜像翻转（新增！50%概率）
        if np.random.random() < 0.5:
            pc[:, 0] = -pc[:, 0]

        # 6. 随机丢点（概率提高）
        if np.random.random() < 0.5 and len(pc) > self.num_points:
            keep = max(int(len(pc) * np.random.uniform(0.75, 0.95)), self.num_points)
            keep = min(keep, len(pc))
            idx = np.random.choice(len(pc), keep, replace=False)
            pc = pc[idx]

        return pc

    def __getitem__(self, idx):
        file_path, backfat, pig_id = self.samples[idx]

        # 读取点云
        try:
            pc = np.loadtxt(file_path, dtype=np.float32)
        except Exception:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)

        if pc.ndim != 2 or pc.shape[1] < 3:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)
        pc = pc[:, :3]

        # 数据增强
        if self.augment:
            pc = self._augment(pc)

        # 统一点数
        n = len(pc)
        if n > self.num_points:
            sel = np.random.choice(n, self.num_points, replace=False)
            pc = pc[sel]
        elif n < self.num_points:
            sel = np.random.choice(n, self.num_points, replace=True)
            pc = pc[sel]

        # 仅中心化，保留尺度
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid

        # 转 tensor
        pc_tensor = torch.from_numpy(pc.T).float()  # [3, N]
        label_tensor = torch.tensor([backfat], dtype=torch.float32)

        # 方案四：获取手工特征
        if pig_id in self.pig_features:
            feat_tensor = torch.from_numpy(self.pig_features[pig_id]).float()
        else:
            feat_tensor = torch.zeros(self.feat_dim, dtype=torch.float32)

        return pc_tensor, label_tensor, feat_tensor


# ==============================================================================
# 4. 训练和评估（适配三元组输出）
# ==============================================================================

def train_one_epoch_v3(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch_idx, (points, labels, feats) in enumerate(loader):
        points = points.to(device)
        labels = labels.to(device)
        feats = feats.to(device)

        optimizer.zero_grad()
        outputs = model(points, feats)
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
def evaluate_v3(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for points, labels, feats in loader:
        points = points.to(device)
        labels = labels.to(device)
        feats = feats.to(device)
        outputs = model(points, feats)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics, all_preds, all_targets


# ==============================================================================
# 5. 工具函数
# ==============================================================================

def load_pig_features(csv_path):
    """从 features_per_pig.csv 加载手工特征。返回 {pig_id: np.array}。"""
    pig_features = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        feature_names = [col for col in reader.fieldnames
                         if col not in ('pig_id', 'split', 'backfat', 'n_frames')]
        for row in reader:
            pid = row['pig_id']
            feat = np.array([float(row[fn]) for fn in feature_names], dtype=np.float32)
            pig_features[pid] = feat
    return pig_features, feature_names


def plot_results_v3(train_losses, val_losses, train_r2s, val_r2s,
                    test_preds, test_targets, save_dir):
    """绘制训练曲线和测试散点图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_r2s, label='Train R2', color='blue')
    axes[1].plot(val_r2s, label='Val R2', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2')
    axes[1].set_title('Training & Validation R2')
    axes[1].legend()
    axes[1].grid(True)

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
    plt.savefig(os.path.join(save_dir, 'training_results_v3.png'), dpi=150)
    plt.close()


# ==============================================================================
# 6. 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser('PointMLP-Elite v3 Backfat Regression')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.01)    # v2是0.0002，提高50倍
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dk_base', type=str, default=r'G:\耿来硕数据\DK')
    parser.add_argument('--fmh_base', type=str, default=r'G:\耿来硕数据\丰美禾数据')
    parser.add_argument('--save_dir', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\网络\pointmlp_backfat\checkpoints_v3')
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

    # 日志文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'training_log_v3.txt')

    def log(msg):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("PointMLP-Elite v3 Backfat Regression Training")
    log("Improvements: stronger augmentation + smaller model + hybrid features")
    log(f"Device: {device}")
    log("=" * 60)

    # --- 加载手工特征（方案四）---
    feat_csv = os.path.join(script_dir, 'traditional_ml', 'features_per_pig.csv')
    if os.path.exists(feat_csv):
        pig_features, feat_names = load_pig_features(feat_csv)
        handcraft_dim = len(feat_names)
        log(f"Loaded handcraft features: {len(pig_features)} pigs, {handcraft_dim} features")

        # 对手工特征做标准化（用所有猪的均值/标准差）
        all_feats = np.stack(list(pig_features.values()))
        feat_mean = np.mean(all_feats, axis=0)
        feat_std = np.std(all_feats, axis=0) + 1e-8
        for pid in pig_features:
            pig_features[pid] = ((pig_features[pid] - feat_mean) / feat_std).astype(np.float32)
        log("Handcraft features standardized (zero-mean, unit-variance)")
    else:
        log(f"WARNING: {feat_csv} not found! Running without handcraft features.")
        pig_features = {}
        handcraft_dim = 41

    # --- 构建数据注册表 ---
    log("\n[1/5] Building dataset registry...")
    registry = build_dataset_registry(args.dk_base, args.fmh_base)
    log(f"  Found {len(registry)} pigs")

    if len(registry) == 0:
        log("ERROR: No data found!")
        return

    # --- 使用与v2完全相同的数据划分 ---
    log("\n[2/5] Loading data split from split_info.json...")
    split_path = os.path.join(script_dir, 'checkpoints', 'split_info.json')
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split_info = json.load(f)
        train_ids = split_info['train_pigs']
        val_ids = split_info['val_pigs']
        test_ids = split_info['test_pigs']
        log(f"  Loaded existing split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        log("  split_info.json not found, creating new split...")
        from train import split_by_pig_id
        _, _, _, train_ids, val_ids, test_ids = split_by_pig_id(registry, seed=args.seed)

    # 构建样本列表（加入 pig_id）
    def make_samples(ids):
        samples = []
        for pid in ids:
            if pid not in registry:
                continue
            data_dir, bf, n_frames = registry[pid]
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.txt'):
                    samples.append((os.path.join(data_dir, f), bf, pid))
        return samples

    train_samples = make_samples(train_ids)
    val_samples = make_samples(val_ids)
    test_samples = make_samples(test_ids)

    log(f"  Train: {len(train_ids)} pigs, {len(train_samples)} samples")
    log(f"  Val:   {len(val_ids)} pigs, {len(val_samples)} samples")
    log(f"  Test:  {len(test_ids)} pigs, {len(test_samples)} samples")

    train_labels = [s[1] for s in train_samples]
    val_labels = [s[1] for s in val_samples]
    test_labels = [s[1] for s in test_samples]
    log(f"  Train labels: {min(train_labels):.1f}-{max(train_labels):.1f}mm (mean={np.mean(train_labels):.1f})")
    log(f"  Val labels:   {min(val_labels):.1f}-{max(val_labels):.1f}mm (mean={np.mean(val_labels):.1f})")
    log(f"  Test labels:  {min(test_labels):.1f}-{max(test_labels):.1f}mm (mean={np.mean(test_labels):.1f})")

    # --- 创建 DataLoader ---
    log("\n[3/5] Creating data loaders...")
    train_dataset = BackfatDatasetV3(train_samples, pig_features, args.num_points, augment=True)
    val_dataset = BackfatDatasetV3(val_samples, pig_features, args.num_points, augment=False)
    test_dataset = BackfatDatasetV3(test_samples, pig_features, args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- 初始化模型（方案二：缩小版 + 方案四：混合特征）---
    log("\n[4/5] Initializing PointMLP-Elite v3 (smaller + hybrid)...")
    model = PointMLPRegressorV3(
        points=args.num_points,
        embed_dim=16,          # v2是32，砍半
        handcraft_feat_dim=handcraft_dim,
        dropout=0.6            # v2是0.3，翻倍
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total parameters:     {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")
    log(f"  (v2 had 483,105 params -- this is {total_params/483105*100:.1f}% of v2)")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    log(f"\n  Hyperparams: lr={args.lr}, weight_decay={args.weight_decay}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}")

    # --- 训练循环 ---
    log(f"\n[5/5] Training for {args.epochs} epochs...")
    log("=" * 60)

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0

    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_metrics = train_one_epoch_v3(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
        val_loss, val_metrics, _, _ = evaluate_v3(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2s.append(train_metrics['R2'])
        val_r2s.append(val_metrics['R2'])

        log(f"Epoch {epoch}/{args.epochs} (lr={lr:.6f})")
        log(f"  Train - Loss: {train_loss:.4f} | MAE: {train_metrics['MAE']:.2f}mm | "
            f"R2: {train_metrics['R2']:.4f}")
        log(f"  Val   - Loss: {val_loss:.4f} | MAE: {val_metrics['MAE']:.2f}mm | "
            f"R2: {val_metrics['R2']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best_model_v3.pth'))
            log(f"  * Best model saved! (Val Loss: {val_loss:.4f}, R2: {val_metrics['R2']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    # --- 测试 ---
    log(f"\n{'=' * 60}")
    log("Final Testing with best model...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model_v3.pth'),
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    log(f"  Loaded best model from epoch {checkpoint['epoch']}")

    test_loss, test_metrics, test_preds, test_targets = evaluate_v3(
        model, test_loader, criterion, device
    )
    log(f"\n  Test Results (frame-level):")
    log(f"    Loss:  {test_loss:.4f}")
    log(f"    MAE:   {test_metrics['MAE']:.2f} mm")
    log(f"    RMSE:  {test_metrics['RMSE']:.2f} mm")
    log(f"    R2:    {test_metrics['R2']:.4f}")
    log(f"    MAPE:  {test_metrics['MAPE']:.2f}%")

    # 按猪平均预测
    log(f"\n  Per-pig averaged predictions:")
    pig_preds_dict = defaultdict(list)
    pig_targets_dict = defaultdict(float)
    sample_idx = 0
    for pid in test_ids:
        if pid not in registry:
            continue
        data_dir, bf, n_frames = registry[pid]
        pig_targets_dict[pid] = bf
        for _ in range(n_frames):
            if sample_idx < len(test_preds):
                pig_preds_dict[pid].append(test_preds[sample_idx])
                sample_idx += 1

    pig_avg_preds = []
    pig_avg_targets = []
    for pid in test_ids:
        if pig_preds_dict[pid]:
            avg_pred = np.mean(pig_preds_dict[pid])
            true_val = pig_targets_dict[pid]
            pig_avg_preds.append(avg_pred)
            pig_avg_targets.append(true_val)
            err = abs(avg_pred - true_val)
            flag = " <-- large error" if err > 3.0 else ""
            log(f"    {pid}: True={true_val:.1f}mm, Pred={avg_pred:.1f}mm, "
                f"Error={err:.1f}mm{flag}")

    if pig_avg_preds:
        pig_metrics = compute_metrics(pig_avg_preds, pig_avg_targets)
        log(f"\n  Per-pig Metrics:")
        log(f"    MAE:  {pig_metrics['MAE']:.2f} mm")
        log(f"    RMSE: {pig_metrics['RMSE']:.2f} mm")
        log(f"    R2:   {pig_metrics['R2']:.4f}")
        log(f"    MAPE: {pig_metrics['MAPE']:.2f}%")

    # 对比总结
    log(f"\n{'=' * 60}")
    log("Version Comparison:")
    log(f"  {'Method':<25} {'Test R2':>10} {'Test MAE':>10} {'Test RMSE':>10}")
    log(f"  {'-' * 55}")
    log(f"  {'PointMLP v2':<25} {'-0.0597':>10} {'2.72mm':>10} {'3.28mm':>10}")
    log(f"  {'XGBoost (plan3)':<25} {'0.1506':>10} {'2.51mm':>10} {'2.94mm':>10}")
    if pig_avg_preds:
        log(f"  {'PointMLP v3 (plan1+2+4)':<25} "
            f"{pig_metrics['R2']:>10.4f} "
            f"{pig_metrics['MAE']:>9.2f}mm "
            f"{pig_metrics['RMSE']:>9.2f}mm")

    # 绘图
    plot_results_v3(train_losses, val_losses, train_r2s, val_r2s,
                    test_preds, test_targets, args.save_dir)
    log(f"Results plot saved to {args.save_dir}")

    # 保存结果JSON
    results = {
        'version': 'v3 (plan1+2+4)',
        'improvements': ['stronger augmentation', 'smaller model', 'hybrid handcraft features'],
        'best_epoch': checkpoint['epoch'],
        'test_metrics_frame': {k: float(v) for k, v in test_metrics.items()},
        'test_metrics_pig': {k: float(v) for k, v in pig_metrics.items()} if pig_avg_preds else {},
        'model_params': total_params,
        'args': vars(args),
    }
    with open(os.path.join(args.save_dir, 'results_v3.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"\n{'=' * 60}")
    log("Training complete!")
    log(f"All outputs saved to: {args.save_dir}")
    log("=" * 60)


if __name__ == '__main__':
    main()
