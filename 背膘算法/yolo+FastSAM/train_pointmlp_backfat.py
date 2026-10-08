# -*- coding: utf-8 -*-
"""
基于官方PointMLP-elite的猪背膘回归训练脚本

架构来源: https://github.com/ma-xu/pointMLP-pytorch
改动:
  1. 分类头 → 回归头 (输出1维, 无Softmax)
  2. FPS用纯PyTorch实现 (不依赖CUDA扩展pointnet2_ops)
  3. 使用PointMLP-elite配置 (0.68M参数, 适合小数据集)
  4. 保留几何仿射模块 (Geometric Affine Module)
  5. 损失函数: SmoothL1Loss

数据来源:
  - DK nkyfirst: 39头猪, 背膘10-22mm
  - 丰美禾 10.29/10.30: 71头猪, 背膘12-29mm
  - 合计: 110头猪, ~19600帧
"""

import os
import re
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# 1. PointMLP 模型 (基于官方代码, 纯PyTorch, 无CUDA扩展依赖)
# ============================================================

def square_distance(src, dst):
    """欧氏距离平方矩阵 [B,N,M]"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """按索引取点 points[B,N,C], idx[B,S,...] -> [B,S,...,C]"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """纯PyTorch实现的最远点采样 (FPS)
    Input:  xyz [B, N, 3]
    Output: centroids [B, npoint] (索引)
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    """K近邻查询
    Input:  xyz [B, N, C], new_xyz [B, S, C]
    Output: group_idx [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    """局部分组 + 几何仿射模块 (Geometric Affine Module)

    核心创新: 对局部邻域特征做自适应标准化,
    隐式编码局部几何信息(通道级均值/方差).
    """
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center"):
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

        # FPS 降采样
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)       # [B, S, 3]
        new_points = index_points(points, fps_idx)  # [B, S, d]

        # kNN 邻域查询
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)         # [B, S, k, 3]
        grouped_points = index_points(points, idx)   # [B, S, k, d]

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        # 几何仿射模块: 标准化 + 可学习缩放/偏移
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            elif self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std(
                (grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True
            ).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        # 拼接锚点特征
        new_points = torch.cat([
            grouped_points,
            new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)
        ], dim=-1)

        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='relu'):
        super().__init__()
        if activation.lower() == 'gelu':
            act = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            act = nn.LeakyReLU(inplace=True)
        else:
            act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=bias),
            nn.BatchNorm1d(out_channels),
            act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    """残差MLP块 (论文核心: FC→BN→ReLU→FC→BN + shortcut)"""
    def __init__(self, channel, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super().__init__()
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'leakyrelu':
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), 1, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(int(channel * res_expansion), channel, 1, bias=bias),
            nn.BatchNorm1d(channel)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)  # 残差连接


class PreExtraction(nn.Module):
    """聚合前的特征提取 Φ_pre"""
    def __init__(self, channels, out_channels, blocks=1, groups=1,
                 res_expansion=1, bias=True, activation='relu', use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    """聚合后的特征提取 Φ_pos"""
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1,
                 bias=True, activation='relu'):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class PointMLPRegressor(nn.Module):
    """PointMLP-elite 回归模型

    与官方分类模型的区别:
      - 输出层: Linear(256, 1), 无Softmax
      - 使用elite配置: embed_dim=32, res_expansion=0.25
      - 4-stage分层结构 + 几何仿射模块 完整保留
    """
    def __init__(self, points=1024, embed_dim=32, groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1],
                 pos_blocks=[1, 1, 2, 1], k_neighbors=[24, 24, 24, 24],
                 reducers=[2, 2, 2, 2]):
        super().__init__()
        self.stages = len(pre_blocks)
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion)

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

        # 回归头 (替代原分类头)
        if activation.lower() == 'gelu':
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)
        self.regressor = nn.Sequential(
            nn.Linear(last_channel, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            act,
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, N, 3) — 点云坐标
        xyz = x  # [B, N, 3]
        x = x.permute(0, 2, 1)  # [B, 3, N]
        x = self.embedding(x)   # [B, embed_dim, N]

        for i in range(self.stages):
            # FPS降采样 + kNN分组 + 几何仿射 + 特征提取
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)

        # 全局最大池化
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)  # [B, C]
        x = self.regressor(x)  # [B, 1]
        return x.squeeze(-1)   # [B]


# ============================================================
# 2. 数据集
# ============================================================

def collect_all_pig_data():
    """收集所有可用的猪只数据 (DK + 丰美禾)

    Returns:
        list of dict: [{pig_id, backfat, pc_dir, source}, ...]
    """
    all_pigs = []

    # --- DK nkyfirst ---
    BACKFAT_DK = {
        '2-1': 16.1, '2-2': 16.0, '2-3': 21.5, '2-5': 14.6,
        '2-8': 15.2, '2-17': 18.8, '2-18': 19.6, '2-22': 17.9,
        '2-25': 13.8, '2-26': 19.8, '2-27': 13.5, '2-28': 12.8,
        '2-29': 17.7, '2-30': 18.1, '2-31': 18.2, '2-32': 18.5,
        '2-33': 13.8, '2-34': 18.8,
        '3-1': 19.0, '3-2': 15.1, '3-3': 17.2, '3-4': 14.1,
        '3-5': 14.5, '3-6': 18.4, '3-7': 13.8, '3-8': 14.1,
        '3-9': 14.5, '3-10': 16.0, '3-11': 16.8, '3-12': 17.2,
        '3-15': 18.8, '3-16': 13.7, '3-17': 10.0, '3-18': 13.9,
        '3-19': 17.6, '3-21': 18.0, '3-22': 20.2, '3-23': 22.0,
        '3-24': 17.1,
    }
    dk_base = r'G:\耿来硕数据\DK\nkyfirst'
    for pid, bf in BACKFAT_DK.items():
        for name in [pid, f'c{pid}']:
            pc_dir = os.path.join(dk_base, name, 'depth_cz_r_d_p', 'master', 'processed_pointclouds')
            if os.path.isdir(pc_dir):
                all_pigs.append({
                    'pig_id': f'dk_{pid}',
                    'backfat': bf,
                    'pc_dir': pc_dir,
                    'source': 'dk'
                })
                break

    # --- 丰美禾 ---
    try:
        import openpyxl
        wb = openpyxl.load_workbook(r'G:\耿来硕数据\丰美禾数据\背膘记录表.xlsx')
        ws = wb['Sheet1']
        tag_to_bf = {}
        for row_idx in range(1, ws.max_row + 1):
            val_b = ws.cell(row=row_idx, column=2).value
            val_c = ws.cell(row=row_idx, column=3).value
            if val_b and val_c:
                try:
                    tag_to_bf[str(val_b).strip()] = float(val_c)
                except (ValueError, TypeError):
                    pass

        fmh_base = r'G:\耿来硕数据\丰美禾数据'
        for dd in ['10.29', '10.30']:
            dd_path = os.path.join(fmh_base, dd)
            if not os.path.isdir(dd_path):
                continue
            for pig_folder in os.listdir(dd_path):
                pc_dir = os.path.join(dd_path, pig_folder, 'master', 'processed_pointclouds')
                if not os.path.isdir(pc_dir):
                    continue
                if not any(f.endswith('.txt') for f in os.listdir(pc_dir)):
                    continue
                m = re.match(r'(\d+)(.*)', pig_folder)
                tag = m.group(2) if m else pig_folder
                if tag in tag_to_bf:
                    all_pigs.append({
                        'pig_id': f'fmh_{dd}_{pig_folder}',
                        'backfat': tag_to_bf[tag],
                        'pc_dir': pc_dir,
                        'source': 'fmh'
                    })
    except ImportError:
        print("警告: openpyxl未安装, 仅使用DK数据")
    except Exception as e:
        print(f"警告: 加载丰美禾数据失败: {e}")

    return all_pigs


class BackfatPointCloudDataset(Dataset):
    """猪背膘点云数据集

    每个样本 = 一帧点云 + 背膘标签
    点云经FPS下采样到固定点数, 可选数据增强
    """
    def __init__(self, pig_entries, num_points=1024, frames_per_pig=30,
                 augment=False, seed=None):
        """
        Args:
            pig_entries: list of dict, 每个dict包含 pig_id, backfat, pc_dir
            num_points: 每帧下采样到的点数
            frames_per_pig: 每头猪随机采样的帧数 (None=全部)
            augment: 是否做数据增强
        """
        self.num_points = num_points
        self.augment = augment
        self.data = []   # list of (file_path, backfat)

        rng = np.random.RandomState(seed)

        for entry in pig_entries:
            pc_dir = entry['pc_dir']
            bf = entry['backfat']
            txt_files = sorted([f for f in os.listdir(pc_dir) if f.endswith('.txt')])

            if frames_per_pig and len(txt_files) > frames_per_pig:
                txt_files = list(rng.choice(txt_files, frames_per_pig, replace=False))

            for f in txt_files:
                self.data.append((os.path.join(pc_dir, f), bf))

    def __len__(self):
        return len(self.data)

    def _fps_numpy(self, points, n_target):
        """NumPy版FPS (数据加载时用, 比torch版更快)"""
        if len(points) <= n_target:
            idx = np.random.choice(len(points), n_target, replace=True)
            return points[idx]

        selected = np.zeros(n_target, dtype=int)
        selected[0] = np.random.randint(len(points))
        dists = np.full(len(points), np.inf)

        for i in range(1, n_target):
            p = points[selected[i - 1]]
            d = np.sum((points - p) ** 2, axis=1)
            dists = np.minimum(dists, d)
            selected[i] = np.argmax(dists)

        return points[selected]

    def __getitem__(self, idx):
        file_path, backfat = self.data[idx]
        pts = np.loadtxt(file_path, dtype=np.float32)

        if len(pts) < 100:
            # 太少的点, 填充
            pts = np.tile(pts, (self.num_points // len(pts) + 1, 1))[:self.num_points]

        # FPS下采样到目标点数
        pts = self._fps_numpy(pts, self.num_points)

        # 数据增强 (保留物理尺寸, 只做不影响尺寸的变换)
        if self.augment:
            # 绕Y轴随机旋转 (猪的朝向不固定)
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ R.T

            # 随机抖动 (模拟点云噪声)
            pts += np.random.normal(0, 0.003, pts.shape).astype(np.float32)

            # 随机缩放 (轻微, 模拟距离变化)
            scale = np.random.uniform(0.97, 1.03)
            pts *= scale

            # 随机平移 (轻微)
            pts += np.random.uniform(-0.02, 0.02, (1, 3)).astype(np.float32)

        return torch.tensor(pts, dtype=torch.float32), torch.tensor(backfat, dtype=torch.float32)


# ============================================================
# 3. 训练与评估
# ============================================================

def train_one_fold(model, train_loader, val_entries, config, fold_idx):
    """训练一折, 返回验证集逐猪预测结果"""
    device = config['device']
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        # --- 训练 ---
        model.train()
        total_loss = 0
        n_samples = 0

        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            pred = model(pts)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        avg_train_loss = total_loss / n_samples

        # --- 验证 (每5轮) ---
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for entry in val_entries:
                    pig_pred = predict_pig(model, entry, config)
                    val_preds.append(pig_pred)
                    val_trues.append(entry['backfat'])

            val_preds = np.array(val_preds)
            val_trues = np.array(val_trues)
            val_mae = mean_absolute_error(val_trues, val_preds)
            val_rmse = np.sqrt(mean_squared_error(val_trues, val_preds))

            if val_mae < best_val_loss:
                best_val_loss = val_mae
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f}, "
                      f"val_MAE={val_mae:.2f}mm, val_RMSE={val_rmse:.2f}mm")

            # 早停
            if patience_counter >= config['patience']:
                print(f"    早停 @ epoch {epoch+1} (patience={config['patience']})")
                break

    # 恢复最优模型
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # 最终验证预测
    model.eval()
    final_preds = []
    with torch.no_grad():
        for entry in val_entries:
            pig_pred = predict_pig(model, entry, config)
            final_preds.append(pig_pred)

    return final_preds


def predict_pig(model, entry, config):
    """对一头猪的所有帧取平均预测"""
    device = config['device']
    pc_dir = entry['pc_dir']
    txt_files = sorted([f for f in os.listdir(pc_dir) if f.endswith('.txt')])

    # 最多用60帧推理
    if len(txt_files) > 60:
        rng = np.random.RandomState(42)
        txt_files = list(rng.choice(txt_files, 60, replace=False))

    all_preds = []
    batch_pts = []

    for f in txt_files:
        pts = np.loadtxt(os.path.join(pc_dir, f), dtype=np.float32)
        if len(pts) < 100:
            continue
        # 简单随机下采样 (推理时不用FPS, 加速)
        if len(pts) > config['num_points']:
            idx = np.random.choice(len(pts), config['num_points'], replace=False)
            pts = pts[idx]
        elif len(pts) < config['num_points']:
            idx = np.random.choice(len(pts), config['num_points'], replace=True)
            pts = pts[idx]
        batch_pts.append(pts)

        # 每16帧批量推理
        if len(batch_pts) >= 16:
            batch_tensor = torch.tensor(np.array(batch_pts), dtype=torch.float32).to(device)
            pred = model(batch_tensor).cpu().numpy()
            all_preds.extend(pred.tolist())
            batch_pts = []

    # 处理剩余
    if batch_pts:
        batch_tensor = torch.tensor(np.array(batch_pts), dtype=torch.float32).to(device)
        pred = model(batch_tensor).cpu().numpy()
        all_preds.extend(pred.tolist())

    return np.mean(all_preds) if all_preds else 0.0


# ============================================================
# 4. 主流程
# ============================================================

def main():
    # --- 配置 ---
    config = {
        'num_points': 1024,
        'frames_per_pig': 30,      # 训练时每头猪采样帧数
        'epochs': 150,
        'batch_size': 16,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'n_folds': 5,
        'patience': 20,            # 早停耐心值 (每5轮检查一次)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }

    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])

    print("=" * 70)
    print("  PointMLP-elite 背膘回归训练")
    print("=" * 70)

    # --- 收集数据 ---
    print("\n[1/3] 收集数据...")
    all_pigs = collect_all_pig_data()

    if not all_pigs:
        print("错误: 未找到任何数据!")
        return

    pig_ids = [p['pig_id'] for p in all_pigs]
    backfats = np.array([p['backfat'] for p in all_pigs])
    n_pigs = len(all_pigs)

    n_dk = sum(1 for p in all_pigs if p['source'] == 'dk')
    n_fmh = sum(1 for p in all_pigs if p['source'] == 'fmh')

    print(f"  总计: {n_pigs} 头猪 (DK={n_dk}, 丰美禾={n_fmh})")
    print(f"  背膘范围: {backfats.min():.1f} - {backfats.max():.1f} mm")
    print(f"  背膘均值: {backfats.mean():.1f} ± {backfats.std():.1f} mm")
    print(f"  设备: {config['device']}")

    # 模型参数量
    test_model = PointMLPRegressor(points=config['num_points'])
    n_params = sum(p.numel() for p in test_model.parameters())
    print(f"  模型参数量: {n_params:,} ({n_params/1e6:.2f}M)")
    del test_model

    # --- 5折交叉验证 ---
    print(f"\n[2/3] {config['n_folds']}折交叉验证 (猪级别)...")
    print(f"  每折训练 {config['epochs']} epochs, batch={config['batch_size']}, "
          f"lr={config['lr']}, 每头猪{config['frames_per_pig']}帧")

    kf = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
    all_preds = np.zeros(n_pigs)

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_pigs)):
        print(f"\n  === Fold {fold+1}/{config['n_folds']}: "
              f"训练{len(train_idx)}头, 测试{len(test_idx)}头 ===")

        train_entries = [all_pigs[i] for i in train_idx]
        test_entries = [all_pigs[i] for i in test_idx]

        # 构建训练数据集
        train_ds = BackfatPointCloudDataset(
            train_entries,
            num_points=config['num_points'],
            frames_per_pig=config['frames_per_pig'],
            augment=True,
            seed=config['seed'] + fold
        )
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True
        )
        print(f"    训练样本数: {len(train_ds)}")

        # 初始化模型
        model = PointMLPRegressor(points=config['num_points'])

        t0 = time.time()
        fold_preds = train_one_fold(model, train_loader, test_entries, config, fold)
        elapsed = time.time() - t0

        for j, ti in enumerate(test_idx):
            all_preds[ti] = fold_preds[j]

        # 打印本折结果
        fold_true = backfats[test_idx]
        fold_pred = np.array(fold_preds)
        fold_mae = mean_absolute_error(fold_true, fold_pred)
        fold_rmse = np.sqrt(mean_squared_error(fold_true, fold_pred))
        fold_r2 = r2_score(fold_true, fold_pred) if len(fold_true) > 1 else 0

        print(f"\n    Fold {fold+1} 结果: MAE={fold_mae:.2f}mm, "
              f"RMSE={fold_rmse:.2f}mm, R²={fold_r2:.4f} ({elapsed:.0f}s)")
        print(f"    {'猪号':<30} {'真实':>6} {'预测':>6} {'误差':>6}")
        for j, ti in enumerate(test_idx):
            err = fold_preds[j] - backfats[ti]
            marker = " **" if abs(err) > 5 else ""
            print(f"    {pig_ids[ti]:<30} {backfats[ti]:>6.1f} {fold_preds[j]:>6.1f} "
                  f"{err:>+6.1f}{marker}")

    # --- 总结 ---
    print(f"\n{'=' * 70}")
    print(f"  总结: {config['n_folds']}折CV ({n_pigs}头猪)")
    print(f"{'=' * 70}")

    overall_mae = mean_absolute_error(backfats, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(backfats, all_preds))
    overall_r2 = r2_score(backfats, all_preds)

    print(f"  MAE  = {overall_mae:.2f} mm")
    print(f"  RMSE = {overall_rmse:.2f} mm")
    print(f"  R²   = {overall_r2:.4f}")

    # 误差分布
    errors = all_preds - backfats
    print(f"\n  误差分布:")
    print(f"    |误差| ≤ 2mm: {np.sum(np.abs(errors) <= 2)}/{n_pigs} "
          f"({np.sum(np.abs(errors) <= 2)/n_pigs*100:.1f}%)")
    print(f"    |误差| ≤ 3mm: {np.sum(np.abs(errors) <= 3)}/{n_pigs} "
          f"({np.sum(np.abs(errors) <= 3)/n_pigs*100:.1f}%)")
    print(f"    |误差| ≤ 5mm: {np.sum(np.abs(errors) <= 5)}/{n_pigs} "
          f"({np.sum(np.abs(errors) <= 5)/n_pigs*100:.1f}%)")

    # 保存最终模型 (用全部数据训练一次)
    print(f"\n[3/3] 用全部数据训练最终模型...")
    full_ds = BackfatPointCloudDataset(
        all_pigs,
        num_points=config['num_points'],
        frames_per_pig=config['frames_per_pig'],
        augment=True,
        seed=config['seed']
    )
    full_loader = DataLoader(
        full_ds, batch_size=config['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )

    final_model = PointMLPRegressor(points=config['num_points']).to(config['device'])
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        final_optimizer, T_max=config['epochs'], eta_min=config['lr'] * 0.01
    )
    criterion = nn.SmoothL1Loss()

    final_model.train()
    for epoch in range(config['epochs']):
        total_loss = 0
        n_samples = 0
        for pts, labels in full_loader:
            pts, labels = pts.to(config['device']), labels.to(config['device'])
            pred = final_model(pts)
            loss = criterion(pred, labels)
            final_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            final_optimizer.step()
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)
        final_scheduler.step()

        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: loss={total_loss/n_samples:.4f}")

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'best_pointmlp_elite_backfat.pth'
    )
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': config,
        'n_pigs': n_pigs,
        'backfat_range': [float(backfats.min()), float(backfats.max())],
        'cv_results': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'r2': overall_r2,
        }
    }, save_path)
    print(f"\n  模型已保存: {save_path}")
    print(f"  CV结果: MAE={overall_mae:.2f}mm, RMSE={overall_rmse:.2f}mm, R²={overall_r2:.4f}")


if __name__ == '__main__':
    main()
