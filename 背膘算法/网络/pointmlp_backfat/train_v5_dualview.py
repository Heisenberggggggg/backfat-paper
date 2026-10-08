"""
双分支 PointMLP 中期融合背膘回归训练脚本 (v5)
================================================
两个独立的 PointMLP-Elite 分支分别处理背部和臀部点云，
在全局特征层拼接后送入共享回归头。

架构:
  背部点云 [B,3,1024] → PointMLP-Branch1 → 128-dim ─┐
                                                       ├→ concat(256) → MLP(256→64→32→1) → 背膘
  臀部点云 [B,3,1024] → PointMLP-Branch2 → 128-dim ─┘

数据:
  背部: processed_pointclouds_v2/ (master 目录)
  臀部: processed_pointclouds_v3_rear/ (sub/sub2/sub10 目录)
  同一头猪同一帧索引配对

与 v4 的区别:
  - 双分支中期融合（而非单分支+手工特征）
  - 两个分支独立权重（背部和臀部形态不同）
  - 拼接后的特征维度 128+128=256

输出保存到 checkpoints_v5_dualview/，不覆盖之前任何结果。

Author: 耿来硕 + Claude
Date: 2026-04-02
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
import random
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用基础函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (square_distance, farthest_point_sample,
                   index_points, knn_point, get_activation, compute_metrics)


# ============================================================
# 1. 标签加载
# ============================================================

def load_labels_from_docx_nky1(docx_path):
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 9:
                ear_tag, bf_str = cells[1], cells[8]
                if ear_tag and bf_str and '-' in ear_tag:
                    try: labels[ear_tag] = float(bf_str)
                    except ValueError: pass
    return labels

def load_labels_from_docx_nky2(docx_path):
    from docx import Document
    doc = Document(docx_path)
    labels = {}
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 13:
                ear_tag = cells[1]
                bf_str = cells[11] if cells[11] else (cells[12] if len(cells) > 12 else '')
                if ear_tag and bf_str and '-' in ear_tag:
                    try: labels[ear_tag] = float(bf_str)
                    except ValueError: pass
    return labels

def load_labels_from_docx_nky3(docx_path):
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
                    dong_num = d; break
            if dong_num and len(cells) >= 6:
                ear_short, bf_str = cells[1], cells[5]
                if ear_short and bf_str and cells[0].isdigit():
                    try: labels[f'{dong_num}-{ear_short}'] = float(bf_str)
                    except ValueError: pass
    return labels

def load_labels_from_xlsx_fmh(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb['Sheet1']
    entries = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        cells = [str(c).strip() if c else '' for c in row[:3]]
        if cells[0] and cells[1] and cells[2]:
            try: entries.append((int(cells[0]), cells[1], float(cells[2])))
            except ValueError: pass
    return entries

def load_labels_from_xlsx_fmh_nov(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    entries = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        cells = [str(c).strip() if c else '' for c in row[:3]]
        if cells[0] and cells[1] and cells[2]:
            try: entries.append((int(cells[0]), cells[1], float(cells[2])))
            except ValueError: pass
    return entries


# ============================================================
# 2. 双视角数据注册表
# ============================================================

def build_dualview_registry(dk_base, fmh_base):
    """扫描所有猪，找到同时有背部和臀部点云的猪。
    返回 {pig_uid: {'back_dir': str, 'rear_dir': str, 'backfat': float}}
    """
    registry = {}
    BF_MIN, BF_MAX = 5.0, 40.0

    # === nkyfirst ===
    labels1 = load_labels_from_docx_nky1(os.path.join(dk_base, '第一次数据7.29.docx'))
    nky1_dir = os.path.join(dk_base, 'nkyfirst')
    if os.path.isdir(nky1_dir):
        for pig_id in os.listdir(nky1_dir):
            if pig_id not in labels1: continue
            bf = labels1[pig_id]
            if not (BF_MIN <= bf <= BF_MAX): continue
            back = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
            if not os.path.isdir(back): continue
            rear = None
            for subdir in ['sub', 'sub2']:
                rp = os.path.join(nky1_dir, pig_id, 'depth_cz_r_d_p', subdir, 'processed_pointclouds_v3_rear')
                if os.path.isdir(rp) and any(f.endswith('.txt') for f in os.listdir(rp)):
                    rear = rp; break
            if rear and any(f.endswith('.txt') for f in os.listdir(back)):
                registry[f'nky1_{pig_id}'] = {'back_dir': back, 'rear_dir': rear, 'backfat': bf}

    # === nkysecond ===
    labels2 = load_labels_from_docx_nky2(os.path.join(dk_base, '第二次数据9.6.docx'))
    nky2_dir = os.path.join(dk_base, 'nkysecond')
    if os.path.isdir(nky2_dir):
        for pig_id in os.listdir(nky2_dir):
            if pig_id not in labels2: continue
            bf = labels2[pig_id]
            if not (BF_MIN <= bf <= BF_MAX): continue
            if pig_id.startswith('6-'):
                master, sub_name = 'master10', 'sub10'
            elif pig_id.startswith('7-'):
                master, sub_name = 'master', 'sub'
            else:
                continue
            back = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', master, 'processed_pointclouds_v2')
            rear = os.path.join(nky2_dir, pig_id, 'depth_cz_r_d_p', sub_name, 'processed_pointclouds_v3_rear')
            if (os.path.isdir(back) and os.path.isdir(rear) and
                any(f.endswith('.txt') for f in os.listdir(back)) and
                any(f.endswith('.txt') for f in os.listdir(rear))):
                registry[f'nky2_{pig_id}'] = {'back_dir': back, 'rear_dir': rear, 'backfat': bf}

    # === nkythird ===
    labels3 = load_labels_from_docx_nky3(os.path.join(dk_base, '第三次数据12.9.docx'))
    nky3_dir = os.path.join(dk_base, 'nkythird')
    if os.path.isdir(nky3_dir):
        for kinect_dir in os.listdir(nky3_dir):
            kinect_path = os.path.join(nky3_dir, kinect_dir)
            if not os.path.isdir(kinect_path): continue
            for pig_dir in os.listdir(kinect_path):
                if pig_dir.startswith('w') or pig_dir.startswith('wc'): continue
                pig_path = os.path.join(kinect_path, pig_dir)
                if not os.path.isdir(pig_path): continue
                m = re.match(r'^[cwk]*(\d+)-(.+)$', pig_dir)
                if not m: continue
                label_key = f'{m.group(1)}-{m.group(2)}'
                if label_key not in labels3: continue
                bf = labels3[label_key]
                if not (BF_MIN <= bf <= BF_MAX): continue
                back = os.path.join(pig_path, 'depth_cz_r_d_p', 'master', 'processed_pointclouds_v2')
                rear = os.path.join(pig_path, 'depth_cz_r_d_p', 'sub', 'processed_pointclouds_v3_rear')
                if (os.path.isdir(back) and os.path.isdir(rear) and
                    any(f.endswith('.txt') for f in os.listdir(back)) and
                    any(f.endswith('.txt') for f in os.listdir(rear))):
                    registry[f'nky3_{pig_dir}'] = {'back_dir': back, 'rear_dir': rear, 'backfat': bf}

    # === 丰美禾 ===
    fmh_dates = ['8.24','8.25','8.26','8.27','10.29','10.30','10.31',
                 '11.1','11.2','11.3','11.4','11.5']
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    fmh_entries = load_labels_from_xlsx_fmh(xlsx_path) if os.path.isfile(xlsx_path) else []
    xlsx_nov = os.path.join(fmh_base, '11.10处理背膘数据.xlsx')
    fmh_nov_entries = load_labels_from_xlsx_fmh_nov(xlsx_nov) if os.path.isfile(xlsx_nov) else []
    all_entries = fmh_entries + fmh_nov_entries

    for date_str in fmh_dates:
        date_dir = os.path.join(fmh_base, date_str)
        if not os.path.isdir(date_dir): continue
        for pig_dirname in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dirname)
            if not os.path.isdir(pig_path): continue
            matched_bf = None
            for seq, tag, bf in all_entries:
                if tag.strip() and tag.strip() in pig_dirname:
                    matched_bf = bf; break
            if matched_bf is None or not (BF_MIN <= matched_bf <= BF_MAX): continue
            back = os.path.join(pig_path, 'master', 'processed_pointclouds_v2')
            rear = os.path.join(pig_path, 'sub', 'processed_pointclouds_v3_rear')
            if (os.path.isdir(back) and os.path.isdir(rear) and
                any(f.endswith('.txt') for f in os.listdir(back)) and
                any(f.endswith('.txt') for f in os.listdir(rear))):
                registry[f'fmh_{date_str}_{pig_dirname}'] = {
                    'back_dir': back, 'rear_dir': rear, 'backfat': matched_bf
                }

    return registry


# ============================================================
# 3. PointMLP-Elite 骨干（单分支，不含回归头）
# ============================================================

class LocalGrouper(nn.Module):
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
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1,1,1, channel + add_channel]))

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
            get_activation(activation))
    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0,
                 bias=True, activation='relu'):
        super().__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)), self.act)
        self.net2 = nn.Sequential(
            nn.Conv1d(int(channel * res_expansion), channel, kernel_size, bias=bias),
            nn.BatchNorm1d(channel))
    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1,
                 res_expansion=1, bias=True, activation='relu', use_xyz=True):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = [ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                     bias=bias, activation=activation) for _ in range(blocks)]
        self.operation = nn.Sequential(*operation)
    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0,1,3,2).reshape(-1, d, s)
        x = self.transfer(x)
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(-1, x.size(1))
        return x.reshape(b, n, -1).permute(0,2,1)


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1,
                 bias=True, activation='relu'):
        super().__init__()
        operation = [ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion,
                                     bias=bias, activation=activation) for _ in range(blocks)]
        self.operation = nn.Sequential(*operation)
    def forward(self, x):
        return self.operation(x)


class PointMLPBackbone(nn.Module):
    """PointMLP-Elite 骨干网络（只输出全局特征，不含回归头）。"""
    def __init__(self, points=1024, embed_dim=16, groups=1, res_expansion=0.25,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2,2,2,1], pre_blocks=[1,1,2,1], pos_blocks=[1,1,2,1],
                 k_neighbors=[24,24,24,24], reducers=[2,2,2,2], dropout=0.5):
        super().__init__()
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = points

        for i in range(self.stages):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = anchor_points // reducers[i]
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize))
            self.pre_blocks_list.append(
                PreExtraction(last_channel, out_channel, pre_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation,
                              use_xyz=use_xyz))
            self.pos_blocks_list.append(
                PosExtraction(out_channel, pos_blocks[i], groups=groups,
                              res_expansion=res_expansion, bias=bias, activation=activation))
            last_channel = out_channel

        self.out_dim = last_channel  # 128 for embed_dim=16
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, 3, N] -> [B, out_dim]"""
        xyz = x.permute(0, 2, 1)
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0,2,1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        feat = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return self.drop(feat)


# ============================================================
# 4. 双分支中期融合模型
# ============================================================

class DualBranchPointMLP(nn.Module):
    """双分支 PointMLP-Elite 中期融合回归模型。

    两个独立的 PointMLP 骨干分别提取背部和臀部特征，
    在全局特征层拼接后通过共享 MLP 回归头预测背膘。
    """
    def __init__(self, points=1024, embed_dim=16, dropout=0.5):
        super().__init__()
        self.back_branch = PointMLPBackbone(points=points, embed_dim=embed_dim, dropout=dropout)
        self.rear_branch = PointMLPBackbone(points=points, embed_dim=embed_dim, dropout=dropout)

        feat_dim = self.back_branch.out_dim  # 128
        combined_dim = feat_dim * 2  # 256

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, back_pc, rear_pc):
        """
        back_pc: [B, 3, N] 背部点云
        rear_pc: [B, 3, N] 臀部点云
        Returns: [B, 1] 预测背膘
        """
        back_feat = self.back_branch(back_pc)  # [B, 128]
        rear_feat = self.rear_branch(rear_pc)  # [B, 128]
        combined = torch.cat([back_feat, rear_feat], dim=1)  # [B, 256]
        return self.regressor(combined)


# ============================================================
# 5. 双视角数据集
# ============================================================

class DualViewDataset(Dataset):
    """双视角点云数据集。每个样本返回 (back_pc, rear_pc, label)。"""

    def __init__(self, samples, num_points=1024, augment=False):
        """
        samples: list of (back_file, rear_file, backfat, pig_id)
        """
        self.samples = samples
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _load_and_process(self, filepath):
        try:
            pc = np.loadtxt(filepath, dtype=np.float32)
        except Exception:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)
        if pc.ndim != 2 or pc.shape[1] < 3:
            pc = np.zeros((self.num_points, 3), dtype=np.float32)
        pc = pc[:, :3]

        if self.augment:
            pc = self._augment(pc)

        n = len(pc)
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
            pc = pc[idx]
        elif n < self.num_points:
            idx = np.random.choice(n, self.num_points, replace=True)
            pc = pc[idx]

        centroid = np.mean(pc, axis=0)
        pc = pc - centroid

        return torch.from_numpy(pc.T).float()  # [3, N]

    def _augment(self, pc):
        # Z轴随机旋转
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t,-sin_t,0],[sin_t,cos_t,0],[0,0,1]], dtype=np.float32)
        pc = pc @ R.T
        # 随机平移
        ranges = np.ptp(pc, axis=0)
        shift = np.random.uniform(-0.05, 0.05, size=3).astype(np.float32) * ranges
        pc = pc + shift
        # 随机缩放
        scale = np.random.uniform(0.85, 1.15)
        pc *= scale
        # 随机抖动
        pc += np.random.normal(0, 0.01, size=pc.shape).astype(np.float32)
        # X轴镜像
        if np.random.random() < 0.5:
            pc[:, 0] = -pc[:, 0]
        # 随机丢点
        if np.random.random() < 0.5 and len(pc) > self.num_points:
            keep = max(int(len(pc) * np.random.uniform(0.75, 0.95)), self.num_points)
            idx = np.random.choice(len(pc), min(keep, len(pc)), replace=False)
            pc = pc[idx]
        return pc

    def __getitem__(self, idx):
        back_file, rear_file, backfat, pid = self.samples[idx]
        back_pc = self._load_and_process(back_file)
        rear_pc = self._load_and_process(rear_file)
        label = torch.tensor([backfat], dtype=torch.float32)
        return back_pc, rear_pc, label


# ============================================================
# 6. 训练与评估
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    for batch_idx, (back, rear, labels) in enumerate(loader):
        back, rear, labels = back.to(device), rear.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(back, rear)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy().flatten())
        all_targets.extend(labels.detach().cpu().numpy().flatten())
    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for back, rear, labels in loader:
        back, rear, labels = back.to(device), rear.to(device), labels.to(device)
        outputs = model(back, rear)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(labels.cpu().numpy().flatten())
    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics, all_preds, all_targets


def plot_results(train_losses, val_losses, train_r2s, val_r2s,
                 test_preds, test_targets, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(train_losses, label='Train Loss', color='blue')
    axes[0].plot(val_losses, label='Val Loss', color='red')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(train_r2s, label='Train R2', color='blue')
    axes[1].plot(val_r2s, label='Val R2', color='red')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('R2')
    axes[1].set_title('Training & Validation R2'); axes[1].legend(); axes[1].grid(True)
    tp, tt = np.array(test_preds), np.array(test_targets)
    axes[2].scatter(tt, tp, alpha=0.3, s=10, color='blue')
    vmin, vmax = min(tt.min(), tp.min())-1, max(tt.max(), tp.max())+1
    axes[2].plot([vmin,vmax],[vmin,vmax],'r--',label='y=x')
    axes[2].set_xlabel('True Backfat (mm)'); axes[2].set_ylabel('Predicted Backfat (mm)')
    m = compute_metrics(tp, tt)
    axes[2].set_title(f'Test: R2={m["R2"]:.3f}, MAE={m["MAE"]:.2f}mm')
    axes[2].legend(); axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results_v5.png'), dpi=150)
    plt.close()


# ============================================================
# 7. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser('Dual-Branch PointMLP v5')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dk_base', type=str, default=r'G:\耿来硕数据\DK')
    parser.add_argument('--fmh_base', type=str, default=r'G:\耿来硕数据\丰美禾数据')
    parser.add_argument('--save_dir', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\网络\pointmlp_backfat\checkpoints_v5_dualview')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.save_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'training_log_v5.txt')

    def log(msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        try: print(line)
        except UnicodeEncodeError: print(line.encode('ascii',errors='replace').decode())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("Dual-Branch PointMLP-Elite v5 (Mid-Fusion) Backfat Regression")
    log("Back branch + Rear branch -> concat -> regressor")
    log(f"Device: {device}")
    log("=" * 60)

    # --- 构建双视角注册表 ---
    log("\n[1/5] Building dual-view registry...")
    registry = build_dualview_registry(args.dk_base, args.fmh_base)
    log(f"  Found {len(registry)} pigs with both back + rear point clouds")

    if len(registry) == 0:
        log("ERROR: No paired data found!")
        return

    # --- 划分 ---
    log("\n[2/5] Splitting by pig ID (80/10/10)...")
    all_ids = sorted(registry.keys())
    random.Random(args.seed).shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    # 构建样本：配对同一头猪的背部和臀部帧
    def make_samples(ids):
        samples = []
        for pid in ids:
            info = registry[pid]
            back_files = sorted([f for f in os.listdir(info['back_dir']) if f.endswith('.txt')])
            rear_files = sorted([f for f in os.listdir(info['rear_dir']) if f.endswith('.txt')])
            # 取两者的最小帧数，按索引配对
            n_frames = min(len(back_files), len(rear_files))
            for i in range(n_frames):
                bf = os.path.join(info['back_dir'], back_files[i])
                rf = os.path.join(info['rear_dir'], rear_files[i])
                samples.append((bf, rf, info['backfat'], pid))
        return samples

    train_samples = make_samples(train_ids)
    val_samples = make_samples(val_ids)
    test_samples = make_samples(test_ids)

    log(f"  Train: {len(train_ids)} pigs, {len(train_samples)} frame-pairs")
    log(f"  Val:   {len(val_ids)} pigs, {len(val_samples)} frame-pairs")
    log(f"  Test:  {len(test_ids)} pigs, {len(test_samples)} frame-pairs")

    # 保存划分
    split_path = os.path.join(args.save_dir, 'split_info_v5.json')
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_pigs': train_ids, 'val_pigs': val_ids, 'test_pigs': test_ids,
            'n_train': len(train_samples), 'n_val': len(val_samples),
            'n_test': len(test_samples), 'seed': args.seed
        }, f, indent=2, ensure_ascii=False)

    train_labels = [s[2] for s in train_samples]
    if train_labels:
        log(f"  Backfat range: {min(train_labels):.1f}-{max(train_labels):.1f}mm "
            f"(mean={np.mean(train_labels):.1f})")

    # --- DataLoader ---
    log("\n[3/5] Creating data loaders...")
    train_dataset = DualViewDataset(train_samples, args.num_points, augment=True)
    val_dataset = DualViewDataset(val_samples, args.num_points, augment=False)
    test_dataset = DualViewDataset(test_samples, args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- 模型 ---
    log("\n[4/5] Initializing Dual-Branch PointMLP-Elite...")
    model = DualBranchPointMLP(
        points=args.num_points, embed_dim=args.embed_dim, dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total parameters:     {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")
    log(f"  Back branch: {sum(p.numel() for p in model.back_branch.parameters()):,}")
    log(f"  Rear branch: {sum(p.numel() for p in model.rear_branch.parameters()):,}")
    log(f"  Regressor:   {sum(p.numel() for p in model.regressor.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    log(f"\n  Hyperparams: lr={args.lr}, wd={args.weight_decay}, "
        f"bs={args.batch_size}, epochs={args.epochs}, dropout={args.dropout}")

    # --- 训练 ---
    log(f"\n[5/5] Training for {args.epochs} epochs...")
    log("=" * 60)

    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    train_losses, val_losses, train_r2s, val_r2s = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
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
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best_model_v5.pth'))
            log(f"  * Best model saved! (Val Loss: {val_loss:.4f}, R2: {val_metrics['R2']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if epoch % 20 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                       os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    # --- 测试 ---
    log("\n" + "=" * 60)
    log("Testing with best model...")
    ckpt = torch.load(os.path.join(args.save_dir, 'best_model_v5.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    log(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_loss, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device)

    log(f"\n  Test Results:")
    log(f"    Loss: {test_loss:.4f}")
    log(f"    MAE:  {test_metrics['MAE']:.2f} mm")
    log(f"    RMSE: {test_metrics['RMSE']:.2f} mm")
    log(f"    R2:   {test_metrics['R2']:.4f}")
    log(f"    MAPE: {test_metrics['MAPE']:.1f}%")

    plot_results(train_losses, val_losses, train_r2s, val_r2s,
                 test_preds, test_targets, args.save_dir)

    log("\n" + "=" * 60)
    log("Results Comparison:")
    log(f"  {'Method':<45} {'Test R2':>8}  {'Test MAE':>9}  {'Test RMSE':>10}")
    log(f"  {'-'*45} {'-'*8}  {'-'*9}  {'-'*10}")
    log(f"  {'XGBoost (41-dim handcraft)':<45} {'0.04':>8}  {'2.93mm':>9}  {'3.61mm':>10}")
    log(f"  {'Projection CNN (BEV+Front+Side)':<45} {'0.09':>8}  {'3.22mm':>9}  {'4.01mm':>10}")
    log(f"  {'PointMLP v4 (back only)':<45} {'~0.10':>8}  {'~3.0mm':>9}  {'~3.6mm':>10}")
    log(f"  {'DepthMap CNN (raw depth, 1ch)':<45} {'0.21':>8}  {'2.99mm':>9}  {'3.72mm':>10}")
    log(f"  {'Dual-Branch PointMLP v5 (back+rear)':<45} {test_metrics['R2']:>8.4f}  "
        f"{test_metrics['MAE']:>8.2f}mm  {test_metrics['RMSE']:>9.2f}mm")
    log(f"  {'Sister: CNN-ViT (RGB)':<45} {'0.74':>8}  {'0.84mm':>9}  {'1.05mm':>10}")
    log("=" * 60)

    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_epoch': int(ckpt['epoch']),
        'total_paired_pigs': len(registry),
        'hyperparams': {'lr': args.lr, 'wd': args.weight_decay, 'bs': args.batch_size,
                        'epochs': args.epochs, 'dropout': args.dropout, 'embed_dim': args.embed_dim}
    }
    with open(os.path.join(args.save_dir, 'results_v5.json'), 'w') as f:
        json.dump(results, f, indent=2)

    log("\nDone!")


if __name__ == '__main__':
    main()
