"""
深度图 CNN 背膘回归训练脚本
================================
使用 ResNet-18 对裁剪后的猪体深度图（256×256, float32, mm）进行背膘厚度回归。

与 PointMLP 路线的区别：
  - 输入: 2D 深度图（而非 3D 点云）
  - 模型: 预训练 ResNet-18（ImageNet权重，首层改为1通道）
  - 优势: CNN 能自动学习空间纹理特征（背部弧度、脊柱形状等），
          不依赖手工设计的统计特征

数据要求：
  - 先运行 generate_depthmap_dataset.py 生成 depthmap_dataset/
  - depthmap_dataset/metadata.csv 包含 (frame_path, pig_id, backfat, source)
  - 每帧为 .npy 文件 (256×256, float32, 单位 mm)

输出保存到 checkpoints_depthmap/，不覆盖任何已有结果。

Author: 耿来硕 + Claude
Date: 2026-04-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import numpy as np
import cv2
import os
import csv
import json
import random
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# 1. 数据集
# ============================================================

class DepthMapDataset(Dataset):
    """深度图背膘数据集。

    每个样本 = 一帧裁剪后的深度图 (256×256, mm) + 背膘真值。
    """
    def __init__(self, samples, augment=False, target_size=224):
        """
        Args:
            samples: list of (npy_path, backfat_value, pig_id)
            augment: 是否数据增强
            target_size: resize 到的尺寸（ResNet 默认 224）
        """
        self.samples = samples
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.samples)

    def _augment(self, depth):
        """数据增强（针对深度图）。

        - 随机水平翻转（猪体左右对称）
        - 随机旋转 ±15 度
        - 随机缩放 0.9-1.1
        - 深度值抖动 ±10mm（模拟深度噪声）
        - 随机平移 ±10% 画布
        """
        h, w = depth.shape

        # 1. 水平翻转（50%概率）
        if random.random() < 0.5:
            depth = np.fliplr(depth).copy()

        # 2. 随机仿射变换（旋转 + 缩放 + 平移）
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.1, 0.1) * w
        ty = random.uniform(-0.1, 0.1) * h

        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        depth = cv2.warpAffine(depth, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 3. 深度值抖动（仅对非零像素）
        noise = np.random.normal(0, 10, depth.shape).astype(np.float32)  # ±10mm
        mask = depth > 0
        depth[mask] += noise[mask]
        depth = np.clip(depth, 0, None)  # 深度不能为负

        return depth

    def __getitem__(self, idx):
        npy_path, backfat, pig_id = self.samples[idx]

        # 读取深度图
        try:
            depth = np.load(npy_path).astype(np.float32)  # (256, 256), mm
        except Exception:
            depth = np.zeros((256, 256), dtype=np.float32)

        # 数据增强
        if self.augment:
            depth = self._augment(depth)

        # 归一化：将 mm 值缩放到 [0, 1] 范围
        # 有效深度范围 300-3000mm，但实际猪体区域一般在 800-2000mm
        # 用固定范围归一化以保证一致性
        NORM_MIN = 300.0   # mm
        NORM_MAX = 2500.0  # mm
        depth = (depth - NORM_MIN) / (NORM_MAX - NORM_MIN)
        depth = np.clip(depth, 0, 1)

        # Resize 到 target_size (ResNet expects 224)
        if depth.shape[0] != self.target_size or depth.shape[1] != self.target_size:
            depth = cv2.resize(depth, (self.target_size, self.target_size),
                               interpolation=cv2.INTER_LINEAR)

        # 转为 tensor [1, H, W]（单通道）
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        label_tensor = torch.tensor([backfat], dtype=torch.float32)

        return depth_tensor, label_tensor


# ============================================================
# 2. 模型
# ============================================================

class DepthMapResNet(nn.Module):
    """基于 ResNet-18 的深度图回归模型。

    修改点：
    1. 首层 conv 从 3 通道改为 1 通道（深度图单通道）
    2. 分类头改为回归头（输出 1 维连续值）
    3. 添加 dropout 防过拟合
    """
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()

        # 加载预训练 ResNet-18
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # 修改首层：3通道 → 1通道
        # 用预训练权重的均值初始化（保留预训练的特征提取能力）
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # 将 3 通道权重平均到 1 通道
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # 复制 ResNet 的其余层
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 回归头
        feat_dim = resnet.fc.in_features  # 512 for ResNet-18
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Input: x [B, 1, 224, 224] — 单通道深度图
        Output: [B, 1] — 预测背膘厚度 (mm)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


# ============================================================
# 3. 训练与评估
# ============================================================

def compute_metrics(preds, targets):
    """计算回归指标。"""
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

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
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
    plt.savefig(os.path.join(save_dir, 'training_results_depthmap.png'), dpi=150)
    plt.close()


# ============================================================
# 4. 数据加载与划分
# ============================================================

def load_metadata(metadata_csv):
    """读取 metadata.csv，返回按 pig_id 分组的数据。"""
    pig_data = defaultdict(list)  # {pig_id: [(frame_path, backfat), ...]}

    with open(metadata_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pig_id = row['pig_id']
            frame_path = row['frame_path']
            backfat = float(row['backfat'])
            pig_data[pig_id].append((frame_path, backfat))

    return pig_data


def split_by_pig_id(pig_data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """按猪ID划分数据集。"""
    pig_ids = sorted(pig_data.keys())
    rng = random.Random(seed)
    rng.shuffle(pig_ids)

    n = len(pig_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = pig_ids[:n_train]
    val_ids = pig_ids[n_train:n_train + n_val]
    test_ids = pig_ids[n_train + n_val:]

    def make_samples(ids, base_dir):
        samples = []
        for pid in ids:
            for frame_path, bf in pig_data[pid]:
                full_path = os.path.join(base_dir, frame_path)
                samples.append((full_path, bf, pid))
        return samples

    return train_ids, val_ids, test_ids, make_samples


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser('DepthMap CNN Backfat Regression')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\depthmap_dataset')
    parser.add_argument('--save_dir', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\网络\depthmap_cnn\checkpoints_depthmap')
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

    # 日志
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'training_log_depthmap.txt')

    def log(msg):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('ascii', errors='replace').decode())
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("DepthMap CNN (ResNet-18) Backfat Regression")
    log(f"Device: {device}")
    log("=" * 60)

    # --- 加载数据 ---
    metadata_csv = os.path.join(args.data_dir, 'metadata.csv')
    if not os.path.exists(metadata_csv):
        log(f"ERROR: {metadata_csv} not found! Run generate_depthmap_dataset.py first.")
        return

    log("\n[1/4] Loading metadata...")
    pig_data = load_metadata(metadata_csv)
    total_pigs = len(pig_data)
    total_frames = sum(len(v) for v in pig_data.values())
    log(f"  Total: {total_pigs} pigs, {total_frames} frames")

    # 背膘分布
    all_backfats = [bf for frames in pig_data.values() for _, bf in frames]
    log(f"  Backfat range: {min(all_backfats):.1f} - {max(all_backfats):.1f} mm "
        f"(mean={np.mean(all_backfats):.1f})")

    # --- 划分数据集 ---
    log("\n[2/4] Splitting by pig ID (80/10/10)...")
    train_ids, val_ids, test_ids, make_samples = split_by_pig_id(
        pig_data, seed=args.seed
    )

    train_samples = make_samples(train_ids, args.data_dir)
    val_samples = make_samples(val_ids, args.data_dir)
    test_samples = make_samples(test_ids, args.data_dir)

    log(f"  Train: {len(train_ids)} pigs, {len(train_samples)} frames")
    log(f"  Val:   {len(val_ids)} pigs, {len(val_samples)} frames")
    log(f"  Test:  {len(test_ids)} pigs, {len(test_samples)} frames")

    # 保存划分
    split_path = os.path.join(args.save_dir, 'split_info_depthmap.json')
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_pigs': train_ids, 'val_pigs': val_ids, 'test_pigs': test_ids,
            'n_train': len(train_samples), 'n_val': len(val_samples),
            'n_test': len(test_samples), 'seed': args.seed
        }, f, indent=2, ensure_ascii=False)

    # --- 创建 DataLoader ---
    log("\n[3/4] Creating data loaders...")

    train_dataset = DepthMapDataset(train_samples, augment=True)
    val_dataset = DepthMapDataset(val_samples, augment=False)
    test_dataset = DepthMapDataset(test_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # --- 初始化模型 ---
    log("\n[4/4] Initializing ResNet-18 (1-channel depth)...")
    model = DepthMapResNet(pretrained=True, dropout=args.dropout).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Total parameters:     {total_params:,}")
    log(f"  Trainable parameters: {trainable_params:,}")

    # SmoothL1Loss（比 MSE 对异常值更鲁棒）
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    log(f"\n  Hyperparams: lr={args.lr}, weight_decay={args.weight_decay}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, dropout={args.dropout}")
    log(f"  Loss: SmoothL1Loss")

    # --- 训练循环 ---
    log(f"\nTraining for {args.epochs} epochs...")
    log("=" * 60)

    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses, val_losses = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs
        )
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

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, os.path.join(args.save_dir, 'best_model_depthmap.pth'))
            log(f"  * Best model saved! (Val Loss: {val_loss:.4f}, R2: {val_metrics['R2']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        # 定期保存 checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))

    # --- 测试 ---
    log("\n" + "=" * 60)
    log("Testing with best model...")

    ckpt = torch.load(os.path.join(args.save_dir, 'best_model_depthmap.pth'),
                       map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    log(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_loss, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )

    log(f"\n  Test Results:")
    log(f"    Loss: {test_loss:.4f}")
    log(f"    MAE:  {test_metrics['MAE']:.2f} mm")
    log(f"    RMSE: {test_metrics['RMSE']:.2f} mm")
    log(f"    R2:   {test_metrics['R2']:.4f}")
    log(f"    MAPE: {test_metrics['MAPE']:.1f}%")

    # --- 绘图 ---
    plot_results(train_losses, val_losses, train_r2s, val_r2s,
                 test_preds, test_targets, args.save_dir)
    log(f"  Plot saved to: {os.path.join(args.save_dir, 'training_results_depthmap.png')}")

    # --- 对比总结 ---
    log("\n" + "=" * 60)
    log("Results Comparison:")
    log(f"  {'Method':<35} {'Test R2':>8}  {'Test MAE':>9}  {'Test RMSE':>10}")
    log(f"  {'-'*35} {'-'*8}  {'-'*9}  {'-'*10}")
    log(f"  {'PointMLP v4 (3D point cloud)':<35} {'~0.10':>8}  {'~3.0mm':>9}  {'~3.6mm':>10}")
    log(f"  {'XGBoost (41-dim handcraft)':<35} {'0.04':>8}  {'2.93mm':>9}  {'3.61mm':>10}")
    log(f"  {'DepthMap CNN (ResNet-18)':<35} {test_metrics['R2']:>8.4f}  "
        f"{test_metrics['MAE']:>8.2f}mm  {test_metrics['RMSE']:>9.2f}mm")
    log(f"  {'Sister: CNN-ViT (RGB only)':<35} {'0.74':>8}  {'0.84mm':>9}  {'1.05mm':>10}")
    log("=" * 60)

    # 保存测试结果
    results = {
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'best_epoch': int(ckpt['epoch']),
        'total_pigs': total_pigs,
        'train_pigs': len(train_ids),
        'val_pigs': len(val_ids),
        'test_pigs': len(test_ids),
        'hyperparams': {
            'lr': args.lr, 'weight_decay': args.weight_decay,
            'batch_size': args.batch_size, 'epochs': args.epochs,
            'dropout': args.dropout, 'loss': 'SmoothL1Loss',
            'model': 'ResNet-18 (1-channel)',
        }
    }
    with open(os.path.join(args.save_dir, 'results_depthmap.json'), 'w') as f:
        json.dump(results, f, indent=2)

    log("\nDone!")


if __name__ == '__main__':
    main()
