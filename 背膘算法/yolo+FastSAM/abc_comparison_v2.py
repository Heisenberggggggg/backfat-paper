# -*- coding: utf-8 -*-
"""
ABC路线对比实验 V2 (改进版)
A: 掩膜深度图 + MobileNetV3-Small 回归
B-改进: 不归一化点云 + 全局元特征 + PointMLP 回归
C-改进: 不归一化点云特征 + 全局元特征 + XGBoost/SVR 回归

数据来自: abc_data/ 目录 (generate_abc_data.py 生成)
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.spatial import cKDTree
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
BACKFAT_LABELS = {
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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'abc_data')
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS_B = 80
EPOCHS_A = 60
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ============================================================
# 数据加载
# ============================================================
def load_all_data():
    """加载所有猪的数据"""
    pig_ids = []
    backfats = []
    all_depth = []   # 每头猪的深度图 (N, 224, 224)
    all_pc = []      # 每头猪的点云 (N, 1024, 3)
    all_meta = []    # 每头猪的全局特征 (N, 8)

    for pid in sorted(BACKFAT_LABELS.keys()):
        d_path = os.path.join(DATA_DIR, f'depth_{pid}.npy')
        p_path = os.path.join(DATA_DIR, f'pc_{pid}.npy')
        m_path = os.path.join(DATA_DIR, f'meta_{pid}.npy')

        if not all(os.path.exists(p) for p in [d_path, p_path, m_path]):
            print(f"  {pid}: 数据不完整，跳过")
            continue

        depth = np.load(d_path)
        pc = np.load(p_path)
        meta = np.load(m_path)

        pig_ids.append(pid)
        backfats.append(BACKFAT_LABELS[pid])
        all_depth.append(depth)
        all_pc.append(pc)
        all_meta.append(meta)

    return pig_ids, np.array(backfats), all_depth, all_pc, all_meta


# ============================================================
# 路线A: 深度图 + MobileNetV3-Small
# ============================================================
class DepthMapDataset(Dataset):
    def __init__(self, depth_list, label_list, augment=False):
        self.data = []
        self.labels = []
        for frames, label in zip(depth_list, label_list):
            for frame in frames:
                self.data.append(frame)
                self.labels.append(label)
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].copy()  # (224, 224)

        if self.augment:
            # 随机水平翻转
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
            # 随机深度抖动 (+/- 2%)
            img *= np.random.uniform(0.98, 1.02)

        # 归一化深度值到 [0, 1] 范围
        img_min, img_max = 0.3, 2.0  # 深度物理范围
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)

        # 单通道 → (1, H, W)
        img = img[np.newaxis, :, :]
        return torch.tensor(img), torch.tensor(self.labels[idx])


class DepthCNNRegressor(nn.Module):
    """轻量深度图CNN (类MobileNet风格)"""
    def __init__(self):
        super().__init__()

        def conv_bn(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )

        def dw_conv(in_c, out_c, stride=1):
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                # Pointwise
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn(1, 16, 2),       # 224→112
            dw_conv(16, 32, 2),      # 112→56
            dw_conv(32, 64, 2),      # 56→28
            dw_conv(64, 64, 1),      # 28→28
            dw_conv(64, 128, 2),     # 28→14
            dw_conv(128, 128, 1),    # 14→14
            dw_conv(128, 256, 2),    # 14→7
            nn.AdaptiveAvgPool2d(1), # 7→1
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


def train_depth_cnn_fold(train_depth, train_labels, test_depth, test_labels):
    train_ds = DepthMapDataset(train_depth, train_labels, augment=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

    model = DepthCNNRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_A)
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(EPOCHS_A):
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = model(imgs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    pig_preds = []
    with torch.no_grad():
        for frames in test_depth:
            # 归一化处理
            imgs = frames.copy()
            imgs = np.clip(imgs, 0.3, 2.0)
            imgs = (imgs - 0.3) / 1.7
            imgs = imgs[:, np.newaxis, :, :]  # (N, 1, 224, 224)
            imgs_t = torch.tensor(imgs, dtype=torch.float32).to(DEVICE)
            preds = []
            for i in range(0, len(imgs_t), BATCH_SIZE):
                batch = imgs_t[i:i + BATCH_SIZE]
                pred = model(batch)
                preds.append(pred.cpu().numpy())
            pig_preds.append(np.concatenate(preds).mean())

    return pig_preds


# ============================================================
# 路线B-改进: 不归一化点云 + 全局元特征 + PointMLP
# ============================================================
class ResMLPBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1), nn.BatchNorm1d(out_c), nn.GELU(),
            nn.Conv1d(out_c, out_c, 1), nn.BatchNorm1d(out_c),
        )
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class PointMLPWithMeta(nn.Module):
    """PointMLP + 全局元特征辅助输入"""
    def __init__(self, meta_dim=8):
        super().__init__()
        self.feat = nn.Sequential(
            ResMLPBlock(3, 64),
            ResMLPBlock(64, 128),
            ResMLPBlock(128, 256),
        )
        # 融合点云全局特征(256) + 元特征(meta_dim)
        self.head = nn.Sequential(
            nn.Linear(256 + meta_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, meta):
        # x: (B, N, 3), meta: (B, meta_dim)
        x = x.transpose(1, 2)
        x = self.feat(x)
        x = torch.max(x, dim=2)[0]  # (B, 256)
        x = torch.cat([x, meta], dim=1)  # (B, 256+meta_dim)
        x = self.head(x)
        return x.squeeze(-1)


class PCMetaDataset(Dataset):
    def __init__(self, pc_list, meta_list, label_list, augment=False):
        self.data_pc = []
        self.data_meta = []
        self.labels = []
        for frames_pc, frames_meta, label in zip(pc_list, meta_list, label_list):
            for pc, meta in zip(frames_pc, frames_meta):
                self.data_pc.append(pc)
                self.data_meta.append(meta)
                self.labels.append(label)
        self.data_pc = np.array(self.data_pc, dtype=np.float32)
        self.data_meta = np.array(self.data_meta, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data_pc[idx].copy()
        meta = self.data_meta[idx].copy()

        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ R.T
            pts += np.random.normal(0, 0.002, pts.shape).astype(np.float32)
            scale = np.random.uniform(0.97, 1.03)
            pts *= scale

        return (torch.tensor(pts), torch.tensor(meta),
                torch.tensor(self.labels[idx]))


def train_pointmlp_v2_fold(train_pc, train_meta, train_labels,
                           test_pc, test_meta, test_labels):
    # 标准化元特征
    all_train_meta = np.vstack([m for m in train_meta])
    meta_mean = all_train_meta.mean(axis=0)
    meta_std = all_train_meta.std(axis=0) + 1e-8

    # 标准化训练集元特征
    train_meta_normed = [((m - meta_mean) / meta_std).astype(np.float32)
                         for m in train_meta]
    test_meta_normed = [((m - meta_mean) / meta_std).astype(np.float32)
                        for m in test_meta]

    train_ds = PCMetaDataset(train_pc, train_meta_normed, train_labels, augment=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

    model = PointMLPWithMeta(meta_dim=8).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_B)
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(EPOCHS_B):
        for pts, meta, labels in train_dl:
            pts = pts.to(DEVICE)
            meta = meta.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(pts, meta)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    pig_preds = []
    with torch.no_grad():
        for frames_pc, frames_meta in zip(test_pc, test_meta_normed):
            pts_t = torch.tensor(frames_pc, dtype=torch.float32).to(DEVICE)
            meta_t = torch.tensor(frames_meta, dtype=torch.float32).to(DEVICE)
            preds = []
            for i in range(0, len(pts_t), BATCH_SIZE):
                p = model(pts_t[i:i+BATCH_SIZE], meta_t[i:i+BATCH_SIZE])
                preds.append(p.cpu().numpy())
            pig_preds.append(np.concatenate(preds).mean())

    return pig_preds


# ============================================================
# 路线C-改进: 不归一化特征 + 全局元特征
# ============================================================
def extract_features_from_pc(pc_frames, meta_frames):
    """从不归一化点云+元特征中提取特征，取帧平均"""
    all_feats = []
    for pts, meta in zip(pc_frames, meta_frames):
        feat = {}
        # 点云统计 (不归一化的，保留绝对尺寸)
        for i, axis in enumerate(['x', 'y', 'z']):
            feat[f'{axis}_std'] = np.std(pts[:, i])
            feat[f'{axis}_range'] = np.ptp(pts[:, i])
            feat[f'{axis}_skew'] = stats.skew(pts[:, i])
            feat[f'{axis}_kurtosis'] = stats.kurtosis(pts[:, i])

        # PCA
        cov = np.cov(pts.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        total = np.sum(eig) + 1e-8
        feat['pca1'] = eig[0] / total
        feat['pca2'] = eig[1] / total
        feat['pca3'] = eig[2] / total
        feat['linearity'] = (eig[0] - eig[1]) / (eig[0] + 1e-8)
        feat['planarity'] = (eig[1] - eig[2]) / (eig[0] + 1e-8)

        # 表面粗糙度
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=6)
        nn_dists = dists[:, 1:]
        feat['roughness_mean'] = np.mean(nn_dists)
        feat['roughness_std'] = np.std(nn_dists)

        # 全局元特征 (bbox, centroid_z, volume 等)
        for j, name in enumerate(['bbox_w', 'bbox_h', 'bbox_d', 'centroid_z',
                                   'n_raw', 'max_radius', 'area_ratio', 'volume']):
            feat[name] = meta[j]

        all_feats.append(feat)

    # 取帧平均
    keys = sorted(all_feats[0].keys())
    avg = {k: np.mean([f[k] for f in all_feats]) for k in keys}
    return avg


# ============================================================
# 评估辅助
# ============================================================
def eval_summary(name, y_true, y_pred, pig_ids):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n  [{name}] MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  R²={r2:.4f}")
    print(f"  {'耳号':<8} {'真实':>6} {'预测':>6} {'误差':>6}")
    for i in range(len(pig_ids)):
        err = y_pred[i] - y_true[i]
        print(f"  {pig_ids[i]:<8} {y_true[i]:>6.1f} {y_pred[i]:>6.1f} {err:>+6.1f}")
    return mae, rmse, r2


# ============================================================
# 主实验
# ============================================================
def main():
    print("=" * 70)
    print("  ABC路线对比实验 V2 (改进版)")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    # 加载数据
    print("\n[加载数据]")
    pig_ids, backfats, all_depth, all_pc, all_meta = load_all_data()
    n_pigs = len(pig_ids)
    print(f"  {n_pigs} 头猪, 背膘 {backfats.min():.1f}-{backfats.max():.1f}mm")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    # --------------------------------------------------
    # 路线C-改进: 形态学特征 + 元特征 + GBR/SVR
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("  路线C-改进: 不归一化特征 + 全局元特征 + GBR/SVR")
    print(f"{'='*70}")

    # 提取特征
    pig_features = []
    for i, pid in enumerate(pig_ids):
        feat = extract_features_from_pc(all_pc[i], all_meta[i])
        pig_features.append(feat)

    feat_keys = sorted(pig_features[0].keys())
    X_C = np.array([[f[k] for k in feat_keys] for f in pig_features])
    print(f"  特征维度: {X_C.shape[1]} (含全局元特征)")

    for model_name, model_fn in [
        ('C-GBR', lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)),
        ('C-SVR', lambda: SVR(kernel='rbf', C=10, gamma='scale')),
    ]:
        preds = np.zeros(n_pigs)
        for fold, (tr, te) in enumerate(kf.split(X_C)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_C[tr])
            X_te = scaler.transform(X_C[te])
            model = model_fn()
            model.fit(X_tr, backfats[tr])
            preds[te] = model.predict(X_te)
        results[model_name] = eval_summary(model_name, backfats, preds, pig_ids)

    # --------------------------------------------------
    # 路线B-改进: 不归一化点云 + 元特征 + PointMLP
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("  路线B-改进: 不归一化点云 + 全局元特征 + PointMLP")
    n_params = sum(p.numel() for p in PointMLPWithMeta().parameters())
    print(f"  参数量: {n_params:,}, 训练: {EPOCHS_B}轮")
    print(f"{'='*70}")

    preds_B = np.zeros(n_pigs)
    for fold, (tr, te) in enumerate(kf.split(backfats)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}: 训练{len(tr)}头, 测试{len(te)}头")
        t0 = time.time()
        fold_preds = train_pointmlp_v2_fold(
            [all_pc[i] for i in tr], [all_meta[i] for i in tr],
            [backfats[i] for i in tr],
            [all_pc[i] for i in te], [all_meta[i] for i in te],
            [backfats[i] for i in te],
        )
        for j, ti in enumerate(te):
            preds_B[ti] = fold_preds[j]
        fold_mae = mean_absolute_error(backfats[te], np.array(fold_preds))
        print(f"  Fold {fold+1} MAE={fold_mae:.2f}mm ({time.time()-t0:.0f}s)")

    results['B-PointMLP+Meta'] = eval_summary(
        'B-PointMLP+Meta', backfats, preds_B, pig_ids)

    # --------------------------------------------------
    # 路线A: 掩膜深度图 + 轻量CNN
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("  路线A: 掩膜深度图 + DepthCNN")
    n_params_a = sum(p.numel() for p in DepthCNNRegressor().parameters())
    print(f"  参数量: {n_params_a:,}, 训练: {EPOCHS_A}轮")
    print(f"{'='*70}")

    preds_A = np.zeros(n_pigs)
    for fold, (tr, te) in enumerate(kf.split(backfats)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}: 训练{len(tr)}头, 测试{len(te)}头")
        t0 = time.time()
        fold_preds = train_depth_cnn_fold(
            [all_depth[i] for i in tr], [backfats[i] for i in tr],
            [all_depth[i] for i in te], [backfats[i] for i in te],
        )
        for j, ti in enumerate(te):
            preds_A[ti] = fold_preds[j]
        fold_mae = mean_absolute_error(backfats[te], np.array(fold_preds))
        print(f"  Fold {fold+1} MAE={fold_mae:.2f}mm ({time.time()-t0:.0f}s)")

    results['A-DepthCNN'] = eval_summary(
        'A-DepthCNN', backfats, preds_A, pig_ids)

    # --------------------------------------------------
    # 最终汇总
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  最终对比汇总 (N={n_pigs}, {N_FOLDS}折CV)")
    print(f"{'='*70}")
    print(f"  {'路线':<25} {'MAE(mm)':>10} {'RMSE(mm)':>10} {'R²':>10}")
    print(f"  {'-'*55}")
    for name, (mae, rmse, r2) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"  {name:<25} {mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}")

    print(f"\n  背膘均值: {backfats.mean():.1f}mm, 标准差: {backfats.std():.1f}mm")
    print(f"  如果直接猜均值: MAE≈{np.mean(np.abs(backfats - backfats.mean())):.2f}mm")


if __name__ == '__main__':
    main()
