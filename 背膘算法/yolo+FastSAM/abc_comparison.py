# -*- coding: utf-8 -*-
"""
ABC三路线对比实验
A: 深度图+轻量CNN（需掩膜深度图，后续单独做）
B: 点云+PointMLP回归
C: 形态学特征+XGBoost/SVR回归

评估方式：猪级别5折交叉验证
指标：MAE(mm), RMSE(mm), R²
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

BASE_DIR = r'G:\耿来硕数据\DK\nkyfirst'
NUM_POINTS = 1024          # 点云下采样目标点数
FRAMES_PER_PIG = 30        # 每头猪随机采样帧数（加速训练）
N_FOLDS = 5                # 交叉验证折数
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 数据加载
# ============================================================

def get_pig_dir(pig_id):
    """获取猪只的processed_pointclouds目录"""
    # 处理 c2-17 -> 2-17 的映射
    candidates = [pig_id, f'c{pig_id}']
    for name in candidates:
        d = os.path.join(BASE_DIR, name, 'depth_cz_r_d_p', 'master', 'processed_pointclouds')
        if os.path.isdir(d):
            return d
    return None


def fps_downsample(points, n_target):
    """最远点采样(FPS)下采样到n_target个点"""
    if len(points) <= n_target:
        # 如果点数不足，随机重复填充
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


def load_pig_data(pig_id, n_frames=None, n_points=NUM_POINTS):
    """加载一头猪的点云数据，返回 (N_frames, n_points, 3)"""
    pcd_dir = get_pig_dir(pig_id)
    if pcd_dir is None:
        return None

    txt_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.txt')])
    if not txt_files:
        return None

    # 随机采样帧
    if n_frames and len(txt_files) > n_frames:
        txt_files = list(np.random.choice(txt_files, n_frames, replace=False))

    frames = []
    for f in txt_files:
        pts = np.loadtxt(os.path.join(pcd_dir, f), dtype=np.float32)
        if len(pts) < 100:
            continue
        pts = fps_downsample(pts, n_points)
        frames.append(pts)

    if not frames:
        return None
    return np.array(frames)  # (N_frames, n_points, 3)


# ============================================================
# 路线B：PointMLP 回归模型
# ============================================================

class ResMLPBlock(nn.Module):
    """残差MLP块"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, 1),
            nn.BatchNorm1d(out_c),
        )
        self.shortcut = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class PointMLPRegressor(nn.Module):
    """轻量级PointMLP回归模型"""
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            ResMLPBlock(3, 64),
            ResMLPBlock(64, 128),
            ResMLPBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)       # (B, 3, N)
        x = self.feat(x)            # (B, 256, N)
        x = torch.max(x, dim=2)[0]  # (B, 256) global max pooling
        x = self.head(x)            # (B, 1)
        return x.squeeze(-1)


class PointCloudDataset(Dataset):
    def __init__(self, frames_list, labels_list, augment=False):
        """
        frames_list: list of (N_frames, n_points, 3) arrays
        labels_list: list of float backfat values (对应每头猪)
        """
        self.data = []
        self.labels = []
        for frames, label in zip(frames_list, labels_list):
            for frame in frames:
                self.data.append(frame)
                self.labels.append(label)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()  # (N, 3)
        if self.augment:
            # 随机旋转 (绕Y轴，因为俯拍Y大致向下)
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ R.T
            # 随机抖动
            pts += np.random.normal(0, 0.005, pts.shape).astype(np.float32)
            # 随机缩放
            scale = np.random.uniform(0.95, 1.05)
            pts *= scale
        return torch.tensor(pts), torch.tensor(self.labels[idx])


def train_pointmlp_fold(train_frames, train_labels, test_frames, test_labels):
    """训练一折PointMLP，返回测试集每头猪的平均预测"""
    train_ds = PointCloudDataset(train_frames, train_labels, augment=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

    model = PointMLPRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.SmoothL1Loss()

    # 训练
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for pts, labels in train_dl:
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            pred = model(pts)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        scheduler.step()

    # 测试：对每头测试猪，预测所有帧取平均
    model.eval()
    pig_preds = []
    with torch.no_grad():
        for frames in test_frames:
            pts_tensor = torch.tensor(frames, dtype=torch.float32).to(DEVICE)
            # 分批预测避免显存不足
            preds = []
            for i in range(0, len(pts_tensor), BATCH_SIZE):
                batch = pts_tensor[i:i + BATCH_SIZE]
                pred = model(batch)
                preds.append(pred.cpu().numpy())
            pig_pred = np.concatenate(preds).mean()
            pig_preds.append(pig_pred)

    return pig_preds


# ============================================================
# 路线C：形态学特征提取
# ============================================================

def extract_features_single(points):
    """从单帧点云提取统计/形态学特征"""
    feat = {}

    # 基本统计
    feat['n_pts'] = len(points)
    for i, axis in enumerate(['x', 'y', 'z']):
        feat[f'{axis}_mean'] = np.mean(points[:, i])
        feat[f'{axis}_std'] = np.std(points[:, i])
        feat[f'{axis}_range'] = np.ptp(points[:, i])
        feat[f'{axis}_skew'] = stats.skew(points[:, i])
        feat[f'{axis}_kurtosis'] = stats.kurtosis(points[:, i])

    # 形状比率
    ranges = sorted([feat['x_range'], feat['y_range'], feat['z_range']])
    feat['flatness'] = ranges[0] / (ranges[2] + 1e-8)
    feat['elongation'] = ranges[1] / (ranges[2] + 1e-8)

    # PCA 特征值
    cov = np.cov(points.T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total = np.sum(eig)
    feat['pca1_ratio'] = eig[0] / (total + 1e-8)
    feat['pca2_ratio'] = eig[1] / (total + 1e-8)
    feat['pca3_ratio'] = eig[2] / (total + 1e-8)
    feat['linearity'] = (eig[0] - eig[1]) / (eig[0] + 1e-8)
    feat['planarity'] = (eig[1] - eig[2]) / (eig[0] + 1e-8)
    feat['sphericity'] = eig[2] / (eig[0] + 1e-8)

    # 表面粗糙度
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=6)
    nn_dists = dists[:, 1:]
    feat['roughness_mean'] = np.mean(nn_dists)
    feat['roughness_std'] = np.std(nn_dists)
    feat['density_cv'] = np.std(nn_dists.mean(1)) / (np.mean(nn_dists.mean(1)) + 1e-8)

    # 高度分布（分位数）
    z = points[:, 2]
    for q in [10, 25, 50, 75, 90]:
        feat[f'z_q{q}'] = np.percentile(z, q)

    return feat


def extract_pig_features(pig_id, n_frames=None):
    """提取一头猪的平均特征向量"""
    pcd_dir = get_pig_dir(pig_id)
    if pcd_dir is None:
        return None

    txt_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.txt')])
    if not txt_files:
        return None

    if n_frames and len(txt_files) > n_frames:
        txt_files = list(np.random.choice(txt_files, n_frames, replace=False))

    all_feats = []
    for f in txt_files:
        pts = np.loadtxt(os.path.join(pcd_dir, f), dtype=np.float32)
        if len(pts) < 100:
            continue
        feat = extract_features_single(pts)
        all_feats.append(feat)

    if not all_feats:
        return None

    # 所有帧取平均
    keys = all_feats[0].keys()
    avg_feat = {k: np.mean([f[k] for f in all_feats]) for k in keys}
    return avg_feat


# ============================================================
# 主实验
# ============================================================

def main():
    print("=" * 70)
    print("  ABC三路线对比实验")
    print(f"  数据: nkyfirst {len(BACKFAT_LABELS)}头猪")
    print(f"  评估: {N_FOLDS}折交叉验证 (猪级别)")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    # --------------------------------------------------
    # 1. 加载数据
    # --------------------------------------------------
    print("\n[1/4] 加载数据...")

    pig_ids = []
    pig_backfats = []
    pig_frames_B = []     # 路线B: 点云数组
    pig_features_C = []   # 路线C: 特征向量

    for pid, bf in sorted(BACKFAT_LABELS.items()):
        print(f"  加载 {pid} (背膘={bf}mm)...", end="")

        # 路线B数据
        frames = load_pig_data(pid, n_frames=FRAMES_PER_PIG, n_points=NUM_POINTS)
        if frames is None:
            print(" 跳过(无数据)")
            continue

        # 路线C数据
        features = extract_pig_features(pid, n_frames=FRAMES_PER_PIG)
        if features is None:
            print(" 跳过(特征提取失败)")
            continue

        pig_ids.append(pid)
        pig_backfats.append(bf)
        pig_frames_B.append(frames)
        pig_features_C.append(features)
        print(f" OK ({len(frames)}帧)")

    pig_backfats = np.array(pig_backfats)
    n_pigs = len(pig_ids)
    print(f"\n  成功加载 {n_pigs} 头猪, 背膘范围 {pig_backfats.min():.1f}-{pig_backfats.max():.1f}mm")

    # --------------------------------------------------
    # 2. 路线C: 形态学特征 + XGBoost/SVR
    # --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  路线C: 形态学特征 + 回归模型")
    print(f"{'=' * 70}")

    # 构建特征矩阵
    feat_keys = sorted(pig_features_C[0].keys())
    X_C = np.array([[f[k] for k in feat_keys] for f in pig_features_C])
    y_C = pig_backfats

    print(f"  特征维度: {X_C.shape[1]}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for model_name, model_fn in [
        ('GBR', lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)),
        ('SVR-RBF', lambda: SVR(kernel='rbf', C=10, gamma='scale')),
    ]:
        preds_all = np.zeros(n_pigs)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_C)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_C[train_idx])
            X_test = scaler.transform(X_C[test_idx])
            y_train = y_C[train_idx]

            model = model_fn()
            model.fit(X_train, y_train)
            preds_all[test_idx] = model.predict(X_test)

        mae = mean_absolute_error(y_C, preds_all)
        rmse = np.sqrt(mean_squared_error(y_C, preds_all))
        r2 = r2_score(y_C, preds_all)
        print(f"\n  [{model_name}] MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  R²={r2:.4f}")

        # 每头猪的预测细节
        print(f"  {'耳号':<8} {'真实':>6} {'预测':>6} {'误差':>6}")
        for i in range(n_pigs):
            err = preds_all[i] - y_C[i]
            print(f"  {pig_ids[i]:<8} {y_C[i]:>6.1f} {preds_all[i]:>6.1f} {err:>+6.1f}")

    # --------------------------------------------------
    # 3. 路线B: PointMLP 回归
    # --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  路线B: PointMLP 回归")
    print(f"  模型参数: {sum(p.numel() for p in PointMLPRegressor().parameters()):,}")
    print(f"  训练: {EPOCHS}轮, lr={LR}, batch={BATCH_SIZE}")
    print(f"  点数: {NUM_POINTS}, 数据增强: 旋转+抖动+缩放")
    print(f"{'=' * 70}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    preds_B = np.zeros(n_pigs)

    for fold, (train_idx, test_idx) in enumerate(kf.split(pig_backfats)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}: "
              f"训练{len(train_idx)}头, 测试{len(test_idx)}头")

        train_frames = [pig_frames_B[i] for i in train_idx]
        train_labels = [pig_backfats[i] for i in train_idx]
        test_frames = [pig_frames_B[i] for i in test_idx]
        test_labels = [pig_backfats[i] for i in test_idx]

        t0 = time.time()
        fold_preds = train_pointmlp_fold(
            train_frames, train_labels, test_frames, test_labels
        )
        elapsed = time.time() - t0

        for j, ti in enumerate(test_idx):
            preds_B[ti] = fold_preds[j]

        # 打印本折结果
        fold_true = pig_backfats[test_idx]
        fold_pred = np.array(fold_preds)
        fold_mae = mean_absolute_error(fold_true, fold_pred)
        print(f"  Fold {fold + 1} MAE={fold_mae:.2f}mm ({elapsed:.0f}s)")
        for j, ti in enumerate(test_idx):
            err = fold_preds[j] - pig_backfats[ti]
            print(f"    {pig_ids[ti]}: 真实={pig_backfats[ti]:.1f}, "
                  f"预测={fold_preds[j]:.1f}, 误差={err:+.1f}")

    mae_B = mean_absolute_error(pig_backfats, preds_B)
    rmse_B = np.sqrt(mean_squared_error(pig_backfats, preds_B))
    r2_B = r2_score(pig_backfats, preds_B)

    print(f"\n  [PointMLP] MAE={mae_B:.2f}mm  RMSE={rmse_B:.2f}mm  R²={r2_B:.4f}")
    print(f"  {'耳号':<8} {'真实':>6} {'预测':>6} {'误差':>6}")
    for i in range(n_pigs):
        err = preds_B[i] - pig_backfats[i]
        print(f"  {pig_ids[i]:<8} {pig_backfats[i]:>6.1f} {preds_B[i]:>6.1f} {err:>+6.1f}")

    # --------------------------------------------------
    # 4. 总结对比
    # --------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  ABC 路线对比总结 (N={n_pigs}头猪, {N_FOLDS}折CV)")
    print(f"{'=' * 70}")
    print(f"  {'路线':<25} {'MAE(mm)':>10} {'RMSE(mm)':>10} {'R²':>10}")
    print(f"  {'-' * 55}")
    print(f"  {'C: GBR(特征)':.<25} {mae:.2f}mm  ...  (见上方详细结果)")
    print(f"  {'B: PointMLP':.<25} {mae_B:>10.2f} {rmse_B:>10.2f} {r2_B:>10.4f}")
    print(f"\n  路线A (深度图+CNN) 需要单独生成掩膜深度图数据，后续进行。")


if __name__ == '__main__':
    main()
