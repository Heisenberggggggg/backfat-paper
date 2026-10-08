# -*- coding: utf-8 -*-
"""
用师兄的点云数据集测试ABC路线 (B+C)
目的: 判断是数据/处理问题还是方法本身不可行

师兄数据:
  - 146头猪, 13140个点云文件
  - 背膘 8.1-22.3mm
  - 点云未归一化, 保留绝对尺寸
  - 每帧约5000点, 3列(X,Y,Z)

路线A无法做 (无原始深度图)
路线B: PointMLP 回归
路线C: 统计特征 + GBR/SVR
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
SENIOR_DIR = r'F:\Postgraduate\Lab\饲喂站\师兄资料\胡宝玖\所有数据'
PC_DIR = os.path.join(SENIOR_DIR, '猪只点云')
LABEL_FILE = os.path.join(SENIOR_DIR, '猪只背膘厚.txt')

NUM_POINTS = 1024
FRAMES_PER_PIG = 30
N_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 80
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
def load_labels():
    """加载背膘标签, 返回 {pig_id: backfat} 和 {filename: backfat}"""
    pig_labels = {}   # pig_id -> backfat
    file_labels = {}  # filename -> backfat

    with open(LABEL_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                fname, bf = parts[0], float(parts[1])
                file_labels[fname] = bf
                # 提取猪ID: "10-10_1.txt" -> "10-10"
                pig_id = fname.rsplit('_', 1)[0]
                pig_labels[pig_id] = bf

    return pig_labels, file_labels


def fps_downsample(points, n_target, max_input=5000):
    """最远点采样，先随机预降采样到max_input再FPS"""
    if len(points) <= n_target:
        idx = np.random.choice(len(points), n_target, replace=True)
        return points[idx]

    if len(points) > max_input:
        idx = np.random.choice(len(points), max_input, replace=False)
        points = points[idx]

    selected = np.zeros(n_target, dtype=int)
    selected[0] = np.random.randint(len(points))
    dists = np.full(len(points), np.inf)
    for i in range(1, n_target):
        p = points[selected[i - 1]]
        d = np.sum((points - p) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return points[selected]


def load_all_data():
    """加载所有猪的点云数据"""
    pig_labels, _ = load_labels()
    print(f"  标签: {len(pig_labels)} 头猪")

    # 按猪ID分组文件
    pig_files = {}
    for fname in os.listdir(PC_DIR):
        if not fname.endswith('.txt'):
            continue
        pig_id = fname.rsplit('_', 1)[0]
        if pig_id not in pig_labels:
            continue
        if pig_id not in pig_files:
            pig_files[pig_id] = []
        pig_files[pig_id].append(fname)

    pig_ids = sorted(pig_files.keys())
    backfats = []
    all_pc = []

    for pid in pig_ids:
        bf = pig_labels[pid]
        files = sorted(pig_files[pid])

        # 随机采样帧
        if len(files) > FRAMES_PER_PIG:
            sel = np.random.choice(len(files), FRAMES_PER_PIG, replace=False)
            files = [files[i] for i in sorted(sel)]

        frames = []
        for fname in files:
            pts = np.loadtxt(os.path.join(PC_DIR, fname), dtype=np.float32)
            if len(pts) < 200:
                continue
            pts = fps_downsample(pts, NUM_POINTS)
            frames.append(pts)

        if len(frames) < 5:
            continue

        backfats.append(bf)
        all_pc.append(np.array(frames, dtype=np.float32))

    # 过滤掉没有成功加载的猪
    valid_ids = []
    for i, pid in enumerate(pig_ids):
        if i < len(backfats):
            valid_ids.append(pid)
    pig_ids = valid_ids[:len(backfats)]

    return pig_ids, np.array(backfats), all_pc


# ============================================================
# 路线B: PointMLP (与我们的实验相同结构)
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


class PointMLPRegressor(nn.Module):
    """与abc_comparison_v2相同的PointMLP (无meta输入)"""
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
        x = x.transpose(1, 2)
        x = self.feat(x)
        x = torch.max(x, dim=2)[0]
        x = self.head(x)
        return x.squeeze(-1)


class PointCloudDataset(Dataset):
    def __init__(self, frames_list, labels_list, augment=False):
        self.data = []
        self.labels = []
        for frames, label in zip(frames_list, labels_list):
            for frame in frames:
                self.data.append(frame)
                self.labels.append(label)
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            pts = pts @ R.T
            pts += np.random.normal(0, 0.002, pts.shape).astype(np.float32)
            scale = np.random.uniform(0.97, 1.03)
            pts *= scale
        return torch.tensor(pts), torch.tensor(self.labels[idx])


def train_pointmlp_fold(train_pc, train_labels, test_pc, test_labels):
    train_ds = PointCloudDataset(train_pc, train_labels, augment=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)

    model = PointMLPRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(EPOCHS):
        for pts, labels in train_dl:
            pts, labels = pts.to(DEVICE), labels.to(DEVICE)
            pred = model(pts)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    pig_preds = []
    with torch.no_grad():
        for frames in test_pc:
            pts_t = torch.tensor(frames, dtype=torch.float32).to(DEVICE)
            preds = []
            for i in range(0, len(pts_t), BATCH_SIZE):
                batch = pts_t[i:i + BATCH_SIZE]
                pred = model(batch)
                preds.append(pred.cpu().numpy())
            pig_preds.append(np.concatenate(preds).mean())

    return pig_preds


# ============================================================
# 路线C: 统计特征 + GBR/SVR
# ============================================================
def extract_features(pc_frames):
    """从点云帧中提取统计特征, 取帧平均"""
    all_feats = []
    for pts in pc_frames:
        feat = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            feat[f'{axis}_std'] = np.std(pts[:, i])
            feat[f'{axis}_range'] = np.ptp(pts[:, i])
            feat[f'{axis}_skew'] = stats.skew(pts[:, i])
            feat[f'{axis}_kurtosis'] = stats.kurtosis(pts[:, i])
            feat[f'{axis}_mean'] = np.mean(pts[:, i])

        # 形状比率
        ranges = sorted([feat['x_range'], feat['y_range'], feat['z_range']])
        feat['flatness'] = ranges[0] / (ranges[2] + 1e-8)
        feat['elongation'] = ranges[1] / (ranges[2] + 1e-8)

        # PCA
        cov = np.cov(pts.T)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        total = np.sum(eig) + 1e-8
        feat['pca1'] = eig[0] / total
        feat['pca2'] = eig[1] / total
        feat['pca3'] = eig[2] / total
        feat['linearity'] = (eig[0] - eig[1]) / (eig[0] + 1e-8)
        feat['planarity'] = (eig[1] - eig[2]) / (eig[0] + 1e-8)
        feat['sphericity'] = eig[2] / (eig[0] + 1e-8)

        # 表面粗糙度
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=6)
        nn_dists = dists[:, 1:]
        feat['roughness_mean'] = np.mean(nn_dists)
        feat['roughness_std'] = np.std(nn_dists)
        feat['density_cv'] = np.std(nn_dists.mean(1)) / (np.mean(nn_dists.mean(1)) + 1e-8)

        # 包围盒尺寸 (绝对值)
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        bbox_dims = bbox_max - bbox_min
        feat['bbox_w'] = bbox_dims[0]
        feat['bbox_h'] = bbox_dims[1]
        feat['bbox_d'] = bbox_dims[2]
        feat['volume'] = np.prod(bbox_dims)

        # 质心位置 (绝对)
        centroid = pts.mean(axis=0)
        feat['centroid_z'] = centroid[2]

        # 最大半径
        feat['max_radius'] = np.max(np.linalg.norm(pts - centroid, axis=1))

        all_feats.append(feat)

    keys = sorted(all_feats[0].keys())
    avg = {k: np.mean([f[k] for f in all_feats]) for k in keys}
    return avg


# ============================================================
# 评估辅助
# ============================================================
def eval_summary(name, y_true, y_pred, pig_ids, show_detail=False):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n  [{name}] MAE={mae:.2f}mm  RMSE={rmse:.2f}mm  R²={r2:.4f}")

    if show_detail:
        # 只显示误差最大的10个和最小的5个
        errors = [(pig_ids[i], y_true[i], y_pred[i], y_pred[i] - y_true[i])
                  for i in range(len(pig_ids))]
        errors.sort(key=lambda x: abs(x[3]), reverse=True)
        print(f"  误差最大的10头:")
        print(f"  {'耳号':<10} {'真实':>6} {'预测':>6} {'误差':>6}")
        for pid, yt, yp, err in errors[:10]:
            print(f"  {pid:<10} {yt:>6.1f} {yp:>6.1f} {err:>+6.1f}")
        print(f"  误差最小的5头:")
        for pid, yt, yp, err in errors[-5:]:
            print(f"  {pid:<10} {yt:>6.1f} {yp:>6.1f} {err:>+6.1f}")

    return mae, rmse, r2


# ============================================================
# 主实验
# ============================================================
def main():
    print("=" * 70)
    print("  师兄数据集 ABC 路线对比实验")
    print(f"  设备: {DEVICE}")
    print("=" * 70)

    # 加载数据
    print("\n[加载数据]")
    t0 = time.time()
    pig_ids, backfats, all_pc = load_all_data()
    n_pigs = len(pig_ids)
    print(f"  加载完成: {n_pigs}头猪, 背膘 {backfats.min():.1f}-{backfats.max():.1f}mm")
    print(f"  均值: {backfats.mean():.1f}mm, 标准差: {backfats.std():.1f}mm")
    print(f"  耗时: {time.time()-t0:.1f}s")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results = {}

    # --------------------------------------------------
    # 路线C: 统计特征 + GBR/SVR
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("  路线C: 统计特征 + GBR/SVR")
    print(f"{'='*70}")

    print("  提取特征中...", end="", flush=True)
    t0 = time.time()
    pig_features = []
    for i, pid in enumerate(pig_ids):
        feat = extract_features(all_pc[i])
        pig_features.append(feat)
        if (i + 1) % 20 == 0:
            print(f" {i+1}", end="", flush=True)
    print(f" 完成 ({time.time()-t0:.1f}s)")

    feat_keys = sorted(pig_features[0].keys())
    X_C = np.array([[f[k] for k in feat_keys] for f in pig_features])
    print(f"  特征维度: {X_C.shape[1]}")

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
        results[model_name] = eval_summary(model_name, backfats, preds, pig_ids,
                                           show_detail=True)

    # --------------------------------------------------
    # 路线B: PointMLP 回归
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print("  路线B: PointMLP 回归 (不归一化)")
    n_params = sum(p.numel() for p in PointMLPRegressor().parameters())
    print(f"  参数量: {n_params:,}, 训练: {EPOCHS}轮")
    print(f"{'='*70}")

    preds_B = np.zeros(n_pigs)
    for fold, (tr, te) in enumerate(kf.split(backfats)):
        print(f"\n  Fold {fold+1}/{N_FOLDS}: 训练{len(tr)}头, 测试{len(te)}头")
        t0 = time.time()
        fold_preds = train_pointmlp_fold(
            [all_pc[i] for i in tr], [backfats[i] for i in tr],
            [all_pc[i] for i in te], [backfats[i] for i in te],
        )
        for j, ti in enumerate(te):
            preds_B[ti] = fold_preds[j]
        fold_mae = mean_absolute_error(backfats[te], np.array(fold_preds))
        print(f"  Fold {fold+1} MAE={fold_mae:.2f}mm ({time.time()-t0:.0f}s)")

    results['B-PointMLP'] = eval_summary(
        'B-PointMLP', backfats, preds_B, pig_ids, show_detail=True)

    # --------------------------------------------------
    # 最终汇总
    # --------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  最终对比汇总 (师兄数据, N={n_pigs}, {N_FOLDS}折CV)")
    print(f"{'='*70}")
    print(f"  {'路线':<20} {'MAE(mm)':>10} {'RMSE(mm)':>10} {'R²':>10}")
    print(f"  {'-'*50}")
    for name, (mae, rmse, r2) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"  {name:<20} {mae:>10.2f} {rmse:>10.2f} {r2:>10.4f}")

    mean_baseline = np.mean(np.abs(backfats - backfats.mean()))
    print(f"\n  背膘均值: {backfats.mean():.1f}mm, 标准差: {backfats.std():.1f}mm")
    print(f"  直接猜均值: MAE≈{mean_baseline:.2f}mm")

    print(f"\n{'='*70}")
    print(f"  与我们数据的对比")
    print(f"{'='*70}")
    print(f"  {'指标':<25} {'我们V2':>12} {'师兄数据':>12}")
    print(f"  {'-'*50}")
    print(f"  {'猪只数':<25} {'39':>12} {f'{n_pigs}':>12}")
    print(f"  {'背膘范围':<25} {'10.0-22.0mm':>12} {f'{backfats.min():.1f}-{backfats.max():.1f}mm':>12}")

    # 我们V2的结果 (硬编码)
    our_results = {
        'B-PointMLP': (2.19, 2.59, -0.001),
        'C-GBR': (2.33, 2.96, -0.305),
        'C-SVR': (2.24, 2.77, -0.145),
    }
    for name in ['B-PointMLP', 'C-GBR', 'C-SVR']:
        our_name = name + '+Meta' if name == 'B-PointMLP' else name
        our = our_results[name]
        sr = results.get(name, (0, 0, 0))
        print(f"  {name + ' R²':<25} {our[2]:>12.4f} {sr[2]:>12.4f}")
        print(f"  {name + ' MAE':<25} {our[0]:>12.2f} {sr[0]:>12.2f}")


if __name__ == '__main__':
    main()
