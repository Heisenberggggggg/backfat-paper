import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm


# ==============================================================================
# 1. PointMLP 模型定义 (这部分保持在顶层，以便被导入)
# ==============================================================================
# ... (模型代码和Dataset类的定义保持不变，这里省略以保持简洁)
class PointMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointMLPBlock, self).__init__()
        self.mlp1 = nn.Linear(in_channels, out_channels)
        self.mlp2 = nn.Linear(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.mlp1(x).permute(0, 2, 1)
        x = self.relu(self.bn1(x)).permute(0, 2, 1)
        x = self.mlp2(x).permute(0, 2, 1)
        x = self.bn2(x).permute(0, 2, 1)
        if identity.shape[-1] != x.shape[-1]:
            identity = nn.Linear(identity.shape[-1], x.shape[-1], device=x.device)(identity)
        x += identity
        x = self.relu(x)
        return x


class PointMLPRegressor(nn.Module):
    def __init__(self, num_points=256, in_dims=3):
        super(PointMLPRegressor, self).__init__()
        self.embedding = nn.Linear(in_dims, 64)
        self.block1 = PointMLPBlock(64, 64)
        self.block2 = PointMLPBlock(64, 128)
        self.block3 = PointMLPBlock(128, 256)
        self.regressor_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        global_feature = torch.max(x, dim=1)[0]
        return self.regressor_head(global_feature)


class BackfatDataset(Dataset):
    def __init__(self, data_dir, label_file, num_points=256):
        self.data_dir = data_dir
        self.num_points = num_points
        self.labels_map = self._load_labels(label_file)
        self.file_list = list(self.labels_map.keys())

    def _load_labels(self, label_file):
        labels_map = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    labels_map[filename] = float(label)
        return labels_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.data_dir, filename)
        point_cloud = np.loadtxt(file_path, dtype=np.float32)

        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices, :]
        elif len(point_cloud) < self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=True)
            point_cloud = point_cloud[indices, :]

        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        dist = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
        if dist > 0:
            point_cloud = point_cloud / dist

        label = self.labels_map[filename]
        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ==============================================================================
# 3. 训练和验证主流程 (将所有执行代码放入这个if块中)
# ==============================================================================
if __name__ == '__main__':
    # --- 你需要修改这4个路径 ---
    TRAIN_DIR = "F:\\Postgraduate\\Lab\\饲喂站\\背膘算法\\网络\\Point-Transformers_jiu_硕改\\train"
    VAL_DIR = "F:\\Postgraduate\\Lab\\饲喂站\\背膘算法\\网络\\Point-Transformers_jiu_硕改\\valid"
    TRAIN_LABEL_FILE = "F:\\Postgraduate\\Lab\\饲喂站\\背膘算法\\网络\\Point-Transformers_jiu_硕改\\train.txt"
    VAL_LABEL_FILE = "F:\\Postgraduate\\Lab\\饲喂站\\背膘算法\\网络\\Point-Transformers_jiu_硕改\\vaild.txt"
    # -----------------------------

    # --- 设置超参数 ---
    EPOCHS = 60  # 之前是50，根据你的截图改成了60
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_POINTS = 512
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 准备数据 ---
    train_dataset = BackfatDataset(TRAIN_DIR, TRAIN_LABEL_FILE, num_points=NUM_POINTS)
    val_dataset = BackfatDataset(VAL_DIR, VAL_LABEL_FILE, num_points=NUM_POINTS)

    # 之前设置了num_workers=4，如果还报错可以尝试设为0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # --- 初始化模型、损失函数、优化器 ---
    model = PointMLPRegressor(num_points=NUM_POINTS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 训练循环 ---
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for points, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_gradrain
            outputs = model(points).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for points, labels in val_loader:
                points, labels = points.to(DEVICE), labels.to(DEVICE)
                outputs = model(points).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val RMSE: {np.sqrt(avg_val_loss):.4f} mm")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_pointmlp_regressor.pth")
            print("-> Best model saved!")