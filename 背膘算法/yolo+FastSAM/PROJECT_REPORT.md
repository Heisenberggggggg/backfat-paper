# 母猪背膘厚度非接触式测量 - 项目工作报告

> 生成日期: 2026-03-11
> 项目目标: 基于深度相机(Azure Kinect)的俯视3D点云/深度图，预测母猪背膘厚度(mm)
> 部署目标: Jetson Nano 8GB

---

## 一、项目概述

### 1.1 技术路线
```
RGB+Depth图像 (Azure Kinect, 俯视安装)
    ↓
YOLO11n 检测猪只位置 (目标检测)
    ↓
FastSAM-s 分割猪体掩膜 (实例分割)
    ↓
掩膜 + 深度图 + 相机内参 → 3D掩膜点云
    ↓
后处理: HDBSCAN去噪 → 深度异常滤波 → 统计滤波 → 体素下采样 → 归一化
    ↓
回归模型 → 背膘厚度预测 (mm)
```

### 1.2 数据概况
| 批次 | 目录 | 猪只数 | 标签来源 |
|------|------|--------|----------|
| nkyfirst | `G:\耿来硕数据\DK\nkyfirst` | 39头 (耳标2-X, 3-X) | 第一次数据7.29.docx |
| nkysecond | `G:\耿来硕数据\DK\nkysecond` | 32头 | - |
| nkythird | `G:\耿来硕数据\DK\nkythird` | 6头 | - |
| nky3.18 | `G:\耿来硕数据\DK\nky3.18` | 18头 | - |
| 丰美禾 | `G:\耿来硕数据\丰美禾数据\` | 约800头 (13个日期子目录) | 背膘记录表.xlsx |

nkyfirst的39头猪背膘标签(B超值,mm):
```python
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
# 均值: 16.6mm, 标准差: 2.6mm, 范围: 10.0-22.0mm
```

### 1.3 相机内参
```python
# 省农科院 Azure Kinect (SN: 000341693512) - nky批次
NKY_INTRINSICS = {'fx': 614.84, 'fy': 614.61, 'cx': 642.54, 'cy': 364.29}

# 丰美禾 Azure Kinect (SN: 000493521612)
FMH_INTRINSICS = {'fx': 608.39, 'fy': 608.10, 'cx': 641.02, 'cy': 363.46}
```

---

## 二、已完成的工作

### 2.1 单猪处理流程验证 (猪 2-1)
- 对猪2-1生成了181帧掩膜点云
- 后处理结果: 猪体被栏杆分割成3段 (头部~13%, 背部~64%, 臀部~21%)
- 用户评价: "干干净净的三个点云块"

### 2.2 批处理脚本 (batch_process_all.py)
完整自动化流程，支持5个批次不同目录结构、断点续传、GPU显存自动清理。

#### 2.2.1 HDBSCAN预降采样Bug修复
**问题**: 原始体素降采样公式 `voxel_size = (bbox_vol / 50000) ** (1/3)` 对扁平点云（Z轴范围很小）计算出极大的体素尺寸，导致282,652点只剩828点（应为~50,000）。

**修复**: 用随机子采样替代体素降采样:
```python
# 修复前 (Bug):
bbox = points.max(0) - points.min(0)
bbox_vol = np.prod(bbox + 1e-6)
voxel_size = (bbox_vol / HDBSCAN_PRE_DOWNSAMPLE) ** (1.0 / 3.0)  # 对扁平点云会算出极大voxel
pcd_down = pcd.voxel_down_sample(voxel_size)

# 修复后:
if len(points) > HDBSCAN_PRE_DOWNSAMPLE:
    indices = np.random.choice(len(points), HDBSCAN_PRE_DOWNSAMPLE, replace=False)
    indices.sort()
    points_for_cluster = points[indices]
```

**验证结果** (猪 pig_00159):
- 原始点数: 282,652 → 预降采样: 50,000 (精确)
- HDBSCAN发现3个簇 → 深度滤波去除16,592个栏杆点 → 最终4,029点

#### 2.2.2 深度异常滤波器
用Z坐标的MAD(中位数绝对偏差)去除与猪体深度不同的栏杆点:
```python
def depth_outlier_filter(points, k=3.0):
    z = points[:, 2]
    median_z = np.median(z)
    mad = np.median(np.abs(z - median_z))
    lower = median_z - k * 1.4826 * mad
    upper = median_z + k * 1.4826 * mad
    mask = (z >= lower) & (z <= upper)
    return points[mask]
```
效果: HDBSCAN去除0.9%噪声点, depth_outlier_filter去除5.9%栏杆点。对于边缘部署(Jetson Nano)，可考虑省略HDBSCAN只保留depth_outlier_filter。

#### 2.2.3 批处理进度 (截至2026-03-11)
| 批次 | 进度 |
|------|------|
| nkyfirst | 39/39 完成 |
| nkysecond | 11/32 进行中 |
| nkythird | 0/6 |
| nky3.18 | 0/18 |
| 丰美禾 | 0/~800 |

脚本PID: 45684, 仍在后台运行中。用户要求跑完丰美禾/8.24后停止。

---

## 三、ABC路线对比实验

### 3.1 实验设计

#### 三条路线:
- **路线A**: 掩膜深度图(224x224) → 轻量CNN(类MobileNet, depthwise separable convs, 86,785参数)
- **路线B**: 3D点云(1024点) → PointMLP(ResMLPBlock x3, 208,513/209,537参数)
- **路线C**: 点云统计/形态学特征 → GBR / SVR

#### 评估方式:
- 5折交叉验证, 猪级别划分(确保同一头猪不会同时出现在训练和测试集)
- 指标: MAE(mm), RMSE(mm), R²
- 基线: 直接猜均值(16.6mm) → MAE≈2.20mm, R²=0.00

### 3.2 第一轮实验 (V1 - 归一化点云)

**使用数据**: `processed_pointclouds/` 目录下的归一化到单位球的.txt点云文件

**脚本**: `abc_comparison.py`

| 路线 | MAE (mm) | RMSE (mm) | R² |
|------|---------|----------|-----|
| C-GBR | 2.08 | - | -0.02 |
| C-SVR | 2.11 | - | 0.05 |
| B-PointMLP | 2.13 | - | 0.03 |
| 猜均值基线 | ~2.20 | ~2.60 | 0.00 |

**结论**: 所有模型本质上在预测均值(~16mm)，R²接近0，无有效信号。

**原因分析**:
归一化到单位球(`points = points / max_radius`)消除了所有绝对尺寸信息。不同大小的猪归一化后点云形状几乎相同，模型无法区分。

### 3.3 第二轮实验 (V2 - 不归一化 + 全局特征 + 深度图)

**改进措施**:
1. 点云只做中心化(减去质心)，不缩放到单位球
2. 加入8维全局元特征: bbox_w, bbox_h, bbox_d, centroid_z, n_raw_pts, max_radius, masked_area_ratio, volume
3. 新增路线A: 掩膜深度图 + 轻量CNN

**数据生成脚本**: `generate_abc_data.py`
- 输出目录: `abc_data/`
- 每头猪: depth_{pid}.npy (30,224,224), pc_{pid}.npy (30,1024,3), meta_{pid}.npy (30,8)
- FPS下采样修复: 对大量点云(>5000点)先随机预降采样到5000再FPS到1024

**对比脚本**: `abc_comparison_v2.py`

| 路线 | MAE (mm) | RMSE (mm) | R² | 对比猜均值 |
|------|---------|----------|-----|-----------|
| A-DepthCNN | 2.10 | 2.50 | 0.070 | 略好 |
| B-PointMLP+Meta | 2.19 | 2.59 | -0.001 | 相当 |
| C-SVR | 2.24 | 2.77 | -0.145 | 更差 |
| C-GBR | 2.33 | 2.96 | -0.305 | 更差 |
| **猜均值基线** | **~2.20** | **~2.60** | **0.00** | - |

**典型预测失败案例**:
- 猪3-17 (真实10.0mm) → 所有模型预测14.7-17.0mm (偏差+4.7~+7.0)
- 猪3-23 (真实22.0mm) → 所有模型预测15.0-17.5mm (偏差-4.5~-7.0)

**结论**: 即使取消归一化并加入全局特征，R²仍接近0。所有模型仍然在预测均值。

---

## 四、关键发现与分析

### 4.1 为什么不work?
1. **俯视角度的根本局限**: 背膘是皮下脂肪层厚度，从俯视角度只能捕获背部表面曲率。表面曲率主要由骨架结构和体型决定，而非脂肪厚度。同样体型的猪可能背膘差异很大。
2. **栏杆遮挡**: 限位栏将猪体分割成3段(头、背、臀)，完整背部形态信息受损严重。
3. **样本量小**: 39头猪(nkyfirst)，背膘分布集中在13-20mm，对深度学习来说样本太少。
4. **帧间姿态差异**: 猪在不同帧的姿态不同(站立/趴卧/走动)，增加了噪声。

### 4.2 师兄方案为什么work?
师兄使用PointNet++/Point Transformer取得了较好效果，可能原因:
- 可能使用了不同的视角或数据采集条件
- 可能有更大的样本量或更宽的背膘分布范围
- 模型更大(不受Jetson Nano部署限制)
- 具体实验细节需进一步确认

### 4.3 关于边缘部署
- 完整推理流程(YOLO+FastSAM+后处理+PointMLP)需部署到Jetson Nano 8GB
- 当前模型大小: DepthCNN 86K参数, PointMLP 209K参数 — 模型本身足够轻量
- 关键瓶颈: YOLO11n + FastSAM-s的推理延迟(非模型本身)
- HDBSCAN在部署时可省略(仅去除0.9%点)，depth_outlier_filter(去除5.9%)更有效且快速

---

## 五、文件清单

### 关键脚本
| 文件 | 用途 | 状态 |
|------|------|------|
| `batch_process_all.py` | 批量处理所有猪只点云(HDBSCAN+滤波+归一化) | 运行中(PID 45684) |
| `generate_abc_data.py` | 生成ABC V2实验数据(不归一化+深度图+元特征) | 完成 |
| `abc_comparison.py` | ABC路线对比V1(归一化点云) | 完成 |
| `abc_comparison_v2.py` | ABC路线对比V2(不归一化+全局特征+深度图) | 完成 |
| `quick_feasibility_check.py` | 5头猪快速可行性检验 | 完成(已删除) |
| `try.py` | YOLO + FastSAM 检测/分割 | 基础工具 |
| `generate_pointcloud.py` | 深度图 → 掩膜点云(.ply) | 基础工具 |
| `postprocess_pointcloud.py` | HDBSCAN + 滤波 + 下采样 + 归一化 | 基础工具 |

### 数据目录
| 目录 | 内容 |
|------|------|
| `各猪目录/processed_pointclouds/` | 归一化后的.txt点云(4000-5000点) |
| `abc_data/` | V2实验数据: depth_*.npy, pc_*.npy, meta_*.npy (39头猪) |

### 已知Bug (已修复)
1. **batch_process_all.py HDBSCAN预降采样**: 体素估算对扁平点云失效 → 改用随机子采样
2. **generate_abc_data.py FPS内存溢出**: 227k点直接FPS导致OOM → 加max_input=5000预降采样
3. **网络/train_mlp.py line 139**: `optimizer.zero_gradrain` 拼写错误 → 应为`zero_grad()`
4. **网络/train_mlp.py**: PointMLPBlock在forward()中创建nn.Linear → 每次前向传播权重随机

---

## 六、后续可能方向

### 6.1 数据层面
- **增加样本量**: 等全部批次(~973头)处理完毕后再次尝试，可能信号更强
- **换视角**: 文献表明侧视/后视角度包含更多背膘信息(臀部肌肉轮廓与脂肪层相关性更强)
- **多视角融合**: 如果可以安装多台相机，融合俯视+侧视

### 6.2 目标层面
- **体重估计**: 体表3D形态与体重相关性远强于与背膘的相关性
- **体况评分(BCS)**: 分3-5个等级而非连续回归，降低任务难度
- **多模态**: RGB纹理 + 深度几何融合

### 6.3 模型层面
- **更深的网络**: 如果不受Jetson Nano限制，PointNet++/Point Transformer可能有更强特征提取能力
- **预训练**: 使用3D预训练模型迁移学习
- **注意力机制**: 让模型自动关注背部区域而非整头猪

### 6.4 Python环境
```
D:\Anaconda\envs\pytorch\python.exe
依赖: torch(CUDA), ultralytics, open3d, hdbscan, scipy, scikit-learn, numpy
```

---

## 七、给接续Claude的上下文摘要

如果你将此报告发给新的Claude对话，请附上以下关键信息:

1. **项目**: 母猪背膘非接触测量, 深度相机(Azure Kinect)俯视安装, 部署到Jetson Nano 8GB
2. **当前结论**: 两轮ABC实验(归一化 vs 不归一化+全局特征)均显示R²≈0, 俯视点云/深度图可能不包含足够的背膘预测信号
3. **批处理状态**: batch_process_all.py 仍在运行(PID 45684), nkyfirst完成, nkysecond处理中, 用户要求跑完丰美禾/8.24后停止
4. **关键修复**: HDBSCAN预降采样(随机替代体素), FPS预降采样(max_input=5000), train_mlp.py的typo和权重bug
5. **用户要求**: 执行代码修改前必须先列出计划让用户确认; 不覆盖已有processed_pointclouds数据
6. **Python环境**: `D:\Anaconda\envs\pytorch\python.exe`
7. **代码位置**: `F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\`
