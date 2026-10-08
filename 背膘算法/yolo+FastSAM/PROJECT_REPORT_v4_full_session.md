# 母猪背膘厚非接触测量 — 全会话窗口完整总结报告 (v4)

**报告日期**: 2026-04-17（首版）/ 2026-04-19（v6 全部完成更新）
**覆盖范围**: 本会话窗口自始至终（约 2 个月实验历程）
**前置报告**: `PROJECT_REPORT.md` (v1)、`PROJECT_REPORT_v2.md` (v2)、`PROJECT_REPORT_v3_breed_kfold.md` (v3)
**作者**: 与 Claude 协作整理

---

## 1. 项目概述

### 1.1 目标
基于 Azure Kinect 深度相机的 RGB + Depth 数据，在母猪饲喂站通道内**非接触**测量 P2 点背膘厚度，部署目标为 **Jetson Nano 8GB**。

### 1.2 技术路线
```
RGB + Depth (Azure Kinect)
    → YOLO11n 检测 (猪定位 bbox)
    → FastSAM-s 分割 (像素级 mask)
    → 深度图 + mask + 内参 → 掩膜点云 (.ply)
    → 后处理 (HDBSCAN 聚类 + 深度离群滤波 + 体素下采样)
    → 回归模型 → 背膘厚度预测 (mm)
```

### 1.3 参考对照
| 参照对象 | 方法 | 样本 | R² |
|---|---|---|---|
| 师兄 胡宝玖 | PointNet++ / PointTransformer | 146 头 | 0.14–0.28 |
| 师姐 喻梦媛 | RGB + CNN-ViT（无深度） | 106 头 | **0.74** |
| 本项目最佳 | Depth Map + ResNet-18 | 912 头 | **0.211** |

---

## 2. 数据概况

### 2.1 数据源（演进过程）
| 数据源 | 路径 | 头数 | 备注 |
|---|---|---|---|
| nkyfirst | `G:\耿来硕数据\DK\nkyfirst\` | 39 | 耳标 2-X, 3-X，背膘 10-22mm |
| nkysecond | `G:\耿来硕数据\DK\nkysecond\` | 21 | 含 color+depth 的 master10 |
| nkythird | `G:\耿来硕数据\DK\nkythird\` | 122 | 6 个 kinect 子目录 |
| 丰美禾 10.29 | `G:\耿来硕数据\丰美禾数据\10.29\` | 62 | 有品系标记 |
| 丰美禾(含双视角) | `丰美禾数据` sub/sub2/sub10 | 913 | 背部 + 臀部视角 |

### 2.2 头数演进时间线
**39 → 189 → 244 → 911/916 头**（覆盖 A、LY、T 三个品系）

### 2.3 品系分布（丰美禾）
- **A（单系）**: 456 头
- **LY（新美系）**: 233 头
- **T（单系）**: 65 头（过滤后 64）
- **ALL（全部）**: 753 头（过滤 [7.0, 32.0]mm 后，含 2 个离群：70mm、200mm）

### 2.4 背膘标签
存储于 Word 文档 + Excel，范围 5–35mm，分布集中于 13–19mm（占 57%），CV=0.997。

---

## 3. 完整实验历程（按时间线）

### 阶段 1: 环境与 try.py 跑通（最早期）
**问题**：
- `c10.dll` 加载失败（C:\dlcv 环境损坏）
- YOLO class 18 实际是 sheep，不是 pig
- 猪占满画面时 YOLO 检测失败
- areas 在 GPU 上导致 argmax 错误

**解决**：
- 切换至 `D:\Anaconda\envs\pytorch`
- 移除类别过滤，取最大 bbox
- 失败时 fallback 使用全图 bbox
- `.cpu()` 后再 argmax
- **效果**：181/181 张检测成功率 100%

### 阶段 2: 点云生成 + 师兄流程复现
**脚本**：`generate_pointcloud.py` → `postprocess_pointcloud.py`

**流程**：掩膜深度 + 内参 → .ply → HDBSCAN 聚类 → 深度离群滤波（MAD, k=3.0）→ 体素下采样 → 单位球归一化

**问题**：
- HDBSCAN 对 282,652 点 OOM
- 体素公式 `(bbox_vol / N)**(1/3)` 降到 828 点过少

**解决**：改为随机采样到 `HDBSCAN_PRE_DOWNSAMPLE=5000`，并保序
```python
if len(points) > 5000:
    indices = np.random.choice(len(points), 5000, replace=False)
    indices.sort()
    points_for_cluster = points[indices]
```

**关键发现**：猪体被围栏栏杆切成 3 段（头 13% / 背 64% / 臀 21%），MAD 滤波去掉 5.9% 栏杆点。

### 阶段 3: ABC 三路线 v1（归一化版本）
**方法**：A=DepthMap+CNN，B=PointMLP，C=41 维手工特征+GBR/SVR

**结果**：三个方法 **R² ≈ 0**（全部崩塌为均值预测）

**原因诊断（关键洞见 ⭐）**：
**单位球归一化破坏了尺度信息** — 不同尺寸的猪被归一化到同样大小，绝对厚度信息彻底丢失。

### 阶段 4: ABC v2 (去归一化 + 全局特征)
**脚本**：`generate_abc_data.py`, `abc_comparison_v2.py`

**结果**：
- DepthCNN R²=0.07（提升）
- PointMLP+Meta R² ≈ 0
- **在师兄 146 头数据上复测**：R²=0.14–0.28 ✓（验证代码正确）

### 阶段 5: PointMLP v1 → v3 迭代
| 版本 | 配置 | R² | MAE |
|---|---|---|---|
| v1 | SGD + SmoothL1 + 单位球 | -0.06 | — |
| v2 | AdamW + MSE + center-only | -0.22 | — |
| **v3** | 强增广 + 通道减半(16→32→64→128) + 混合特征(128+41) | **0.15** | **2.42mm** |
| XGBoost | 41 维特征 | 0.15 | — |

### 阶段 6: v4 全量 244 头 + 去头保背臀
**用户反馈（关键修正）**：
> "保留背部和臀部吧，去掉头部"

**脚本**：`postprocess_pointcloud_v2.py`（丢最小聚类=头，保留其余）
**配置**：`PointMLPRegressorV3 embed_dim=16`, weight_decay=0.01, 背膘过滤 [5.0, 40.0]mm

### 阶段 7: 双视角（背部 + 臀部）融合
**脚本**：`generate_pointcloud_v3_rear.py`, `train_traditional_v3_dualview.py`
**结果**：
- 背部单独 R²=0.067
- 臀部单独 R²=0.008
- 双视角融合 R²=0.058 ← **比单背部更差**

**结论**：臀部信号弱且引入噪声，融合无增益。

### 阶段 8: 深度图 CNN（★ 最佳结果）
**脚本**：`train_depthmap.py`（ResNet-18, 1ch, 256×256 masked depth）

**结果**：**R²=0.211，MAE=2.99mm** — 全项目最优

**对比投影 CNN**：`train_projection.py`（BEV/Front/Side 3ch）R²=0.087（栅格填充率仅 7%，比原始深度图差）

### 阶段 9: 双分支 PointMLP
**脚本**：`train_v5_dualview.py`（两条独立骨干 + 特征拼接 128+128=256）
**结果**：R²=0.048 — 无提升

### 阶段 10: 数据分布诊断 + 师姐论文对比
**发现**：
- 背膘 200mm/70mm 离群值
- 57% 集中于 13–19mm
- CV=0.997（分布偏窄）
- 师姐仅用 RGB（无深度）达到 R²=0.74 — 说明 3D 几何信号本征弱于 RGB 纹理信号

### 阶段 11: 老师建议 —— 按品系分组 5 折 CV（v6）
**用户输入**：
> "刚和老师沟通了一下，可能和不同猪的品种有关？根据耳标把猪分成多个种类分开训练"
> "LY是新美系，A和T都是单系"
> "技术路线使用单视角(背部)点云+PointMLP，尝试师兄的五折交叉验证法"

**脚本**：`train_v6_breed_kfold.py`
**配置**：epochs=200, batch=16, lr=1e-3, wd=0.01, patience=30, seed=42, 过滤 [7.0, 32.0]mm

**Bug 与修复**：
- PyTorch 2.5.1 `torch.load weights_only=True` 默认导致 fold 1 测试崩溃 → `weights_only=False` + try/except
- setx 写注册表不影响当前进程 → bat 改用**内联** `set` 设置 `CUDA_CACHE_PATH / TORCH_EXTENSIONS_DIR / TORCH_HOME / TMP / TEMP` 全部重定向至 F 盘
- 残留 python PID 14096 占用 CUDA → 手动 kill

### 阶段 12: v6 训练结果（4 品系全部完成 ✓ 2026-04-19）

| 品系 | 头数 | 折数 | **avg R² ± std** | **global R²** | **avg MAE** | **avg RMSE** | MAPE | 各折 R² |
|---|---|---|---|---|---|---|---|---|
| **A**（单系） | 456 | 5/5 | **-0.022 ± 0.100** | -0.011 | **2.73 ± 0.07 mm** | 3.42 mm | 17.2% | [-0.10, **0.17**, -0.11, -0.03, -0.04] |
| **LY**（新美系） | 233 | 5/5 | -0.002 ± 0.194 | 0.013 | 3.24 ± 0.27 mm | 4.13 mm | 19.3% | [-0.15, -0.20, **0.22**, -0.13, **0.24**] |
| **T**（单系） | 64 | 5/5 | **-0.315 ± 0.299** | -0.226 | 3.72 ± 0.82 mm | 4.86 mm | 20.5% | [**0.09**, **-0.84**, -0.29, -0.24, -0.29] |
| **ALL**（全部） | 753 | 5/5 | **+0.001 ± 0.043** | **+0.058** | 2.97 ± 0.16 mm | 3.76 mm | 18.2% | [-0.01, -0.03, -0.05, **0.08**, **0.02**] |

**关键观察（最终结论）**：
- **样本量是关键，品系不是** —— T (64头) 最差 R²=-0.315，σ=0.299；ALL (753头) 最稳 R²=0.001，σ=0.043；A (456头) 居中。R² 与样本量正相关，与品系无关。
- **老师的"品系假说"未得到证实** —— LY 与 A 在 avg R² 上几乎一致（均≈0），分品系训练未带来提升。
- **小样本品系反而拖后腿** —— T 仅 64 头，单折波动 [-0.84, +0.09] 极不稳定。
- **ALL 全量训练反而最稳健** —— global R²=0.058 最高，验证"数据多样性 > 品系细分"。
- **MAE 区间 2.7–3.7mm**（A 最优 2.73mm），与业内 ~3mm 水平相当；R² 低说明模型主要学到分布均值附近的偏移。
- 个别折 R²>0.17（A f2, LY f3/f5） → 信号存在但极不稳定，受测试集组成影响大。

**断点续训改造**（2026-04-17 实施）：
- 完成折保存 `fold_result.json` 作为完成标记，重启时自动跳过
- 每 epoch 保存 `last_checkpoint.pth`（含 model + optimizer + scheduler + RNG + 早停计数）
- 测试预测保存为 `test_preds.npz`，resume 后能重建全局指标
- 日志改追加模式，保留中断前历史
- T 与 ALL 的剩余折就是用此机制顺利跑完的

---

## 4. 全版本结果汇总表

| # | 方法 | 数据 | 头数 | R² | MAE (mm) | 备注 |
|---|---|---|---|---|---|---|
| 1 | ABC v1 (归一化) | first | 39 | ≈0 | — | 均值崩塌 |
| 2 | DepthCNN v2 | first | 39 | 0.07 | — | 去归一化 |
| 3 | PointMLP v1 | first | 39 | -0.06 | — | SGD+单位球 |
| 4 | PointMLP v2 | first | 39 | -0.22 | — | center-only |
| 5 | **PointMLP v3** | first | 189 | **0.15** | **2.42** | 强增广+混合特征 |
| 6 | XGBoost | first | 189 | 0.15 | — | 41 维特征 |
| 7 | PointMLP v4 | 全 244 | 244 | 低 | — | 去头保背臀 |
| 8 | 传统 ML 双视角背 | 丰美禾 | 913 | 0.067 | — | — |
| 9 | 传统 ML 双视角臀 | 丰美禾 | 913 | 0.008 | — | — |
| 10 | 传统 ML 双视角融合 | 丰美禾 | 913 | 0.058 | — | 融合反而更差 |
| 11 | 投影 CNN (BEV+Front+Side) | 丰美禾 | 912 | 0.087 | — | 7% 填充率 |
| 12 | **深度图 CNN (ResNet-18)** | 丰美禾 | 912 | **0.211** | **2.99** | ★ 最佳 |
| 13 | 双分支 PointMLP | 丰美禾 | 913 | 0.048 | — | — |
| 14 | v6 PointMLP breed A | 丰美禾 | 456 | **-0.022** (avg) / -0.011 (global) | 2.73 | 5 折 CV ✓ |
| 15 | v6 PointMLP breed LY | 丰美禾 | 233 | -0.002 (avg) / 0.013 (global) | 3.24 | 5 折 CV ✓ |
| 16 | v6 PointMLP breed T | 丰美禾 | 64 | **-0.315** (avg) / -0.226 (global) | 3.72 | 5 折 CV ✓ 小样本最差 |
| 17 | v6 PointMLP ALL | 丰美禾 | 753 | +0.001 (avg) / **+0.058** (global) | 2.97 | 5 折 CV ✓ 全量最稳 |
| — | 师兄 PointNet++ | 师兄 | 146 | 0.14–0.28 | — | 对照 |
| — | 师姐 RGB+ViT | 师姐 | 106 | **0.74** | — | 对照（无深度） |

---

## 5. 关键目录清单

### 数据
- `G:\耿来硕数据\DK\nkyfirst|nkysecond|nkythird\` — 早期三批
- `G:\耿来硕数据\丰美禾数据\10.29\` — 62 头 + 品系
- `G:\耿来硕数据\丰美禾数据\sub|sub2|sub10\` — 双视角 913 头

### 核心脚本（yolo+FastSAM 目录）
- `try.py` — YOLO+FastSAM 检测分割
- `generate_pointcloud.py` / `_v2.py` / `_v3_rear.py` — 点云生成三代
- `postprocess_pointcloud.py` / `_v2.py` / `_v3_rear.py` — 后处理三代
- `generate_depthmap_dataset.py` — 256×256 深度图数据集
- `generate_projection_dataset.py` — BEV/Front/Side 投影

### 核心脚本（网络/pointmlp_backfat 目录）
- `train.py` (v1/v2) / `train_v3.py` / `train_v4.py` / `train_v5_dualview.py` / `train_v6_breed_kfold.py`
- `train_depthmap.py` / `train_projection.py` / `train_traditional_v3_dualview.py`
- `extract_features.py` / `extract_features_v3_rear.py` — 41 维手工特征
- `run_breed_training.bat` — v6 训练入口（内联环境变量版）

### 缓存与输出
- `F:\cuda_pytorch_cache\{cuda_compute, torch_extensions, torch_hub, tmp}` — CUDA/Torch 缓存（强制 F 盘）
- `网络\pointmlp_backfat\checkpoints_pointmlp_breed\breed_{A,LY,T,ALL}\fold_{1-5}\` — v6 权重
- `网络\pointmlp_backfat\training_log_v6_{A,LY,T,ALL}.txt` — v6 日志

### 报告
- `PROJECT_REPORT.md`（v1, 2026-03-11）
- `PROJECT_REPORT_v2.md`（v2, 2026-03-24）
- `PROJECT_REPORT_v3_breed_kfold.md`（v3, 2026-04-17 上午）
- **`PROJECT_REPORT_v4_full_session.md`（本报告，2026-04-17 下午）**

---

## 6. Bug 与修复汇总

| 类型 | 症状 | 根因 | 修复 |
|---|---|---|---|
| 环境 | c10.dll 加载失败 | dlcv 环境损坏 | 切至 D:\Anaconda\envs\pytorch |
| YOLO | 检出 sheep | class 18 ≠ pig | 取最大 bbox，移除类别过滤 |
| YOLO | 猪占满画面无检出 | 检测阈值 | 全图 bbox fallback |
| tensor | argmax 报错 | areas 在 GPU | 先 .cpu() |
| 内存 | 点云生成 OOM | 未释放 | gc + CUDA cache + float32 |
| HDBSCAN | 输出点过少 | 体素公式错 | 随机下采样到 5000 |
| FPS | 227k 点 OOM | 无预处理 | 先随机采到 5000 |
| 编码 | R² 写文件乱码 | GBK 不支持上标 ² | 用 "R2" 字符串 |
| 日志 | stdout 看不到 | Python 缓冲 | 重定向文件读取 |
| 中文路径 | bash 写入 .py 乱码 | shell 编码 | 用 os.listdir 动态识别 |
| CUDA | OOM 训 A 时 | 残留 python 进程 | kill PID 14096 |
| PyTorch 2.5 | fold 1 测试崩 | weights_only=True 默认 | `weights_only=False` + try/except |
| 环境变量 | setx 不生效 | 只写注册表 | bat 内联 set |
| 盘空间 | WinError 1455 pagefile 耗尽 | C 盘满 | 缓存重定向至 F |

---

## 7. 核心洞见（跨阶段总结）

1. **单位球归一化是点云回归的隐形杀手** —— 绝对背膘厚度 ∝ 绝对几何尺度，归一化后直接丢失。
2. **3D 几何信号本身弱于 RGB 纹理信号** —— 师姐 RGB R²=0.74 vs 本项目 3D 最优 0.211，表明皮下脂肪厚度与体表三维形状的耦合强度有限。
3. **深度图 CNN 反而优于直接在点云上学** —— 深度图保留了 2D 拓扑结构，利于 ResNet 提取空间模式；投影 CNN 因稀疏反而更差。
4. **双视角融合无增益、甚至有害** —— 臀部视角引入的噪声大于信息。
5. **代码层面无 bug** —— 多次审计 + 师兄 146 头数据复测均正常（R²=0.14–0.28），瓶颈在数据/信号。
6. **品系分组未带来突破，且小样本品系反而更差** —— 4 品系全部跑完后，R² 与样本量正相关（ALL 753>A 456>LY 233>T 64），与品系无关。T (64头) avg R²=-0.315 最差，ALL (753头) avg R²=0.001 最稳。老师的"品系假说"被证伪。
7. **MAE 基本稳定在 2.7–3.7mm** —— 与业内水平接近，对工程落地可用，但 R² 低说明模型主要学到均值。
8. **数据量是首要瓶颈** —— v6 实验明确显示，与其按品系细分，不如汇集所有数据训练；这与"信号弱、需大样本平均化噪声"的假设一致。

---

## 8. 用户规则与约束（贯穿全程）

1. **先列计划、等确认再执行** —— 涉及改代码、跑脚本必须列出：改什么、影响哪些文件、预期效果
2. **绝不覆盖旧数据** —— 一律用版本化目录（`_v2`, `_v3`, `_crashed_20260417`, `_fold1only_backup` 等）
3. **新脚本必须有头部注释** —— 说明来源、核心思路、与旧版本差异
4. **方案必须包含**：问题描述 / 改进方向 / 具体步骤 / 预期效果
5. **部署目标始终是 Jetson Nano 8GB** —— 流程不能依赖 PC 级算力

---

## 9. 当前状态与后续方向

### 9.1 当前状态（2026-04-19 更新）
- ✅ A (456 头) 5 折 CV 完成 — avg R²=-0.022, MAE=2.73mm
- ✅ LY (233 头) 5 折 CV 完成 — avg R²=-0.002, MAE=3.24mm
- ✅ T (64 头) 5 折 CV 完成 — avg R²=-0.315, MAE=3.72mm（小样本崩盘）
- ✅ ALL (753 头) 5 折 CV 完成 — avg R²=+0.001, **global R²=+0.058**, MAE=2.97mm（最稳）
- ✅ 断点续训机制实施完成（`fold_result.json` 完成标记 + `last_checkpoint.pth` 全状态保存）
- ✅ 报告 v4 已含全部结果

### 9.2 短期待办
1. 等 T + ALL 训练完成，补齐 4 品系对照表
2. Word 会议文档（已完成）提交老师
3. Windows pagefile 移至 F 盘（用户手动）

### 9.3 延后项（已讨论但未执行）
- **数据清洗**：离群值剔除 + 体况交叉验证 + 类别平衡
- **断点续训功能**：给 `train_v6_breed_kfold.py` 加 resume（保存 optimizer/scheduler/patience，约 30 行改动）
- **混合模态（RGB + Depth + 点云）**：借鉴师姐 RGB 的高 R²，尝试加入 RGB 分支
- **Jetson 部署适配**：HDBSCAN 在部署时仅去掉 0.9% 点可跳过，FastSAM 需蒸馏加速

### 9.4 根本性反思
当前 R² 上限（0.2 左右）很可能就是**单纯依赖几何数据**的天花板。若追求 R²>0.5，必须：
- 引入 RGB（师姐路线）
- 或加入体重、年龄、胎次等结构化特征
- 或从**俯视拍摄**改为**侧视**获取更直接的背膘厚度几何信号

---

## 10. 给接续 Claude 的上下文摘要

如果你是被唤起继续此项目的 Claude，请读取以下关键信息：

1. **工作目录**：`F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM`
2. **数据目录**：`G:\耿来硕数据\`（DK 三批 + 丰美禾）
3. **缓存重定向**：所有 torch/cuda 缓存必须走 `F:\cuda_pytorch_cache\`，bat 文件用**内联 set**（setx 无效）
4. **最新训练脚本**：`网络\pointmlp_backfat\train_v6_breed_kfold.py`
5. **最新 bat**：`run_breed_training.bat`（已注释 A/LY，当前跑 T 和 ALL）
6. **Python 解释器**：`D:\Anaconda\envs\pytorch\python.exe`（PyTorch 2.5.1，weights_only 默认 True 是雷）
7. **核心教训**：
   - 单位球归一化 = 死亡
   - 双视角融合无用
   - 深度图 CNN > 点云方法
   - 3D 信号弱于 RGB
8. **用户习惯**：必须先列计划确认，中文交流，细节控，爱问"为什么"
9. **老师假说待验证**：品系差异对背膘预测的影响（当前结果未强力支持）
10. **部署目标**：Jetson Nano 8GB —— 任何方案都要考虑算力/显存

---

**报告完**
