# traditional_ml_v3 — 双视角早期融合实验

## 目录来源
- 创建时间：2026-03-31
- 创建脚本：`extract_features_v3_rear.py` + `train_traditional_v3_dualview.py`

## 问题背景
单视角（背部）XGBoost/PointMLP 在 914 头猪上 R²≈0.08~0.15，效果有限。
猜测原因：单一视角信息量不足以反映皮下背膘厚度。

## 解决思路
双视角早期融合（Feature Concatenation）：
- 背部点云 → 41维统计特征（来自 `traditional_ml_v2/features_per_pig.csv`）
- 臀部点云 → 41维统计特征（本目录 `features_rear_per_pig.csv`）
- 拼接成 82维 → XGBoost/RF/Lasso 回归

## 文件说明
| 文件 | 说明 |
|------|------|
| `features_rear_per_frame.csv` | 臀部视角帧级41维特征 |
| `features_rear_per_pig.csv` | 臀部视角猪级41维特征（帧均值） |
| `results_dualview.json` | 三组对比实验结果（背部/臀部/融合） |
| `dualview_comparison.png` | 散点图对比 |
| `split_info_v3.json` | 数据划分信息 |
| `training_log_dualview.txt` | 训练日志 |

## 实验设计
同时对比三组实验验证臀部视角是否有价值：
1. 单视角背部（41维）— 基线
2. 单视角臀部（41维）— 对比
3. 双视角融合（82维）— 验证

## 数据流水线
```
generate_pointcloud_v3_rear.py  → masked_pointclouds_v3_rear/
postprocess_pointcloud_v3_rear.py → processed_pointclouds_v3_rear/
extract_features_v3_rear.py     → traditional_ml_v3/features_rear_*.csv
train_traditional_v3_dualview.py → traditional_ml_v3/results_dualview.json
```
