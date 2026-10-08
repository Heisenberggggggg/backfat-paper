# traditional_ml_v2/ 目录说明

## 来源
由 `extract_features_v2.py` 和 `train_traditional_v2.py` 生成

## 与 v1 (traditional_ml/) 的区别

| 项目 | v1 (traditional_ml/) | v2 (本目录) |
|------|---------------------|-------------|
| 数据源 | processed_pointclouds (师兄处理, 单位球归一化) | processed_pointclouds_v2 (去头+保留尺度+中心化) |
| 猪只数 | 189头 | 916+头 (DK 163 + 丰美禾全部12个日期) |
| 标签来源 | 3个docx + 1个xlsx | 3个docx + 2个xlsx (新增11.10处理背膘数据.xlsx) |
| 每头帧数 | 全部帧(~181帧) | 均匀抽20帧 |
| 点云预处理 | 保留头+背+臀, 单位球归一化 | 去头(只保留背+臀), 仅中心化 |

## 代码变更记录
- 2026-03-25: 创建 extract_features_v2.py, train_traditional_v2.py
- 改动原因: v1使用的点云包含猪头部(与背膘无关), 且单位球归一化丢失尺寸信息, 数据量仅189头
- 解决方案: 使用v2处理后的点云(去头+保留尺度), 扩大数据到916+头
- 预期效果: 更多数据+更好的预处理 → XGBoost R²从0.15提升

## 文件列表
- `features_per_frame.csv` - 每帧一行的41维特征 (由extract_features_v2.py生成)
- `features_per_pig.csv` - 每猪一行的41维特征均值 (由extract_features_v2.py生成)
- `split_info_v2.json` - 训练/验证/测试集划分 (由train_traditional_v2.py生成)
- `results_all_v2.json` - XGBoost/RF/Lasso模型结果
- `traditional_ml_results_v2.png` - 预测散点图+特征重要性图
- `training_log_traditional_v2.txt` - 训练日志
- `extract_features_log_v2.txt` - 特征提取日志
