"""
传统机器学习背膘回归训练脚本 v2
================================
背景：v1 XGBoost 在189头猪（unit-sphere归一化点云）上仅取得 R2=0.15，传统ML和PointMLP
均严重过拟合，主要原因是样本量太少（189头）且点云预处理不佳（unit-sphere归一化丢失尺度信息）。

改进：现使用 v2 处理后的点云（去除头部、仅保留中心区域、保留真实尺度），数据量扩展到916+头猪
（4个数据源），预期从更多数据和更好的预处理两方面获得提升。

本脚本属于"方案三 v2"的第二步（模型训练），前置条件是 extract_features_v2.py 已经跑完，
生成了 traditional_ml_v2/features_per_pig.csv。

与 v1 的区别：
  - 读取 traditional_ml_v2/features_per_pig.csv（v2点云特征）
  - CSV中split字段为'unknown'（extract_features_v2.py不含划分信息），因此本脚本自行按
    pig_id 做 80%/10%/10% 的 train/val/test 划分（seed=42），保证同一头猪的所有帧在同一集
  - 划分信息保存到 traditional_ml_v2/split_info_v2.json
  - 对比总结中同时列出 v1 传统ML、PointMLP v2/v3 的结果

输入：traditional_ml_v2/features_per_pig.csv
输出：traditional_ml_v2/ 目录下的模型结果、图表、日志

Author: 耿来硕 + Claude
Date: 2026-03-25
"""

import numpy as np
import os
import json
import time
import csv
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: xgboost 未安装，将跳过 XGBoost 模型。可通过 pip install xgboost 安装。")


# ==============================================================================
# 工具函数
# ==============================================================================

def load_features_csv(csv_path):
    """
    读取 features_per_pig.csv，返回:
        data: list of dict, 每个 dict = {pig_id, split, backfat, n_frames, feat1, feat2, ...}
        feature_names: list of str
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
        feature_names = [col for col in reader.fieldnames
                         if col not in ('pig_id', 'split', 'backfat', 'n_frames')]
    return data, feature_names


def split_by_pig_id(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    按 pig_id 进行 train/val/test 划分。
    同一头猪的所有帧都在同一集中。
    返回: train_data, val_data, test_data, split_info_dict
    """
    rng = np.random.RandomState(seed)

    # 获取所有唯一的 pig_id
    pig_ids = sorted(list(set(d['pig_id'] for d in data)))
    n_pigs = len(pig_ids)

    # 随机打乱
    rng.shuffle(pig_ids)

    # 计算划分点
    n_train = int(n_pigs * train_ratio)
    n_val = int(n_pigs * val_ratio)
    # 剩余归 test

    train_ids = set(pig_ids[:n_train])
    val_ids = set(pig_ids[n_train:n_train + n_val])
    test_ids = set(pig_ids[n_train + n_val:])

    train_data = [d for d in data if d['pig_id'] in train_ids]
    val_data = [d for d in data if d['pig_id'] in val_ids]
    test_data = [d for d in data if d['pig_id'] in test_ids]

    split_info = {
        'seed': seed,
        'total_pigs': n_pigs,
        'train_pigs': len(train_ids),
        'val_pigs': len(val_ids),
        'test_pigs': len(test_ids),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'train_pig_ids': sorted(list(train_ids)),
        'val_pig_ids': sorted(list(val_ids)),
        'test_pig_ids': sorted(list(test_ids))
    }

    return train_data, val_data, test_data, split_info


def compute_metrics(y_true, y_pred):
    """计算回归指标。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                          / (np.array(y_true) + 1e-8))) * 100
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'R2': round(r2, 4), 'MAPE': round(mape, 2)}


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'traditional_ml_v2')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'training_log_traditional_v2.txt')

    def log(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("传统ML背膘回归训练 v2（v2点云特征，916+头猪）")
    log("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, 'features_per_pig.csv')
    if not os.path.exists(csv_path):
        log(f"错误: 未找到 {csv_path}")
        log("请先运行 extract_features_v2.py 提取特征")
        return

    data, feature_names = load_features_csv(csv_path)
    log(f"加载了 {len(data)} 头猪, {len(feature_names)} 个特征")

    # 过滤异常背膘值（正常范围 5-40mm）
    BACKFAT_MIN, BACKFAT_MAX = 5.0, 40.0
    n_before = len(data)
    outliers = [d for d in data if not (BACKFAT_MIN <= float(d['backfat']) <= BACKFAT_MAX)]
    data = [d for d in data if BACKFAT_MIN <= float(d['backfat']) <= BACKFAT_MAX]
    if outliers:
        log(f"过滤掉 {len(outliers)} 条异常背膘值 (范围外 [{BACKFAT_MIN}, {BACKFAT_MAX}]mm):")
        for o in outliers:
            log(f"  {o['pig_id']}: {o['backfat']}mm")
    log(f"过滤后剩余 {len(data)} 头猪 (移除 {n_before - len(data)} 条)")

    # ------------------------------------------------------------------
    # 2. 按 pig_id 做 80/10/10 划分（v2 CSV 无 split 信息）
    # ------------------------------------------------------------------
    train_data, val_data, test_data, split_info = split_by_pig_id(
        data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # 保存划分信息
    split_info_path = os.path.join(output_dir, 'split_info_v2.json')
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    log(f"划分信息保存到: {split_info_path}")

    log(f"训练集: {len(train_data)} 头猪")
    log(f"验证集: {len(val_data)} 头猪")
    log(f"测试集: {len(test_data)} 头猪")

    if not train_data or not test_data:
        log("错误: 训练集或测试集为空")
        return

    def to_xy(subset):
        X = np.array([[float(d[fn]) for fn in feature_names] for d in subset])
        y = np.array([float(d['backfat']) for d in subset])
        ids = [d['pig_id'] for d in subset]
        return X, y, ids

    X_train, y_train, ids_train = to_xy(train_data)
    X_val, y_val, ids_val = to_xy(val_data)
    X_test, y_test, ids_test = to_xy(test_data)

    # 合并 train+val 用于交叉验证（测试集保持不变）
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    log(f"特征矩阵: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    log(f"背膘范围: train={y_train.min():.1f}-{y_train.max():.1f}mm, "
        f"test={y_test.min():.1f}-{y_test.max():.1f}mm")

    # ------------------------------------------------------------------
    # 3. 特征标准化
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_trainval_s = scaler.fit_transform(X_trainval)
    # 注意：XGBoost/RF 不需要标准化，但 Lasso 需要

    # ------------------------------------------------------------------
    # 4. 训练多个模型
    # ------------------------------------------------------------------
    results_all = {}
    predictions = {}  # {model_name: (y_pred_test, y_test)}

    # ---- 4a. XGBoost ----
    if HAS_XGBOOST:
        log("\n--- XGBoost 回归 ---")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            verbosity=0
        )

        # 5折交叉验证（在 train+val 上）
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, X_trainval, y_trainval,
                                     cv=cv, scoring='r2')
        log(f"  5折CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        log(f"  各折: {[f'{s:.4f}' for s in cv_scores]}")

        cv_mae = -cross_val_score(xgb_model, X_trainval, y_trainval,
                                   cv=cv, scoring='neg_mean_absolute_error')
        log(f"  5折CV MAE: {cv_mae.mean():.4f} +/- {cv_mae.std():.4f}")

        # 在 train+val 上训练，test 上评估
        xgb_model.fit(X_trainval, y_trainval)
        y_pred_test = xgb_model.predict(X_test)
        y_pred_train = xgb_model.predict(X_trainval)

        test_metrics = compute_metrics(y_test, y_pred_test)
        train_metrics = compute_metrics(y_trainval, y_pred_train)
        log(f"  训练集: {train_metrics}")
        log(f"  测试集: {test_metrics}")

        results_all['xgboost'] = {
            'cv_r2_mean': round(cv_scores.mean(), 4),
            'cv_r2_std': round(cv_scores.std(), 4),
            'cv_mae_mean': round(cv_mae.mean(), 4),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        predictions['XGBoost'] = (y_pred_test, y_test, ids_test)

        # 特征重要性
        importance = xgb_model.feature_importances_
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: -x[1])
        log("  Top 10 特征重要性:")
        for fname, imp in feat_imp[:10]:
            log(f"    {fname}: {imp:.4f}")

        # 保存特征重要性
        with open(os.path.join(output_dir, 'feature_importance_xgb.json'), 'w') as f:
            json.dump({fn: float(imp) for fn, imp in feat_imp}, f, indent=2)

    # ---- 4b. 随机森林 ----
    log("\n--- 随机森林回归 ---")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_rf = cross_val_score(rf_model, X_trainval, y_trainval,
                                    cv=cv, scoring='r2')
    log(f"  5折CV R2: {cv_scores_rf.mean():.4f} +/- {cv_scores_rf.std():.4f}")

    cv_mae_rf = -cross_val_score(rf_model, X_trainval, y_trainval,
                                  cv=cv, scoring='neg_mean_absolute_error')
    log(f"  5折CV MAE: {cv_mae_rf.mean():.4f} +/- {cv_mae_rf.std():.4f}")

    rf_model.fit(X_trainval, y_trainval)
    y_pred_rf_test = rf_model.predict(X_test)
    y_pred_rf_train = rf_model.predict(X_trainval)

    test_metrics_rf = compute_metrics(y_test, y_pred_rf_test)
    train_metrics_rf = compute_metrics(y_trainval, y_pred_rf_train)
    log(f"  训练集: {train_metrics_rf}")
    log(f"  测试集: {test_metrics_rf}")

    results_all['random_forest'] = {
        'cv_r2_mean': round(cv_scores_rf.mean(), 4),
        'cv_r2_std': round(cv_scores_rf.std(), 4),
        'cv_mae_mean': round(cv_mae_rf.mean(), 4),
        'train_metrics': train_metrics_rf,
        'test_metrics': test_metrics_rf
    }
    predictions['RandomForest'] = (y_pred_rf_test, y_test, ids_test)

    # RF 特征重要性
    importance_rf = rf_model.feature_importances_
    feat_imp_rf = sorted(zip(feature_names, importance_rf), key=lambda x: -x[1])
    log("  Top 10 特征重要性:")
    for fname, imp in feat_imp_rf[:10]:
        log(f"    {fname}: {imp:.4f}")

    # ---- 4c. Lasso 回归 ----
    log("\n--- Lasso 回归 ---")
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_trainval_s, y_trainval)
    log(f"  最优 alpha: {lasso_cv.alpha_:.6f}")

    lasso_model = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    cv_scores_lasso = cross_val_score(lasso_model, X_trainval_s, y_trainval,
                                       cv=cv, scoring='r2')
    log(f"  5折CV R2: {cv_scores_lasso.mean():.4f} +/- {cv_scores_lasso.std():.4f}")

    cv_mae_lasso = -cross_val_score(lasso_model, X_trainval_s, y_trainval,
                                     cv=cv, scoring='neg_mean_absolute_error')
    log(f"  5折CV MAE: {cv_mae_lasso.mean():.4f} +/- {cv_mae_lasso.std():.4f}")

    lasso_model.fit(X_trainval_s, y_trainval)
    y_pred_lasso_test = lasso_model.predict(X_test_s)
    y_pred_lasso_train = lasso_model.predict(X_trainval_s)

    test_metrics_lasso = compute_metrics(y_test, y_pred_lasso_test)
    train_metrics_lasso = compute_metrics(y_trainval, y_pred_lasso_train)
    log(f"  训练集: {train_metrics_lasso}")
    log(f"  测试集: {test_metrics_lasso}")

    # Lasso 系数（非零=重要特征）
    nonzero_coefs = [(fn, c) for fn, c in zip(feature_names, lasso_model.coef_) if abs(c) > 1e-6]
    nonzero_coefs.sort(key=lambda x: -abs(x[1]))
    log(f"  Lasso 选中的非零特征: {len(nonzero_coefs)}/{len(feature_names)}")
    for fname, coef in nonzero_coefs[:10]:
        log(f"    {fname}: {coef:.4f}")

    results_all['lasso'] = {
        'cv_r2_mean': round(cv_scores_lasso.mean(), 4),
        'cv_r2_std': round(cv_scores_lasso.std(), 4),
        'cv_mae_mean': round(cv_mae_lasso.mean(), 4),
        'train_metrics': train_metrics_lasso,
        'test_metrics': test_metrics_lasso,
        'alpha': float(lasso_cv.alpha_),
        'n_nonzero_features': len(nonzero_coefs)
    }
    predictions['Lasso'] = (y_pred_lasso_test, y_test, ids_test)

    # ------------------------------------------------------------------
    # 5. 保存结果
    # ------------------------------------------------------------------
    with open(os.path.join(output_dir, 'results_all_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    log(f"\n所有结果保存到: {os.path.join(output_dir, 'results_all_v2.json')}")

    # ------------------------------------------------------------------
    # 6. 绘图
    # ------------------------------------------------------------------
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 5))

    # 各模型的散点图
    for i, (model_name, (y_pred, y_true, ids)) in enumerate(predictions.items()):
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        # 标注每头猪的ID
        for j, pid in enumerate(ids):
            short_id = pid.split('_')[-1] if '_' in pid else pid
            ax.annotate(short_id, (y_true[j], y_pred[j]), fontsize=6, alpha=0.6)

        min_val = min(y_true.min(), y_pred.min()) - 1
        max_val = max(y_true.max(), y_pred.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x', alpha=0.7)
        metrics = compute_metrics(y_true, y_pred)
        ax.set_title(f'{model_name}\nR2={metrics["R2"]:.3f}, MAE={metrics["MAE"]:.2f}mm')
        ax.set_xlabel('True Backfat (mm)')
        ax.set_ylabel('Predicted Backfat (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 特征重要性图（用 XGBoost 或 RF 的）
    ax_imp = axes[-1]
    if HAS_XGBOOST:
        imp_data = feat_imp[:15]
        title = 'XGBoost Feature Importance (Top 15)'
    else:
        imp_data = feat_imp_rf[:15]
        title = 'RF Feature Importance (Top 15)'

    fnames_plot = [x[0] for x in imp_data][::-1]
    fvals_plot = [x[1] for x in imp_data][::-1]
    ax_imp.barh(range(len(fnames_plot)), fvals_plot, color='steelblue')
    ax_imp.set_yticks(range(len(fnames_plot)))
    ax_imp.set_yticklabels(fnames_plot, fontsize=8)
    ax_imp.set_xlabel('Importance')
    ax_imp.set_title(title)
    ax_imp.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'traditional_ml_results_v2.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"结果图保存到: {fig_path}")

    # ------------------------------------------------------------------
    # 7. 逐猪预测详情（测试集）
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("测试集逐猪预测详情（XGBoost）:" if HAS_XGBOOST else "测试集逐猪预测详情（RF）:")
    log(f"{'猪ID':<25} {'真实(mm)':>8} {'预测(mm)':>8} {'误差(mm)':>8}")
    log("-" * 55)

    if HAS_XGBOOST:
        y_pred_detail, y_true_detail, ids_detail = predictions['XGBoost']
    else:
        y_pred_detail, y_true_detail, ids_detail = predictions['RandomForest']

    for pid, yt, yp in sorted(zip(ids_detail, y_true_detail, y_pred_detail)):
        err = abs(yt - yp)
        flag = " <-- 大误差" if err > 3.0 else ""
        log(f"  {pid:<25} {yt:>8.1f} {yp:>8.1f} {err:>8.1f}{flag}")

    # ------------------------------------------------------------------
    # 8. 与 v1 传统ML 及 PointMLP v2/v3 对比总结
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("方法对比总结（v1 vs v2）")
    log("=" * 60)
    log(f"{'方法':<25} {'数据版本':<10} {'猪数':<8} {'测试R2':>10} {'测试MAE':>10} {'测试RMSE':>10}")
    log("-" * 78)

    # v1 历史结果（硬编码参考值）
    log(f"{'PointMLP-Elite v2':<25} {'v1':<10} {'189':<8} {'-0.0597':>10} {'2.72mm':>10} {'3.28mm':>10}")
    log(f"{'PointMLP v3':<25} {'v1':<10} {'189':<8} {'~0':>10} {'~2.7mm':>10} {'~3.3mm':>10}")
    log(f"{'XGBoost v1':<25} {'v1':<10} {'189':<8} {'0.15':>10} {'2.38mm':>10} {'2.94mm':>10}")
    log(f"{'RandomForest v1':<25} {'v1':<10} {'189':<8} {'~0.10':>10} {'~2.5mm':>10} {'~3.0mm':>10}")
    log(f"{'Lasso v1':<25} {'v1':<10} {'189':<8} {'~0.05':>10} {'~2.6mm':>10} {'~3.1mm':>10}")
    log("-" * 78)

    # v2 本次结果
    n_pigs_v2 = split_info['total_pigs']
    for model_name in ['xgboost', 'random_forest', 'lasso']:
        if model_name in results_all:
            m = results_all[model_name]['test_metrics']
            display_name = {'xgboost': 'XGBoost v2', 'random_forest': 'RandomForest v2',
                           'lasso': 'Lasso v2'}[model_name]
            log(f"  {display_name:<23} {'v2':<10} {str(n_pigs_v2):<8} "
                f"{m['R2']:>10.4f} {m['MAE']:>9.2f}mm {m['RMSE']:>9.2f}mm")

    log("\n注: v1 结果来自189头猪 + unit-sphere归一化点云")
    log("    v2 结果来自916+头猪 + 去头/中心区域/保留尺度的点云")
    log("    v1 参考值中带 ~ 的为近似值")

    log("\n" + "=" * 60)
    log("训练完成！")
    log("=" * 60)


if __name__ == '__main__':
    main()
