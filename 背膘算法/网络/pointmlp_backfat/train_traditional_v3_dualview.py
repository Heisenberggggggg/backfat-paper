"""
双视角早期融合背膘回归训练脚本 v3
==================================
背景：
    单视角（背部）XGBoost/PointMLP 在 914 头猪上取得 R²≈0.08~0.15，效果有限。
    单一背部视角的点云外表面信息不足以准确反映皮下背膘厚度。

问题分析：
    背膘是皮下脂肪层的厚度，从外表面形状差异微妙。
    臀部视角包含互补信息（臀部丰满度、骨骼突出程度），与体况评分高度相关。

解决思路：
    双视角早期融合（Feature Concatenation）：
    1. 背部点云 → 41维统计特征（已有 features_per_pig.csv）
    2. 臀部点云 → 41维统计特征（features_rear_per_pig.csv）
    3. 拼接成 82 维向量 → XGBoost/RF/Lasso 回归
    这是验证臀部视角是否有用的最快方式。

与 v2 的区别：
    - 输入: 背部41维 + 臀部41维 = 82维特征（vs v2仅背部41维）
    - 同时对比: 单视角背部、单视角臀部、双视角融合
    - 过滤异常背膘值（5-40mm范围外）

前置条件：
    - extract_features_v2.py 已跑完 → traditional_ml_v2/features_per_pig.csv
    - extract_features_v3_rear.py 已跑完 → traditional_ml_v3/features_rear_per_pig.csv

输出：traditional_ml_v3/ 目录下的模型结果、图表、日志

Author: 耿来硕 + Claude
Date: 2026-03-31
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
    """读取 features_per_pig.csv，返回 {pig_id: {backfat, features_dict}}"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = {}
        feature_names = [col for col in reader.fieldnames
                         if col not in ('pig_id', 'split', 'backfat', 'n_frames')]
        for row in reader:
            pid = row['pig_id']
            data[pid] = {
                'backfat': float(row['backfat']),
                'features': {fn: float(row[fn]) for fn in feature_names}
            }
    return data, feature_names


def split_by_pig_id(pig_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """按 pig_id 做 train/val/test 划分，返回集合"""
    rng = np.random.RandomState(seed)
    ids = sorted(list(pig_ids))
    n = len(ids)
    rng.shuffle(ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    return train_ids, val_ids, test_ids


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred))
                          / (np.array(y_true) + 1e-8))) * 100
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4),
            'R2': round(r2, 4), 'MAPE': round(mape, 2)}


def train_and_eval(X_trainval, y_trainval, X_test, y_test, ids_test, log_fn, label=""):
    """训练 XGBoost/RF/Lasso 并返回结果字典"""
    results = {}
    predictions = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # XGBoost
    if HAS_XGBOOST:
        log_fn(f"\n--- {label} XGBoost ---")
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, min_child_weight=3, random_state=42, verbosity=0
        )
        cv_r2 = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring='r2')
        cv_mae = -cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring='neg_mean_absolute_error')
        log_fn(f"  5折CV R2: {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
        log_fn(f"  5折CV MAE: {cv_mae.mean():.4f} +/- {cv_mae.std():.4f}")

        model.fit(X_trainval, y_trainval)
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_trainval)

        test_m = compute_metrics(y_test, y_pred)
        train_m = compute_metrics(y_trainval, y_pred_train)
        log_fn(f"  训练集: {train_m}")
        log_fn(f"  测试集: {test_m}")

        results['xgboost'] = {
            'cv_r2': round(cv_r2.mean(), 4), 'test': test_m, 'train': train_m
        }
        predictions['XGBoost'] = (y_pred, y_test, ids_test)

        # 特征重要性
        importance = model.feature_importances_
        feat_imp = sorted(zip(range(len(importance)), importance), key=lambda x: -x[1])
        log_fn(f"  Top 10 特征重要性 (index: importance):")
        for idx, imp in feat_imp[:10]:
            log_fn(f"    [{idx}]: {imp:.4f}")
        results['xgboost']['top_features'] = feat_imp[:10]

    # RandomForest
    log_fn(f"\n--- {label} RandomForest ---")
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=6, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=-1
    )
    cv_r2_rf = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='r2')
    cv_mae_rf = -cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='neg_mean_absolute_error')
    log_fn(f"  5折CV R2: {cv_r2_rf.mean():.4f} +/- {cv_r2_rf.std():.4f}")

    rf.fit(X_trainval, y_trainval)
    y_pred_rf = rf.predict(X_test)
    test_m_rf = compute_metrics(y_test, y_pred_rf)
    train_m_rf = compute_metrics(y_trainval, rf.predict(X_trainval))
    log_fn(f"  训练集: {train_m_rf}")
    log_fn(f"  测试集: {test_m_rf}")

    results['random_forest'] = {
        'cv_r2': round(cv_r2_rf.mean(), 4), 'test': test_m_rf, 'train': train_m_rf
    }
    predictions['RandomForest'] = (y_pred_rf, y_test, ids_test)

    # Lasso
    log_fn(f"\n--- {label} Lasso ---")
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_trainval)
    X_te_s = scaler.transform(X_test)

    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_tv_s, y_trainval)
    log_fn(f"  最优 alpha: {lasso_cv.alpha_:.6f}")

    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    cv_r2_lasso = cross_val_score(lasso, X_tv_s, y_trainval, cv=cv, scoring='r2')
    log_fn(f"  5折CV R2: {cv_r2_lasso.mean():.4f} +/- {cv_r2_lasso.std():.4f}")

    lasso.fit(X_tv_s, y_trainval)
    y_pred_lasso = lasso.predict(X_te_s)
    test_m_lasso = compute_metrics(y_test, y_pred_lasso)
    train_m_lasso = compute_metrics(y_trainval, lasso.predict(X_tv_s))
    log_fn(f"  训练集: {train_m_lasso}")
    log_fn(f"  测试集: {test_m_lasso}")

    results['lasso'] = {
        'cv_r2': round(cv_r2_lasso.mean(), 4), 'test': test_m_lasso, 'train': train_m_lasso
    }
    predictions['Lasso'] = (y_pred_lasso, y_test, ids_test)

    return results, predictions


# ==============================================================================
# 主流程
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'traditional_ml_v3')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'training_log_dualview.txt')

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
    log("双视角早期融合背膘回归训练 v3")
    log("背部41维 + 臀部41维 = 82维特征 → XGBoost/RF/Lasso")
    log("=" * 60)

    # ------------------------------------------------------------------
    # 1. 加载背部特征
    # ------------------------------------------------------------------
    back_csv = os.path.join(script_dir, 'traditional_ml_v2', 'features_per_pig.csv')
    if not os.path.exists(back_csv):
        log(f"错误: 未找到背部特征文件 {back_csv}")
        return
    back_data, back_feat_names = load_features_csv(back_csv)
    log(f"背部特征: {len(back_data)} 头猪, {len(back_feat_names)} 维")

    # ------------------------------------------------------------------
    # 2. 加载臀部特征
    # ------------------------------------------------------------------
    rear_csv = os.path.join(output_dir, 'features_rear_per_pig.csv')
    if not os.path.exists(rear_csv):
        log(f"错误: 未找到臀部特征文件 {rear_csv}")
        log("请先运行 extract_features_v3_rear.py 提取臀部视角特征")
        return
    rear_data, rear_feat_names = load_features_csv(rear_csv)
    log(f"臀部特征: {len(rear_data)} 头猪, {len(rear_feat_names)} 维")

    # ------------------------------------------------------------------
    # 3. 取交集（同时有背部和臀部特征的猪）
    # ------------------------------------------------------------------
    common_pigs = set(back_data.keys()) & set(rear_data.keys())
    log(f"双视角交集: {len(common_pigs)} 头猪 (背部{len(back_data)}, 臀部{len(rear_data)})")

    # 过滤异常背膘值
    BACKFAT_MIN, BACKFAT_MAX = 5.0, 40.0
    valid_pigs = []
    outliers = []
    for pid in sorted(common_pigs):
        bf = back_data[pid]['backfat']
        if BACKFAT_MIN <= bf <= BACKFAT_MAX:
            valid_pigs.append(pid)
        else:
            outliers.append((pid, bf))

    if outliers:
        log(f"过滤掉 {len(outliers)} 条异常背膘值 (范围外 [{BACKFAT_MIN}, {BACKFAT_MAX}]mm):")
        for pid, bf in outliers:
            log(f"  {pid}: {bf}mm")
    log(f"有效猪数: {len(valid_pigs)}")

    # ------------------------------------------------------------------
    # 4. 构建特征矩阵
    # ------------------------------------------------------------------
    # 背部特征名加前缀 back_, 臀部加前缀 rear_
    back_names_prefixed = [f'back_{fn}' for fn in back_feat_names]
    rear_names_prefixed = [f'rear_{fn}' for fn in rear_feat_names]
    dual_feat_names = back_names_prefixed + rear_names_prefixed

    # 构建数据
    X_back_all = np.array([[back_data[pid]['features'][fn] for fn in back_feat_names] for pid in valid_pigs])
    X_rear_all = np.array([[rear_data[pid]['features'][fn] for fn in rear_feat_names] for pid in valid_pigs])
    X_dual_all = np.hstack([X_back_all, X_rear_all])
    y_all = np.array([back_data[pid]['backfat'] for pid in valid_pigs])

    log(f"特征矩阵: 背部={X_back_all.shape}, 臀部={X_rear_all.shape}, 融合={X_dual_all.shape}")
    log(f"背膘范围: {y_all.min():.1f}-{y_all.max():.1f}mm, 均值={y_all.mean():.1f}mm")

    # ------------------------------------------------------------------
    # 5. 数据划分（同 v2 保持一致的 seed=42）
    # ------------------------------------------------------------------
    train_ids, val_ids, test_ids = split_by_pig_id(valid_pigs, seed=42)

    def get_split_indices(ids_set):
        return [i for i, pid in enumerate(valid_pigs) if pid in ids_set]

    train_idx = get_split_indices(train_ids)
    val_idx = get_split_indices(val_ids)
    test_idx = get_split_indices(test_ids)
    trainval_idx = train_idx + val_idx

    log(f"划分: 训练={len(train_idx)}, 验证={len(val_idx)}, 测试={len(test_idx)}")

    # 保存划分信息
    split_info = {
        'seed': 42, 'total_pigs': len(valid_pigs),
        'train_pigs': len(train_idx), 'val_pigs': len(val_idx), 'test_pigs': len(test_idx),
        'train_pig_ids': sorted([valid_pigs[i] for i in train_idx]),
        'val_pig_ids': sorted([valid_pigs[i] for i in val_idx]),
        'test_pig_ids': sorted([valid_pigs[i] for i in test_idx])
    }
    with open(os.path.join(output_dir, 'split_info_v3.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    ids_test = [valid_pigs[i] for i in test_idx]

    # ------------------------------------------------------------------
    # 6. 三组实验对比
    # ------------------------------------------------------------------
    all_results = {}

    # 实验1: 单视角背部 (41维)
    log("\n" + "=" * 60)
    log("实验1: 单视角背部 (41维)")
    log("=" * 60)
    r1, p1 = train_and_eval(
        X_back_all[trainval_idx], y_all[trainval_idx],
        X_back_all[test_idx], y_all[test_idx], ids_test, log, "背部"
    )
    all_results['back_only'] = r1

    # 实验2: 单视角臀部 (41维)
    log("\n" + "=" * 60)
    log("实验2: 单视角臀部 (41维)")
    log("=" * 60)
    r2, p2 = train_and_eval(
        X_rear_all[trainval_idx], y_all[trainval_idx],
        X_rear_all[test_idx], y_all[test_idx], ids_test, log, "臀部"
    )
    all_results['rear_only'] = r2

    # 实验3: 双视角融合 (82维)
    log("\n" + "=" * 60)
    log("实验3: 双视角融合 (82维)")
    log("=" * 60)
    r3, p3 = train_and_eval(
        X_dual_all[trainval_idx], y_all[trainval_idx],
        X_dual_all[test_idx], y_all[test_idx], ids_test, log, "融合"
    )
    all_results['dual_view'] = r3

    # ------------------------------------------------------------------
    # 7. 对比总结
    # ------------------------------------------------------------------
    log("\n" + "=" * 60)
    log("三组实验对比总结")
    log("=" * 60)
    log(f"{'实验':<20} {'模型':<12} {'特征维度':<8} {'CV R2':>8} {'Test R2':>8} {'Test MAE':>10} {'Test RMSE':>10}")
    log("-" * 80)

    for exp_name, exp_label, n_dim in [
        ('back_only', '背部单视角', 41),
        ('rear_only', '臀部单视角', 41),
        ('dual_view', '双视角融合', 82)
    ]:
        for model_name in ['xgboost', 'random_forest', 'lasso']:
            if model_name in all_results[exp_name]:
                r = all_results[exp_name][model_name]
                display = {'xgboost': 'XGBoost', 'random_forest': 'RF', 'lasso': 'Lasso'}[model_name]
                log(f"  {exp_label:<18} {display:<12} {n_dim:<8} "
                    f"{r['cv_r2']:>8.4f} {r['test']['R2']:>8.4f} "
                    f"{r['test']['MAE']:>9.2f}mm {r['test']['RMSE']:>9.2f}mm")
        log("-" * 80)

    # 判断双视角是否有效
    best_single = max(
        all_results['back_only'].get('xgboost', {}).get('test', {}).get('R2', -999),
        all_results['back_only'].get('random_forest', {}).get('test', {}).get('R2', -999),
        all_results['rear_only'].get('xgboost', {}).get('test', {}).get('R2', -999),
        all_results['rear_only'].get('random_forest', {}).get('test', {}).get('R2', -999),
    )
    best_dual = max(
        all_results['dual_view'].get('xgboost', {}).get('test', {}).get('R2', -999),
        all_results['dual_view'].get('random_forest', {}).get('test', {}).get('R2', -999),
    )

    log(f"\n最佳单视角 Test R2: {best_single:.4f}")
    log(f"最佳双视角 Test R2: {best_dual:.4f}")
    if best_dual > best_single + 0.02:
        log("结论: 双视角融合有明显提升，臀部视角提供了有价值的额外信息！")
        log("建议: 继续探索中期融合（双分支PointMLP）")
    elif best_dual > best_single:
        log("结论: 双视角略有提升，但幅度不大")
        log("建议: 可以尝试中期融合，也考虑其他改进方向")
    else:
        log("结论: 双视角未带来提升，臀部视角可能不包含额外的背膘相关信息")
        log("建议: 考虑其他改进方向（如深度图CNN、多帧时序建模等）")

    # ------------------------------------------------------------------
    # 8. 保存结果
    # ------------------------------------------------------------------
    # 清理不可序列化的内容
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {}
        for mk, mv in v.items():
            save_results[k][mk] = {kk: vv for kk, vv in mv.items() if kk != 'top_features'}

    with open(os.path.join(output_dir, 'results_dualview.json'), 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果保存到: {os.path.join(output_dir, 'results_dualview.json')}")

    # ------------------------------------------------------------------
    # 9. 绘图：三组实验 × XGBoost 散点图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    all_preds = [
        ('Back Only (41d)', p1),
        ('Rear Only (41d)', p2),
        ('Dual View (82d)', p3),
    ]
    for ax, (title, preds_dict) in zip(axes, all_preds):
        # 用 XGBoost 或 RF
        if 'XGBoost' in preds_dict:
            y_pred, y_true, ids = preds_dict['XGBoost']
            model_label = 'XGBoost'
        else:
            y_pred, y_true, ids = preds_dict['RandomForest']
            model_label = 'RF'

        ax.scatter(y_true, y_pred, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        min_val = min(y_true.min(), y_pred.min()) - 1
        max_val = max(y_true.max(), y_pred.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        m = compute_metrics(y_true, y_pred)
        ax.set_title(f'{title}\n{model_label}: R2={m["R2"]:.3f}, MAE={m["MAE"]:.2f}mm')
        ax.set_xlabel('True Backfat (mm)')
        ax.set_ylabel('Predicted Backfat (mm)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'dualview_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"对比图保存到: {fig_path}")

    log("\n" + "=" * 60)
    log("训练完成！")
    log("=" * 60)


if __name__ == '__main__':
    main()
