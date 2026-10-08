"""
PointMLP-Elite v6 — 按品系分组 + 五折交叉验证
=============================================================
基于 train_v4.py 的 PointMLP-Elite 架构，新增：
  - 只使用丰美禾数据（按品系分组训练）
  - 按耳标前缀区分品系: A(单系), LY(新美系), T(单系), ALL(全部)
  - 五折交叉验证（猪级别划分，每折 8:1:1）
  - 每折的测试猪不重复，5折合计所有猪测试一次
  - 最终报告5折平均 R², MAE, RMSE

输出: checkpoints_pointmlp_breed/{breed}/fold_{k}/
不覆盖之前任何版本的结果。

Author: 耿来硕 + Claude
Date: 2026-04-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import sys
import re
import json
import csv
import random
import datetime
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# 复用 train.py 中的基础函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (square_distance, farthest_point_sample,
                   index_points, knn_point, get_activation, compute_metrics)

# 复用 train_v4.py 中的标签加载函数
from train_v4 import (load_labels_from_xlsx_fmh, load_labels_from_xlsx_fmh_nov,
                       load_pig_features)

# 复用 train_v4.py 中的模型组件
from train_v4 import (LocalGrouper, ConvBNReLU1D, ConvBNReLURes1D,
                       PreExtraction, PosExtraction, PointMLPRegressorV3,
                       BackfatDatasetV3, train_one_epoch_v3, evaluate_v3)


# ==============================================================================
# 0. 丰美禾专用数据注册表（带品系标签）
# ==============================================================================

def build_fmh_registry_with_breed(fmh_base):
    """扫描丰美禾所有日期目录，提取品系前缀，匹配标签。
    返回 {pig_uid: (data_dir, backfat, n_frames, breed)}。
    """
    fmh_dates = ['8.24', '8.25', '8.26', '8.27', '10.29', '10.30', '10.31',
                 '11.1', '11.2', '11.3', '11.4', '11.5']

    # 加载标签
    xlsx_path = os.path.join(fmh_base, '背膘记录表.xlsx')
    fmh_entries = []
    if os.path.isfile(xlsx_path):
        fmh_entries = load_labels_from_xlsx_fmh(xlsx_path)

    xlsx_nov_path = os.path.join(fmh_base, '11.10处理背膘数据.xlsx')
    fmh_nov_entries = []
    if os.path.isfile(xlsx_nov_path):
        fmh_nov_entries = load_labels_from_xlsx_fmh_nov(xlsx_nov_path)

    all_fmh_entries = fmh_entries + fmh_nov_entries

    registry = {}
    for date_str in fmh_dates:
        date_dir = os.path.join(fmh_base, date_str)
        if not os.path.isdir(date_dir):
            continue

        for pig_dirname in os.listdir(date_dir):
            pig_path = os.path.join(date_dir, pig_dirname)
            if not os.path.isdir(pig_path):
                continue

            pp = os.path.join(pig_path, 'master', 'processed_pointclouds_v2')
            if not os.path.isdir(pp):
                continue

            files = [f for f in os.listdir(pp) if f.endswith('.txt')]
            if not files:
                continue

            # 提取品系前缀
            m = re.match(r'^\d+([A-Za-z]+)', pig_dirname)
            if not m:
                continue  # 跳过没有字母前缀的
            breed = m.group(1).upper()

            # 匹配标签
            matched = False
            for seq, tag, bf in all_fmh_entries:
                tag_clean = tag.strip()
                if tag_clean and tag_clean in pig_dirname:
                    uid = f'fmh_{date_str}_{pig_dirname}'
                    registry[uid] = (pp, bf, len(files), breed)
                    matched = True
                    break

    return registry


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser('PointMLP-Elite v6 Breed K-Fold')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--fmh_base', type=str, default=r'G:\耿来硕数据\丰美禾数据')
    parser.add_argument('--breed', type=str, default='ALL',
                        help='品系: A, LY, T, ALL')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--save_base', type=str,
                        default=r'F:\Postgraduate\Lab\饲喂站\背膘算法\网络\pointmlp_backfat\checkpoints_pointmlp_breed')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 日志
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, f'training_log_v6_{args.breed}.txt')
    # 2026-04-17 resume 改动：不再清空日志，用追加模式保留中断前的历史
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('')

    def log(msg=''):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            pass
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log("=" * 60)
    log(f"PointMLP-Elite v6 — Breed={args.breed}, {args.n_folds}-Fold CV")
    log(f"Device: {device}")
    log("=" * 60)

    # --- 构建丰美禾数据注册表 ---
    log("\n[1] Building FMH registry with breed labels...")
    full_registry = build_fmh_registry_with_breed(args.fmh_base)
    log(f"  Total FMH pigs with breed label: {len(full_registry)}")

    # 统计品系分布
    breed_counts = defaultdict(int)
    for uid, (_, _, _, breed) in full_registry.items():
        breed_counts[breed] += 1
    for b, c in sorted(breed_counts.items(), key=lambda x: -x[1]):
        log(f"    {b}: {c} pigs")

    # 过滤异常背膘值
    BACKFAT_MIN, BACKFAT_MAX = 7.0, 32.0
    registry = {}
    n_filtered = 0
    for uid, (data_dir, bf, nf, breed) in full_registry.items():
        if BACKFAT_MIN <= bf <= BACKFAT_MAX:
            registry[uid] = (data_dir, bf, nf, breed)
        else:
            n_filtered += 1
            log(f"  Filtered: {uid} bf={bf}mm")
    log(f"  After filtering [{BACKFAT_MIN}, {BACKFAT_MAX}]mm: {len(registry)} pigs ({n_filtered} removed)")

    # 按品系过滤
    if args.breed != 'ALL':
        registry = {k: v for k, v in registry.items() if v[3] == args.breed}
        log(f"  After breed filter ({args.breed}): {len(registry)} pigs")

    if len(registry) < 10:
        log(f"ERROR: Not enough data ({len(registry)} pigs). Exiting.")
        return

    # --- 加载手工特征 ---
    feat_csv = os.path.join(script_dir, 'traditional_ml_v2', 'features_per_pig.csv')
    if os.path.exists(feat_csv):
        pig_features, feat_names = load_pig_features(feat_csv)
        handcraft_dim = len(feat_names)
        log(f"  Loaded handcraft features: {len(pig_features)} pigs, {handcraft_dim} dims")
    else:
        log(f"  WARNING: {feat_csv} not found! Using zero features.")
        pig_features = {}
        handcraft_dim = 41

    # --- 五折交叉验证 ---
    all_pigs = sorted(registry.keys())
    n_pigs = len(all_pigs)
    log(f"\n[2] {args.n_folds}-Fold Cross-Validation with {n_pigs} pigs")

    # 固定种子打乱
    rng = random.Random(args.seed)
    rng.shuffle(all_pigs)

    # 划分成 n_folds 份
    fold_size = n_pigs // args.n_folds
    folds = []
    for k in range(args.n_folds):
        start = k * fold_size
        if k == args.n_folds - 1:
            end = n_pigs  # 最后一折包含剩余
        else:
            end = start + fold_size
        folds.append(all_pigs[start:end])

    log(f"  Fold sizes: {[len(f) for f in folds]}")

    # 背膘分布
    all_bfs = [registry[pid][1] for pid in all_pigs]
    log(f"  Backfat range: {min(all_bfs):.1f}-{max(all_bfs):.1f}mm, mean={np.mean(all_bfs):.1f}, std={np.std(all_bfs):.1f}")

    # 保存目录
    breed_save_dir = os.path.join(args.save_base, f'breed_{args.breed}')
    os.makedirs(breed_save_dir, exist_ok=True)

    # 保存实验配置
    config = {
        'breed': args.breed,
        'n_folds': args.n_folds,
        'n_pigs': n_pigs,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'seed': args.seed,
        'backfat_filter': [BACKFAT_MIN, BACKFAT_MAX],
        'fold_sizes': [len(f) for f in folds],
    }
    with open(os.path.join(breed_save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 收集所有折的结果
    all_fold_results = []
    all_test_preds = []
    all_test_targets = []

    for fold_idx in range(args.n_folds):
        log(f"\n{'='*60}")
        log(f"FOLD {fold_idx+1}/{args.n_folds}")
        log(f"{'='*60}")

        fold_dir = os.path.join(breed_save_dir, f'fold_{fold_idx+1}')
        os.makedirs(fold_dir, exist_ok=True)
        result_path = os.path.join(fold_dir, 'fold_result.json')
        # 2026-04-17 resume: 若本折已完成，直接加载并跳过
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    prev = json.load(f)
                log(f"  [RESUME] Fold {fold_idx+1} already done, loading cached result.")
                log(f"    MAE={prev['test_mae']:.2f}mm R2={prev['test_r2']:.4f}")
                all_fold_results.append(prev)
                # 恢复 test 预测以便汇总时重建全局指标（若文件不存在则跳过）
                preds_npz = os.path.join(fold_dir, 'test_preds.npz')
                if os.path.exists(preds_npz):
                    try:
                        d = np.load(preds_npz)
                        all_test_preds.extend(d['preds'].tolist())
                        all_test_targets.extend(d['targets'].tolist())
                    except Exception as e:
                        log(f"    [WARN] Failed to load test_preds.npz: {e}")
                continue
            except Exception as e:
                log(f"  [RESUME] Failed to load {result_path}: {e}. Retraining.")

        # 固定种子（每折不同但可复现）
        fold_seed = args.seed + fold_idx
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        # 测试集 = 当前折
        test_pigs = folds[fold_idx]

        # 剩余的猪用于训练+验证 (8:1 比例)
        remaining = []
        for k in range(args.n_folds):
            if k != fold_idx:
                remaining.extend(folds[k])

        # 从remaining中划出 1/(n_folds-1) 作为验证集 ≈ 总数的1/n_folds
        rng_fold = random.Random(fold_seed)
        rng_fold.shuffle(remaining)
        n_val = len(test_pigs)  # 验证集与测试集大小相同
        val_pigs = remaining[:n_val]
        train_pigs = remaining[n_val:]

        log(f"  Train: {len(train_pigs)} pigs")
        log(f"  Val:   {len(val_pigs)} pigs")
        log(f"  Test:  {len(test_pigs)} pigs")

        # 构建样本列表
        def make_samples(pids):
            samples = []
            for pid in pids:
                data_dir, bf, nf, breed = registry[pid]
                for fname in sorted(os.listdir(data_dir)):
                    if fname.endswith('.txt'):
                        samples.append((os.path.join(data_dir, fname), bf, pid))
            return samples

        train_samples = make_samples(train_pigs)
        val_samples = make_samples(val_pigs)
        test_samples = make_samples(test_pigs)

        log(f"  Train samples: {len(train_samples)}")
        log(f"  Val samples:   {len(val_samples)}")
        log(f"  Test samples:  {len(test_samples)}")

        # 手工特征标准化（仅用训练集统计）
        train_feat_pids = [pid for pid in train_pigs if pid in pig_features]
        if train_feat_pids:
            train_feats = np.stack([pig_features[pid] for pid in train_feat_pids])
            feat_mean = np.mean(train_feats, axis=0)
            feat_std = np.std(train_feats, axis=0) + 1e-8
            # 标准化所有猪的特征（用训练集的mean/std）
            pig_features_fold = {}
            for pid in registry:
                if pid in pig_features:
                    pig_features_fold[pid] = ((pig_features[pid] - feat_mean) / feat_std).astype(np.float32)
                else:
                    pig_features_fold[pid] = np.zeros(handcraft_dim, dtype=np.float32)
        else:
            pig_features_fold = {}

        # DataLoader
        train_dataset = BackfatDatasetV3(train_samples, pig_features_fold, args.num_points, augment=True)
        val_dataset = BackfatDatasetV3(val_samples, pig_features_fold, args.num_points, augment=False)
        test_dataset = BackfatDatasetV3(test_samples, pig_features_fold, args.num_points, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)

        # 模型
        model = PointMLPRegressorV3(
            points=args.num_points,
            embed_dim=16,
            handcraft_feat_dim=handcraft_dim,
            dropout=0.6
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        criterion = nn.SmoothL1Loss()

        # 训练循环
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        train_losses, val_losses = [], []
        train_r2s, val_r2s = [], []
        start_epoch = 1

        # 2026-04-17 resume: 若 last_checkpoint.pth 存在，恢复完整训练状态
        last_ckpt_path = os.path.join(fold_dir, 'last_checkpoint.pth')
        if os.path.exists(last_ckpt_path):
            try:
                lc = torch.load(last_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(lc['model_state_dict'])
                optimizer.load_state_dict(lc['optimizer_state_dict'])
                scheduler.load_state_dict(lc['scheduler_state_dict'])
                best_val_loss = lc['best_val_loss']
                best_epoch = lc['best_epoch']
                patience_counter = lc['patience_counter']
                train_losses = lc.get('train_losses', [])
                val_losses = lc.get('val_losses', [])
                train_r2s = lc.get('train_r2s', [])
                val_r2s = lc.get('val_r2s', [])
                start_epoch = lc['epoch'] + 1
                if 'torch_rng' in lc:
                    torch.set_rng_state(lc['torch_rng'])
                if 'cuda_rng' in lc and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(lc['cuda_rng'])
                    except Exception:
                        pass
                log(f"  [RESUME] Resumed fold {fold_idx+1} from epoch {start_epoch} "
                    f"(best_val_loss={best_val_loss:.4f} @ ep{best_epoch}, patience={patience_counter})")
            except Exception as e:
                log(f"  [RESUME] Failed to load {last_ckpt_path}: {e}. Starting fresh.")
                start_epoch = 1

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_metrics = train_one_epoch_v3(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            val_loss, val_metrics, _, _ = evaluate_v3(
                model, val_loader, criterion, device)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2s.append(train_metrics['R2'])
            val_r2s.append(val_metrics['R2'])

            lr_now = optimizer.param_groups[0]['lr']

            # 每5个epoch或改善时打印
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_r2': val_metrics['R2'],
                }, os.path.join(fold_dir, 'best_model.pth'))

            if epoch % 5 == 0 or improved or epoch == 1:
                log(f"  Epoch {epoch:>3}/{args.epochs} (lr={lr_now:.6f})")
                log(f"    Train - Loss: {train_loss:.4f} | MAE: {train_metrics['MAE']:.2f}mm | R2: {train_metrics['R2']:.4f}")
                log(f"    Val   - Loss: {val_loss:.4f} | MAE: {val_metrics['MAE']:.2f}mm | R2: {val_metrics['R2']:.4f}")
                if improved:
                    log(f"    * Best model saved!")

            if not improved:
                patience_counter += 1

            # 2026-04-17 resume: 每个 epoch 保存完整训练状态（含 optimizer/scheduler/RNG）
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_r2s': train_r2s,
                    'val_r2s': val_r2s,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }, last_ckpt_path)
            except Exception as e:
                log(f"    [WARN] Failed to save last_checkpoint: {e}")

            if not improved and patience_counter >= args.patience:
                log(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # 测试（Fix: weights_only=False 避免 PyTorch 2.5+ 严格模式崩溃；
        # 并用 try/except 保护，单折失败也不影响后续 fold）
        log(f"\n  Testing fold {fold_idx+1} with best model (epoch {best_epoch})...")
        try:
            ckpt = torch.load(os.path.join(fold_dir, 'best_model.pth'), map_location=device,
                              weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            test_loss, test_metrics, test_preds, test_targets = evaluate_v3(
                model, test_loader, criterion, device)

            log(f"  Fold {fold_idx+1} Test Results:")
            log(f"    MAE:  {test_metrics['MAE']:.2f} mm")
            log(f"    RMSE: {test_metrics['RMSE']:.2f} mm")
            log(f"    R2:   {test_metrics['R2']:.4f}")
            log(f"    MAPE: {test_metrics['MAPE']:.1f}%")

            fold_result = {
                'fold': fold_idx + 1,
                'n_train': len(train_pigs),
                'n_val': len(val_pigs),
                'n_test': len(test_pigs),
                'best_epoch': best_epoch,
                'test_loss': float(test_loss),
                'test_mae': float(test_metrics['MAE']),
                'test_rmse': float(test_metrics['RMSE']),
                'test_r2': float(test_metrics['R2']),
                'test_mape': float(test_metrics['MAPE']),
            }
            all_fold_results.append(fold_result)
            all_test_preds.extend(test_preds)
            all_test_targets.extend(test_targets)

            # 保存 test 预测（resume 时用于全局指标重建）
            try:
                np.savez(os.path.join(fold_dir, 'test_preds.npz'),
                         preds=np.asarray(test_preds), targets=np.asarray(test_targets))
            except Exception as e:
                log(f"    [WARN] Failed to save test_preds.npz: {e}")

            # 保存fold结果（"已完成"标记）
            with open(os.path.join(fold_dir, 'fold_result.json'), 'w') as f:
                json.dump(fold_result, f, indent=2)

            # 2026-04-17 resume: 本折完毕后删除 last_checkpoint 节省空间
            try:
                if os.path.exists(last_ckpt_path):
                    os.remove(last_ckpt_path)
            except Exception:
                pass
        except Exception as e:
            import traceback
            log(f"  [ERROR] Fold {fold_idx+1} testing failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            log(f"  Continuing to next fold...")

        # 清理GPU
        try:
            del model, optimizer, scheduler
        except Exception:
            pass
        torch.cuda.empty_cache()

    # ==================== 汇总结果 ====================
    log(f"\n{'='*60}")
    log(f"SUMMARY: Breed={args.breed}, {args.n_folds}-Fold CV")
    log(f"{'='*60}")

    maes = [r['test_mae'] for r in all_fold_results]
    rmses = [r['test_rmse'] for r in all_fold_results]
    r2s = [r['test_r2'] for r in all_fold_results]
    mapes = [r['test_mape'] for r in all_fold_results]

    log(f"\n  Per-fold results:")
    log(f"  {'Fold':<6} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE':>8} {'Best Ep':>8}")
    log(f"  {'-'*50}")
    for r in all_fold_results:
        log(f"  {r['fold']:<6} {r['test_mae']:>7.2f}mm {r['test_rmse']:>7.2f}mm "
            f"{r['test_r2']:>7.4f} {r['test_mape']:>7.1f}% {r['best_epoch']:>8}")

    log(f"\n  Average across {args.n_folds} folds:")
    log(f"    MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f} mm")
    log(f"    RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f} mm")
    log(f"    R2:   {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    log(f"    MAPE: {np.mean(mapes):.1f} ± {np.std(mapes):.1f}%")

    # 全局指标（所有fold的预测合并计算）
    global_metrics = compute_metrics(all_test_preds, all_test_targets)
    log(f"\n  Global metrics (all folds combined):")
    log(f"    MAE:  {global_metrics['MAE']:.2f} mm")
    log(f"    RMSE: {global_metrics['RMSE']:.2f} mm")
    log(f"    R2:   {global_metrics['R2']:.4f}")
    log(f"    MAPE: {global_metrics['MAPE']:.1f}%")

    # 保存汇总结果
    summary = {
        'breed': args.breed,
        'n_pigs': n_pigs,
        'n_folds': args.n_folds,
        'per_fold': all_fold_results,
        'avg_mae': float(np.mean(maes)),
        'avg_rmse': float(np.mean(rmses)),
        'avg_r2': float(np.mean(r2s)),
        'avg_mape': float(np.mean(mapes)),
        'std_mae': float(np.std(maes)),
        'std_rmse': float(np.std(rmses)),
        'std_r2': float(np.std(r2s)),
        'std_mape': float(np.std(mapes)),
        'global_mae': float(global_metrics['MAE']),
        'global_rmse': float(global_metrics['RMSE']),
        'global_r2': float(global_metrics['R2']),
        'global_mape': float(global_metrics['MAPE']),
    }
    with open(os.path.join(breed_save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # 绘制汇总散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    preds_arr = np.array(all_test_preds)
    targets_arr = np.array(all_test_targets)

    axes[0].scatter(targets_arr, preds_arr, alpha=0.2, s=8, color='blue')
    mn = min(targets_arr.min(), preds_arr.min()) - 1
    mx = max(targets_arr.max(), preds_arr.max()) + 1
    axes[0].plot([mn, mx], [mn, mx], 'r--', label='y=x')
    axes[0].set_xlabel('True Backfat (mm)')
    axes[0].set_ylabel('Predicted Backfat (mm)')
    axes[0].set_title(f'Breed={args.breed}: R2={global_metrics["R2"]:.3f}, MAE={global_metrics["MAE"]:.2f}mm')
    axes[0].legend()
    axes[0].grid(True)

    # 每折的R2柱状图
    fold_labels = [f'Fold {r["fold"]}' for r in all_fold_results]
    axes[1].bar(fold_labels, r2s, color=['#2E75B6']*len(r2s))
    axes[1].axhline(y=np.mean(r2s), color='r', linestyle='--', label=f'Mean R2={np.mean(r2s):.3f}')
    axes[1].set_ylabel('R2')
    axes[1].set_title(f'{args.n_folds}-Fold CV R2 (Breed={args.breed})')
    axes[1].legend()
    axes[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(breed_save_dir, f'results_{args.breed}.png'), dpi=150)
    plt.close()

    log(f"\n  Results saved to: {breed_save_dir}")
    log(f"\nDone!")


if __name__ == '__main__':
    main()
