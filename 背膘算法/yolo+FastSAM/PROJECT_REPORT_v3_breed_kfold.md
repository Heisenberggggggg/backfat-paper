# 母猪背膘厚度非接触式测量 - 项目工作报告 (v3)

> 会话时间范围: 2026-04-16 ~ 2026-04-17（完整会话窗口，含过夜训练）
> 阶段定位: **v6 品种分组 5 折交叉验证训练** + 两次崩溃（bug 崩溃 + C 盘崩溃）+ 修复
> 承接版本: v2 (2026-03-24) → v3 (本次)
> 项目目标: Azure Kinect 俯视 3D 点云 → 母猪背膘厚度 (mm) 回归
> 部署目标: Jetson Nano 8GB

---

## 〇、本报告说明

本报告覆盖**本次会话窗口内全部工作**（2026-04-16 ~ 2026-04-17）。按时间线串起 5 个阶段 + 2 次崩溃 + 3 次修复，是延续 v2 报告"v2 数据处理 + PointMLP v3 训练"之后的工作日志。

**本次会话最核心的发现不是训练结果，而是两件事**：
1. **`torch.load` 在 PyTorch 2.5.1 下默认 `weights_only=True`，pickle 含 numpy scalar 的 checkpoint 会崩** —— 导致首次 v6 训练全部 4 个品种的 Fold 1 结束时同步崩溃
2. **C 盘被填满的真凶不是 PyTorch/CUDA，而是 AnyTXT Searcher 的全文索引 `C:\Windows\Temp\G.dbl = 56 GB`** —— 它触发 Windows pagefile 耗尽，导致训练中途 `shm.dll` 加载失败（WinError 1455）

---

## 一、训练背景

### 1.1 数据集规模（承接 v2）
- DK 数据（省农科院）: nkyfirst 39 + nkysecond 32 + nkythird 92 = **163 头**
- 丰美禾数据（12 个日期全量）: **753 头**
- **合计 916 头有标签猪**（背膘 7.1–33.0 mm）

### 1.2 品种分组划分（导师提供的品种字典）
本次按品种分组训练，共 4 个子集：

| 品种 | 数据量 | 说明 |
|------|--------|------|
| A | 456 头 | 丰美禾单系 |
| LY | 233 头 | 长白 × 大白 新美系 |
| T | 65 头 | 单系少数品种 |
| ALL | 753 头 | 4 品种混合（作为对照） |

品种映射写死在 `train_v6_breed_kfold.py` 的 `BREED_MAP` 里（按耳标前缀映射）。

### 1.3 训练配置（v6 脚本）
```json
{
  "model": "PointMLP-Elite-v6",
  "n_folds": 5,
  "epochs": 200,
  "batch_size": 16,
  "lr": 0.001,
  "weight_decay": 0.01,
  "patience": 30,
  "seed": 42,
  "backfat_filter": [7.0, 32.0]
}
```
- **5 折猪级别划分**（同一头猪不跨折，即同一头猪的所有帧都在同一折）
- Early stopping (patience=30)
- 固定随机种子 `seed=42` 保证折划分可复现 → 重跑也会落到同样的 5 折上

### 1.4 导师假设
> "品种分组训练应优于混合训练" —— LY（新美系）的体型、脂肪分布与 A/T（单系）差异大，混合训练互相干扰；分组训练应该能看到 R² 提升。

本次实验目的就是验证这个假设：独立训练 A/LY/T 3 个子集，与混合的 ALL 对比。

---

## 二、阶段一：v6 脚本设计 + 首次运行的 `torch.load` bug

### 2.1 v6 脚本设计（`train_v6_breed_kfold.py`）
基于 v3 训练脚本（见 v2 报告）扩展而来，主要改动：
- 加 `--breed {A|LY|T|ALL}` 命令行参数
- 内置 `BREED_MAP` 按耳标前缀过滤
- 输出路径按品种分: `checkpoints_pointmlp_breed/breed_{A|LY|T|ALL}/fold_{1..5}/`
- 日志文件: `training_log_v6_{A|LY|T|ALL}.txt`
- 每折训练结束自动加载 best checkpoint 做测试集评估

### 2.2 启动方式: `run_breed_training.bat`
为了一次性跑完 4 个品种（≈ 6–8 小时），写了 bat 顺序执行：
```bat
[1/4] python train_v6_breed_kfold.py --breed A
[2/4] python train_v6_breed_kfold.py --breed LY
[3/4] python train_v6_breed_kfold.py --breed T
[4/4] python train_v6_breed_kfold.py --breed ALL
```

### 2.3 首次运行 —— **4 个品种全部 Fold 1 崩溃**
首次运行（会话窗口内更早时段）的症状：
- A 品种 Fold 1 跑到 200 epoch 正常结束
- Fold 1 结束后加载 best checkpoint 做测试集评估时崩溃
- **然后 Fold 2 没开始就结束了**
- LY / T / ALL 遇到同样的崩溃模式

### 2.4 Bug 根因: PyTorch 2.5.1 `weights_only` 默认值变更
```
Traceback:
  File "train_v6_breed_kfold.py", line ~180, in test_fold
    ckpt = torch.load(best_path)
UnpicklingError: Weights only load failed. numpy scalar types are not allowed ...
```
PyTorch 2.5.1 把 `torch.load(..., weights_only=True)` 改成默认值。而保存时 checkpoint 里有 numpy scalar（如 `np.float32(best_val_r2)`），新版 pickle safelist 不允许这些类型。

### 2.5 Bug 修复
```python
# train_v6_breed_kfold.py 测试阶段加载 best checkpoint:
try:
    ckpt = torch.load(best_path, weights_only=False)
except Exception as e:
    print(f"[warn] torch.load fallback: {e}")
    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
```
**效果（本次过夜跑验证）**: A 和 LY 两个品种 5 折全部正常完成，测试阶段不再崩。

---

## 三、阶段二：修 bug + 清理 + 过夜重跑（2026-04-16 晚）

### 3.1 备份清理脚本: `cleanup_before_retrain.py`
为了不破坏历史，在重跑前：
- `training_log_v6_A.txt` → `training_log_v6_A_fold1only_backup.txt`（LY/T/ALL 同样）
- 删除 `checkpoints_pointmlp_breed/breed_*` 里残留的不完整 fold 目录
- 目的：让重跑从 Fold 1 epoch 0 干净开始，避免脚本以为已经跑完

### 3.2 缓存重定向脚本: `clean_cuda_cache.py`
考虑到训练会写大量 CUDA kernel 编译缓存 + PyTorch JIT 扩展，担心 C 盘空间：
```python
# 清理 C 盘现有缓存
paths_to_clean = [
    r"C:\Users\Gengls\AppData\Local\NVIDIA\ComputeCache",
    r"C:\Users\Gengls\AppData\Local\NVIDIA Corporation\NV_Cache",
    r"C:\ProgramData\NVIDIA Corporation\NV_Cache",
    r"C:\Users\Gengls\.cache\torch",
    r"C:\Users\Gengls\AppData\Local\torch_extensions",
    # ...
]
# 把未来缓存重定向到 F 盘
subprocess.run(['setx', 'CUDA_CACHE_PATH', r'F:\cuda_pytorch_cache\cuda_compute'])
subprocess.run(['setx', 'TORCH_EXTENSIONS_DIR', r'F:\cuda_pytorch_cache\torch_extensions'])
subprocess.run(['setx', 'TORCH_HOME', r'F:\cuda_pytorch_cache\torch_hub'])
```
脚本执行后用户双击启动 bat —— **这里埋下了后续 C 盘危机的种子**（`setx` 只改注册表，当前 shell 拿不到；详见阶段五）。

### 3.3 启动过夜训练
- 时间: **2026-04-16 19:40:56**
- 命令: 双击 `run_breed_training.bat`
- 预期: 次日上午跑完 A/LY/T/ALL 4 个品种
- 用户睡觉前确认：首先看到 `[1/4] Training Breed A`，A 开始跑 Fold 1

---

## 四、阶段三：早上发现混合崩溃（2026-04-17 早）

### 4.1 早上 cmd 窗口完整状态
用户早上打开 cmd 窗口看到的内容（用户原话："以上是早上来cmd窗口里面的所有信息"）：

| 阶段 | 状态 | 时间戳（日志） |
|------|------|---------------|
| Breed A Fold 1–5 | ✅ 全部完成 | 2026-04-17 03:06:58 |
| Breed LY Fold 1–5 | ✅ 全部完成 | 2026-04-17 09:13:03 |
| Breed T Fold 1 epoch ~30 | ❌ 崩溃 | 2026-04-17 09:25 附近 |
| Breed ALL | ❌ 从未启动 | - |

### 4.2 崩溃堆栈（T 品种）
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\Anaconda\envs\pytorch\Lib\multiprocessing\spawn.py", line 122, in spawn_main
    ...
  File "F:\...\train_v6_breed_kfold.py", line 18, in <module>
    import torch
  File "D:\Anaconda\envs\pytorch\Lib\site-packages\torch\__init__.py", line 262
    _load_dll_libraries()
  File "D:\Anaconda\envs\pytorch\Lib\site-packages\torch\__init__.py", line 245
    raise err
OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading
  "D:\Anaconda\envs\pytorch\Lib\site-packages\torch\lib\shm.dll" or one of its dependencies.
```

**关键线索**：
- 崩溃点在 `multiprocessing/spawn.py` + `torch/__init__.py` —— 子进程加载 torch 时崩
- `WinError 1455 (ERROR_COMMITMENT_LIMIT)` = Windows 提交内存超限，系统想扩 pagefile 但 C 盘没空间
- `shm.dll` 是 PyTorch 共享内存 DLL，DataLoader 的 num_workers 子进程需要它

这是典型的"主进程还活着，但 spawn 的子进程拿不到新内存"。

### 4.3 训练结果（A + LY）

**品种 A（456 头）—— 5 折全部完成**

| Fold | MAE (mm) | RMSE (mm) | R² | MAPE | Best Epoch |
|------|---------|----------|-----|------|-----------|
| 1 | 2.79 | 3.48 | -0.1013 | 17.6% | 40 |
| 2 | 2.61 | 3.22 | 0.1668 | 17.0% | 27 |
| 3 | 2.67 | 3.42 | -0.1122 | 15.7% | 20 |
| 4 | 2.78 | 3.46 | -0.0283 | 18.3% | 28 |
| 5 | 2.79 | 3.54 | -0.0372 | 17.1% | 23 |
| **均值** | **2.73 ± 0.07** | **3.42 ± 0.11** | **-0.02 ± 0.10** | **17.2 ± 0.9%** | - |

全局（所有折合并）: MAE 2.73mm, R² -0.0105

**品种 LY（233 头）—— 5 折全部完成**

| Fold | MAE (mm) | RMSE (mm) | R² | MAPE | Best Epoch |
|------|---------|----------|-----|------|-----------|
| 1 | 3.56 | 4.68 | -0.1467 | 21.2% | 50 |
| 2 | 3.47 | 4.47 | -0.1992 | 19.1% | 24 |
| 3 | 2.82 | 3.62 | **0.2248** | 17.9% | 72 |
| 4 | 3.28 | 4.10 | -0.1322 | 19.0% | 32 |
| 5 | 3.06 | 3.75 | **0.2437** | 19.5% | 69 |
| **均值** | **3.24 ± 0.27** | **4.13 ± 0.41** | **0.00 ± 0.19** | **19.3 ± 1.1%** | - |

全局: MAE 3.24mm, R² 0.0126
**注**: Fold 3 / 5 的 R² 能到 0.22 / 0.24，但其余 3 折为负，方差极大。

**初步解读**:
- A (大样本): R² 稳定接近 0，模型本质在预测均值
- LY (中等样本): 整体 R²≈0 但方差大；2 折幸运命中可能源自验证集分布较窄
- **导师假设暂不可判定** —— 需要 T 和 ALL 的结果才能横向对比

---

## 五、阶段四：C 盘危机诊断

### 5.1 用户的第一个关键观察
> "我的c盘还是满了，你昨天虽然把目录改到了f盘，但是是不是没修改bat文件？运行bat文件后还是存进了c盘。"

**用户的怀疑完全正确** —— 阅读 `run_breed_training.bat` 原文后确认：bat 里**没有**任何 `set CUDA_CACHE_PATH=...` 的**内联**命令，完全依赖 `setx` 写入的注册表环境变量。

### 5.2 `setx` 为什么失效
- `setx` 只写入 Windows 注册表的 HKCU\Environment
- 需要 Explorer 收到 `WM_SETTINGCHANGE` 广播才会刷新环境块缓存
- 用户双击 bat 启动的 cmd 继承了 Explorer 缓存的**旧**环境
- PyTorch / CUDA 在 DLL 初始化时读取 cache 路径 —— 此时环境变量还是旧的，cache 落到 C 盘默认位置

### 5.3 第一轮排查 —— `clean_cuda_cache.py` 误判
用户再次运行 `clean_cuda_cache.py`，输出：
```
[NVIDIA Compute缓存] 不存在，跳过: C:\Users\Gengls\AppData\Local\NVIDIA\ComputeCache
[NVIDIA NV_Cache]    不存在，跳过: ...
[PyTorch缓存(.cache)] 已为空: C:\Users\Gengls\.cache\torch
...
小计释放: 0.00 B
```
**释放 0 B** —— 这脚本查的都是空的。**遗漏**: 没检查 `C:\Windows\Temp`。

### 5.4 第二轮排查 —— `diagnose_c_drive.py` 全盘扫描
为了不误判，我们写了 `diagnose_c_drive.py` 扫描所有常见 C 盘大户：

| 目录 | 占用 |
|------|------|
| **C:\Windows\Temp** | **53.21 GB** ← 头号元凶 |
| C:\Users\Gengls\AppData\Local (整体) | 44.44 GB |
| C:\ProgramData | 18.75 GB |
| C:\Users\Gengls\AppData\Roaming | 17.48 GB |
| C:\Windows\Installer | 7.12 GB |
| C:\Users\Gengls\Downloads | 7.46 GB |

`Windows\Temp` 占 53 GB —— `clean_cuda_cache.py` 根本没检查这个目录。

### 5.5 第三轮排查 —— PowerShell 下钻 `C:\Windows\Temp`

```powershell
Get-ChildItem -Path 'C:\Windows\Temp' -Force | ForEach-Object { ... } | Sort Size -Descending
```

输出（Top 文件）：

| 文件 | 大小 | 说明 |
|------|------|------|
| **G.dbl** | **56 GB** (!!) | **AnyTXT Searcher G 盘全文索引** |
| G.dbi | 109 MB | AnyTXT G 盘索引头 |
| CreativeCloud | 5.9 MB | Adobe |
| C.dbl / D.dbl / F.dbl | 4 MB each | AnyTXT 其他盘索引（很小） |

**真相**: AnyTXT 把 G 盘（用户所有猪只数据所在盘）全量索引了一遍，生成 56 GB 单一文件 `G.dbl`。这根本与 PyTorch / CUDA / 训练无关，是后台软件长期膨胀的结果。

### 5.6 NVIDIA 相关目录下钻（次要）
```
AppData\Local\NVIDIA:
  DXCache  = 3.65 GB  (DirectX shader cache, 安全可删)
  GLCache  = 2.3 MB   (忽略)

ProgramData\NVIDIA Corporation:
  NVIDIA app = 4.5 GB  (程序本体, 不能删)
  Downloader = 1.28 GB (旧驱动安装包, 安全可删)
```

### 5.7 诊断结论
1. **C 盘真凶是 AnyTXT 的 G.dbl**，不是 PyTorch
2. `setx` 虽然也没生效（bat 未内联 env var），但就算生效了，对 AnyTXT 也无效 —— 所以清理 C 盘必须直接删 `Windows\Temp\*.dbl`
3. **pagefile 在 C 盘且是"系统管理"** —— C 盘被填满时 Windows 没空间扩 pagefile → spawn 子进程无法加载 shm.dll → WinError 1455

---

## 六、阶段五：清理 + bat 修复（2026-04-17 上午）

### 6.1 C 盘清理（用户手动执行管理员 cmd）
```cmd
:: 1. AnyTXT 索引（最大收益 ~56 GB）
del /F /Q "C:\Windows\Temp\*.dbl"
del /F /Q "C:\Windows\Temp\*.dbi"

:: 2. NVIDIA DX shader cache (~3.6 GB)
rd /S /Q "C:\Users\Gengls\AppData\Local\NVIDIA\DXCache"

:: 3. NVIDIA 旧驱动安装包 (~1.3 GB)
rd /S /Q "C:\ProgramData\NVIDIA Corporation\Downloader"
```
**总释放: ~61 GB**

**给用户的长期建议**：在 AnyTXT 设置中移除 G 盘索引，或直接卸载 AnyTXT（用户反馈很少用），防止下次又膨胀。

### 6.2 备份 T 品种崩溃日志
```bash
cp training_log_v6_T.txt training_log_v6_T_crashed_20260417.txt
# 5932 bytes, 含 Fold 1 epoch 1–30 的记录
```
遵守项目铁律「不覆盖历史数据」（见 `MEMORY.md` feedback_no_overwrite.md）。

### 6.3 `run_breed_training.bat` 修改
**修改前**（问题版本）：
```bat
@echo off
chcp 65001 >nul
set PYTHON=D:\Anaconda\envs\pytorch\python.exe
"%PYTHON%" train_v6_breed_kfold.py --breed A
"%PYTHON%" train_v6_breed_kfold.py --breed LY
"%PYTHON%" train_v6_breed_kfold.py --breed T
"%PYTHON%" train_v6_breed_kfold.py --breed ALL
pause
```
问题：没有任何内联环境变量 → 完全依赖 `setx` 注册表传播。

**修改后**（恢复版本）：
```bat
@echo off
chcp 65001 >nul

:: === Force all PyTorch / CUDA caches to F drive (inline, do NOT rely on setx) ===
set CUDA_CACHE_PATH=F:\cuda_pytorch_cache\cuda_compute
set TORCH_EXTENSIONS_DIR=F:\cuda_pytorch_cache\torch_extensions
set TORCH_HOME=F:\cuda_pytorch_cache\torch_hub
set TMP=F:\cuda_pytorch_cache\tmp
set TEMP=F:\cuda_pytorch_cache\tmp
if not exist "F:\cuda_pytorch_cache\tmp" mkdir "F:\cuda_pytorch_cache\tmp"

echo ============================================================
echo PointMLP-Elite v6 - Breed-grouped 5-Fold CV Training
echo Resume run: T + ALL (A and LY already completed 2026-04-17)
echo Started: %date% %time%
echo CUDA_CACHE_PATH      = %CUDA_CACHE_PATH%
echo TORCH_EXTENSIONS_DIR = %TORCH_EXTENSIONS_DIR%
echo TMP/TEMP             = %TMP%
echo ============================================================

set PYTHON=D:\Anaconda\envs\pytorch\python.exe
cd /d "%~dp0"

REM === Skipped: A and LY already finished in prior run ===
REM "%PYTHON%" train_v6_breed_kfold.py --breed A
REM "%PYTHON%" train_v6_breed_kfold.py --breed LY

echo [1/2] Training Breed T (65 pigs)...
"%PYTHON%" train_v6_breed_kfold.py --breed T

echo [2/2] Training ALL breeds (753 pigs)...
"%PYTHON%" train_v6_breed_kfold.py --breed ALL

pause
```

**5 处关键改动**:
1. 头部内联 `set CUDA_CACHE_PATH / TORCH_EXTENSIONS_DIR / TORCH_HOME / TMP / TEMP` 全指向 `F:\cuda_pytorch_cache\*`（**不依赖 `setx`**）
2. `if not exist ... mkdir` 保证 TEMP 目录存在
3. 启动时 echo 环境变量，肉眼可验证未走 C 盘
4. A / LY 两段 `REM` 注释（保留原始行方便回看）
5. 编号 `[3/4]` → `[1/2]`、`[4/4]` → `[2/2]`

### 6.4 Windows pagefile 建议（未执行，需重启）
**建议用户后续手动操作**：
- 入口：「系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存」
- 把 pagefile 从 C 盘移到 F 盘，或设固定大小 32 GB
- 需重启生效

这一步没代劳，因为需要 UI 操作 + 重启，由用户自己做更稳。

---

## 七、当前状态 + 下一步

### 7.1 当前完成度（截至 2026-04-17 上午）

| 任务 | 状态 |
|------|------|
| v6 脚本 torch.load bug 修复 | ✅ 已验证有效（A/LY 完整跑完） |
| A 品种 5 折训练 | ✅ 完成，MAE 2.73mm / R² -0.02 |
| LY 品种 5 折训练 | ✅ 完成，MAE 3.24mm / R² 0.00 |
| T 品种 5 折训练 | ❌ Fold 1 ep 30 崩溃，**待重跑** |
| ALL 品种 5 折训练 | ❌ 未启动，**待重跑** |
| C 盘清理 ~61 GB | ✅ 已完成 |
| bat 文件内联 env var + 注释 A/LY | ✅ 已修改 |
| T 崩溃日志备份 | ✅ `training_log_v6_T_crashed_20260417.txt` |
| Windows pagefile 移到 F 盘 | ⏳ 建议用户手动做 |

### 7.2 下一步（用户操作 checklist）
1. [ ] 关闭所有旧 cmd 窗口（让新环境干净）
2. [ ] 双击 `run_breed_training.bat`
3. [ ] **检查第一屏输出**: `CUDA_CACHE_PATH = F:\cuda_pytorch_cache\cuda_compute` 必须显示 F 盘路径
4. [ ] 开跑后观察任务管理器 → C 盘剩余空间是否稳定（不再掉）
5. [ ] 预期时长: T ≈ 1 小时, ALL ≈ 3 小时

### 7.3 跑完之后要做的事（下次会话）
1. **4 品种横向对比**:
   - 若 ALL 的 R² 显著低于 A/LY/T 均值 → 导师假设成立
   - 若 ALL 的 R² 与各品种接近 → 假设不成立，说明俯视点云信号本身弱
2. **写 v3 最终结论**（补全 v3 报告的"训练结果"章节）
3. **下一步方向**（继承 v2 报告 §7）:
   - 多视角配准（利用已有的臀部 sub 视角数据）
   - 体况评分（BCS）替代连续回归，降任务难度
   - 验证 YOLO + FastSAM 分割质量

---

## 八、关键文件 / 目录清单

### 8.1 训练代码（`网络/pointmlp_backfat/`）
| 文件 | 用途 | 本次会话动作 |
|------|------|-------------|
| `train_v6_breed_kfold.py` | 品种分组 5 折 CV 训练 | 修 `torch.load weights_only=False` |
| `run_breed_training.bat` | 4 品种顺序启动脚本 | 内联 env var + 注释 A/LY |
| `clean_cuda_cache.py` | 清 NVIDIA/PyTorch 缓存 + `setx` | 执行过，释放 0B（路径都空） |
| `cleanup_before_retrain.py` | 备份日志 + 清不完整 fold | 过夜跑前执行 |
| `diagnose_c_drive.py` | C 盘全盘大户扫描 | 本次关键诊断工具 |

### 8.2 训练日志
| 文件 | 状态 |
|------|------|
| `training_log_v6_A.txt` | 5 折完整，归档 |
| `training_log_v6_A_fold1only_backup.txt` | 首次 bug 崩溃残留（fold1 only） |
| `training_log_v6_LY.txt` | 5 折完整，归档 |
| `training_log_v6_LY_fold1only_backup.txt` | 同上 |
| `training_log_v6_T.txt` | 本次重跑后会覆盖 |
| `training_log_v6_T_crashed_20260417.txt` | 崩溃日志备份（5932B, fold1 ep1-30） |
| `training_log_v6_T_fold1only_backup.txt` | 首次 bug 崩溃残留 |
| `training_log_v6_ALL.txt` | 待生成 |

### 8.3 模型权重
```
checkpoints_pointmlp_breed/
├── breed_A/   fold_1/ ... fold_5/   ← 5 折完整
├── breed_LY/  fold_1/ ... fold_5/   ← 5 折完整
├── breed_T/   fold_1/               ← 待重跑
└── breed_ALL/                       ← 待重跑
```

### 8.4 F 盘缓存目录（通过 bat 内联 env var 重定向）
```
F:\cuda_pytorch_cache\
├── cuda_compute\       # CUDA kernel compile cache
├── torch_extensions\   # PyTorch JIT 扩展缓存
├── torch_hub\          # torch.hub 下载缓存
└── tmp\                # bat 自动 mkdir
```

### 8.5 C 盘清理掉的目录（历史参考）
| 路径 | 大小 | 说明 |
|------|------|------|
| `C:\Windows\Temp\*.dbl` + `*.dbi` | ~56 GB | AnyTXT 全文索引 |
| `C:\Users\Gengls\AppData\Local\NVIDIA\DXCache\*` | ~3.6 GB | DX shader cache |
| `C:\ProgramData\NVIDIA Corporation\Downloader\*` | ~1.3 GB | 旧驱动安装包 |

---

## 九、本次会话学到的经验教训

### 9.1 关于 PyTorch 升级
- PyTorch 2.5+ 默认 `weights_only=True`，新旧 checkpoint 不兼容
- 保存 checkpoint 时避免 numpy scalar，或加载时显式 `weights_only=False`
- **经验**: 升 PyTorch 后要完整跑一次端到端，不能只看单个 epoch

### 9.2 关于 Windows 环境变量
- **`setx` 不可靠**：只写注册表，cmd 进程启动时是否拿到新值取决于 Explorer 广播
- **bat 脚本里用 `set VAR=...` 内联最稳** —— 立刻生效，不跨进程传播但当前进程树内 100% 有效
- **经验**: 任何对缓存路径敏感的脚本，都应该在入口处再 `set` 一遍，不要仅依赖系统级 env var

### 9.3 关于 C 盘"爆满"排查
- **不要盲信自家脚本**：`clean_cuda_cache.py` 查的是自以为的 C 盘占用点，但真凶是无关的第三方软件
- **全盘扫描比"针对性清理"更可靠**：先 `diagnose_c_drive.py` 全量扫描找大户，再 PowerShell 下钻单文件级别，最后才动删除
- **Windows pagefile 的故障模式很隐蔽**：不会直接报"空间不足"，而是表现为子进程 `shm.dll` 加载失败，容易误诊为 CUDA / PyTorch 问题

### 9.4 关于 AnyTXT
- 全盘索引类工具（AnyTXT / Everything / Listary）在大数据盘上会把索引文件撑得非常大
- 索引文件默认放 `C:\Windows\Temp`（系统临时目录），用户几乎不会去看
- **经验**：给研究数据盘（G: 几 TB 级别）建索引，单个索引文件可能接近原数据大小的 1–5%

---

## 十、给接续 Claude 的上下文摘要

将此报告发给新的 Claude Code 对话时，附上以下关键信息：

1. **项目**: 母猪背膘非接触测量, Azure Kinect 俯视安装, 部署到 Jetson Nano 8GB
2. **代码位置**:
   - `F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\` —— 数据处理 + 报告
   - `F:\Postgraduate\Lab\饲喂站\背膘算法\网络\pointmlp_backfat\` —— 训练
3. **历史最佳 (v3)**: R²=0.15, MAE=2.42mm（PointMLP v3，见 v2 报告）
4. **当前阶段 (v6)**: 品种分组 5 折 CV
   - A (456 头): MAE 2.73±0.07mm, R² **-0.02±0.10** (5 折完成)
   - LY (233 头): MAE 3.24±0.27mm, R² **0.00±0.19** (5 折完成)
   - T (65 头): 崩溃，待重跑
   - ALL (753 头): 未启动，待重跑
5. **本次会话解决的 2 个核心 bug**:
   - **`torch.load(weights_only=True)` 默认值变更** → 改 `weights_only=False` + try/except
   - **C 盘爆满 → WinError 1455 pagefile 耗尽**
     - 真因: `C:\Windows\Temp\G.dbl` = 56 GB 的 AnyTXT 索引（不是 PyTorch）
     - 修复: 手动删 `*.dbl/*.dbi` + NVIDIA cache，释放 ~61 GB
     - 配套: bat 内联 `set CUDA_CACHE_PATH=...` 不依赖 `setx`
6. **Python 环境**: `D:\Anaconda\envs\pytorch\python.exe`, PyTorch 2.5.1
7. **F 盘缓存目录**: `F:\cuda_pytorch_cache\{cuda_compute, torch_extensions, torch_hub, tmp}`
8. **用户铁律**（见 `C:\Users\Gengls\.claude\projects\...\memory\MEMORY.md`）:
   - 执行代码修改 / 运行脚本前**必须先列计划让用户确认**
   - 不覆盖已有数据，新输出用版本后缀（如 `_crashed_20260417`）
   - 脚本头部注释说明来历和解决思路
9. **关键 bat 文件**: `网络/pointmlp_backfat/run_breed_training.bat`（已内联 env var，只剩 T + ALL）
10. **导师假设**: 品种分组训练应优于混合训练（**待 ALL 跑完验证**）
11. **崩溃排查速查**:
    - 若再遇 **WinError 1455 / shm.dll 加载失败** → 先跑 `diagnose_c_drive.py` 查 C 盘 → PowerShell 下钻 `C:\Windows\Temp` → 查 `*.dbl` 等大文件
    - 若再遇 **`torch.load UnpicklingError`** → 加 `weights_only=False`
12. **已知待办**:
    - 重跑 T + ALL
    - 跑完后横向对比 4 品种验证导师假设
    - 建议用户把 Windows pagefile 从 C 盘移到 F 盘（防止再满时崩）
    - 建议用户卸载或禁用 AnyTXT 对 G 盘的索引

---

**报告生成时间**: 2026-04-17
**下一个版本**: v4（待 T + ALL 跑完后横向对比 4 品种）
