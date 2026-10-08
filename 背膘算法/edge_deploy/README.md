# edge_deploy — 现场命令速查

> 配套 `F:\Postgraduate\Lab\饲喂站\26.8农科院实验\实验方案\D1_实验方案_26.8农科院.md`
> 三个脚本：演示界面、性能实测、每日校验。

## 0. 路径约定（按现场实际改）

| 变量 | 本机（PC 演练） | 现场 Jetson |
|---|---|---|
| PY | `D:\Anaconda\envs\pytorch\python.exe` | `python3` |
| 权重目录 | `F:\Postgraduate\Lab\饲喂站\背膘算法\网络\depthmap_cnn\checkpoints_depthmap\` | 拷贝过去 |
| YOLO/FastSAM | `F:\Postgraduate\Lab\饲喂站\背膘算法\yolo+FastSAM\` | 拷贝过去 |
| 数据盘 | — | `D:\nky_feeder_2026` 或 `/data/nky_feeder_2026` |

> `yolo11n.pt` 与 `FastSAM-s.pt` 默认从**当前工作目录**找，
> 所以下面的命令都先 `cd` 到放权重的目录，或用 `--yolo/--fastsam` 给绝对路径。

---

## 1. 演示界面（拍样机运行照片用）

### 现场实时流

```bash
python backfat_edge_demo.py --source live \
  --weights checkpoints_depthmap/best_model_depthmap.pth \
  --device-name "Jetson Orin NX 8GB" \
  --earid 00123 --sta-id 1 --gest-day 62 --feed "2.4 kg"
```

### 回放已录制的帧（**推荐，最稳**）

```bash
python backfat_edge_demo.py --source replay \
  --dir D:/nky_feeder_2026/20260819/1/00123_1 \
  --weights checkpoints_depthmap/best_model_depthmap.pth \
  --device-name "Jetson Orin NX 8GB" \
  --earid 00123 --sta-id 1 --gest-day 62 --feed "2.4 kg"
```

按 `q` / `ESC` 退出，按 `s` 存当前画面。默认全屏，加 `--windowed` 走窗口模式。

### 直接导出界面 PNG（论文配图从这里选，比拍屏清晰）

```bash
python backfat_edge_demo.py --source replay --dir <帧目录> \
  --weights checkpoints_depthmap/best_model_depthmap.pth \
  --headless --save-dir out_render --max-frames 100
```

> **算法改进后**：换掉 `--weights`，用**同一批现场帧**重跑这条命令，
> 就能得到视觉一致但数值更新的新图 —— 不必再进猪场。

### 没有中文字体时

```bash
python backfat_edge_demo.py ... --font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

不指定且找不到 CJK 字体时，界面会自动降级为英文标签（会打印告警）。

---

## 2. 边缘部署性能实测（补论文贡献②）

### PC 基线

```bash
python edge_profile.py --dir <帧目录> \
  --weights checkpoints_depthmap/best_model_depthmap.pth \
  --modes pt-fp32,pt-fp16 --warmup 10 --iters 100 \
  --platform "RTX 4090" --out bench_pc
```

### Jetson（先导出 engine，再全模式测）

```bash
python edge_profile.py --dir <帧目录> \
  --weights checkpoints_depthmap/best_model_depthmap.pth \
  --modes pt-fp32,pt-fp16,trt-fp16 --export \
  --warmup 10 --iters 100 \
  --platform "Jetson Orin NX 8GB" --out bench_jetson
```

产出 `bench_out/bench_table.md`（三张表，可直接贴进论文）、`bench_result.csv`、`bench_raw.json`。

**注意三点**
1. 功耗：Jetson 自动走 `tegrastats`（板级输入功率），PC 走 `nvidia-smi`（仅 GPU）——
   **两者口径不同，论文里必须注明**。都取不到时列 `n/a`，不要自己填数。
2. 回放模式下"采集+对齐"计 0，端到端 = 检测+分割+深度图+回归，与 FPS 口径一致。
   **相机真实采集耗时要用 `--source live` 单独测**。
3. 表 3 是"量化对同一批输入的输出扰动"，**不是**相对真值的精度。
   论文里的"INT8 前后 MAE 变化"必须在带标签测试集上离线评测。

### 固定功率模式后再测（避免降频导致数字不可复现）

```bash
sudo nvpmodel -m 0      # 或 15W 模式，按论文要报的口径选
sudo jetson_clocks
```

---

## 3. 每日校验

```bash
python check_daily.py --root D:/nky_feeder_2026 --date 20260819
```

带服务器上报比对：

```bash
python check_daily.py --root D:/nky_feeder_2026 --date 20260819 \
  --server-json server_20260819.json
```

退出码：`0` 全部通过 / `1` 有告警 / `2` 有致命问题（当天必须处理）。
报告写到 `<root>/check_YYYYMMDD.txt`。

---

## 4. 依赖

`torch` `torchvision` `ultralytics` `opencv-python` `pillow` `numpy`

本机已验证：torch 2.5.1 / torchvision 0.20.1 / ultralytics 8.4.21 / cv2 4.13.0 / PIL 11.0.0。

> Jetson 上装 torch 要用 NVIDIA 提供的对应 JetPack 版本的 wheel，`pip install torch` 装不上。
> 另外**不要**在部署链路里引入 `hdbscan` / `sklearn`：ARM 上依赖不友好，
> 且实测 HDBSCAN 只去掉 0.9% 噪声，本管线已改用 2D 连通域方案。
