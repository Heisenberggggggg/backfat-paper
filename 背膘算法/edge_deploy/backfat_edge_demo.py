# -*- coding: utf-8 -*-
"""
母猪背膘边缘端在线监测演示程序
================================================================
用途
    部署在饲喂站旁的 Jetson 上，实时展示"检测 -> 分割 -> 深度图 -> 背膘回归"
    全链路结果，用于拍摄论文中的"样机运行照片"（对标王起繁论文图 8）。

设计要点（两条，决定了这个程序为什么这样写）
    1) 数据源可切换：--source replay <dir> 回放现场录制帧；--source live 走相机实时流。
       回放模式使得在实验室也能复现与猪场完全一致的界面，因此现场只需要把原始帧采回来，
       算法改进后无需再进猪场补拍。
    2) 模型热插拔：--weights <path> 指定回归权重。
       今天挂当前 ResNet-18，将来换成改进模型后，同一界面、同一批帧重新截图即可。

    界面显示的背膘数值始终是所加载模型的真实输出，不做任何写死或修饰。
    论文中的定量精度指标另取自离线评测，两者不混用。

界面用 OpenCV 画布 + PIL 文字渲染，不依赖 PyQt/Qt，Jetson 上最省事。

算法链路与既有工程保持一致（见 01_课题总览 §5.1）：
    YOLO11n 检测（取消类别过滤，取最大框，失败时全图兜底）
      -> FastSAM-s 分割（bbox 作先验）
      -> 掩膜 x 深度图，裁 bbox，resize 256 -> 224
      -> 归一化 (depth-300)/(2500-300) -> DepthMapResNet -> 背膘 mm
    部署链路刻意不做 HDBSCAN（ARM 上 sklearn 依赖不友好，且实测仅去除 0.9% 噪声）。

Author: 耿来硕 + Claude
Date:   2026-08-18
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

# ------------------------------------------------------------------
# 常量：与 train_depthmap.py 严格一致，改动会导致权重失配
# ------------------------------------------------------------------
NORM_MIN = 300.0        # mm
NORM_MAX = 2500.0       # mm
DEPTHMAP_SIZE = 256     # 掩膜深度图中间尺寸
MODEL_INPUT = 224       # ResNet 输入
YOLO_CONF = 0.15        # 低阈值保召回（通用权重会把母猪判成绵羊，故不过滤类别）
FASTSAM_CONF = 0.4

# 体况三分级阈值（mm）。可用 --bcs-thresh 覆盖。
BCS_THIN, BCS_FAT = 13.0, 19.0

CANVAS_W, CANVAS_H = 1600, 900

# 全部按 OpenCV 的 BGR 顺序书写（注释里给等价 RGB，便于核对）
COL_BG = (30, 26, 24)         # RGB(24,26,30)
COL_PANEL = (45, 39, 36)      # RGB(36,39,45)
COL_LINE = (84, 75, 70)       # RGB(70,75,84)
COL_TEXT = (238, 234, 232)    # RGB(232,234,238)
COL_DIM = (165, 155, 150)     # RGB(150,155,165)
COL_ACC = (233, 180, 86)      # RGB(86,180,233) 主强调色（蓝）
COL_OK = (130, 200, 90)       # RGB(90,200,130) 正常（绿）
COL_WARN = (70, 180, 240)     # RGB(240,180,70) 偏瘦（琥珀）
COL_BAD = (110, 110, 230)     # RGB(230,110,110) 偏肥（红）


# ==================================================================
# 文字渲染：OpenCV 不支持中文，用 PIL 叠字
# ==================================================================
class TextDrawer:
    """把中文/英文文字画到 BGR ndarray 上。找不到 CJK 字体时自动降级为英文标签。"""

    CJK_CANDIDATES = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    LATIN_FALLBACK = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        r'C:\Windows\Fonts\arialbd.ttf',
    ]

    def __init__(self, font_path=None):
        from PIL import ImageFont
        self._ImageFont = ImageFont
        self.path = None
        self.cjk_ok = False

        for p in ([font_path] if font_path else []) + self.CJK_CANDIDATES:
            if p and os.path.exists(p):
                self.path, self.cjk_ok = p, True
                break
        if self.path is None:
            for p in self.LATIN_FALLBACK:
                if os.path.exists(p):
                    self.path = p
                    break
        self._cache = {}
        self._queue = []
        if not self.cjk_ok:
            print('[warn] 未找到中文字体，界面将使用英文标签。'
                  '可用 --font 指定一个 CJK 字体文件。', file=sys.stderr)

    def font(self, size):
        if size not in self._cache:
            if self.path:
                self._cache[size] = self._ImageFont.truetype(self.path, size)
            else:
                self._cache[size] = self._ImageFont.load_default()
        return self._cache[size]

    def label(self, zh, en):
        """有中文字体就用中文，否则退回英文。"""
        return zh if self.cjk_ok else en

    def draw(self, img, xy, text, size=18, color=COL_TEXT, anchor='la'):
        """只入队，不立即绘制。

        每帧约 40 次文字调用，若每次都做一遍 1600x900 的 BGR->PIL->BGR 往返，
        渲染开销会盖过推理本身（实测把整体帧率从 ~20 拉到 ~3.6）。
        统一入队、每帧 flush 一次，往返只发生 1 次。
        队列里的文字最后统一绘制，因此总是压在 cv2 图元之上 —— 当前布局
        没有"文字需被后续色块覆盖"的情况，故与逐次绘制的视觉结果一致。
        """
        self._queue.append((xy, text, size, color, anchor))

    def flush(self, img):
        from PIL import Image, ImageDraw
        if not self._queue:
            return
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for xy, text, size, color, anchor in self._queue:
            d.text(xy, text, font=self.font(size),
                   fill=(color[2], color[1], color[0]), anchor=anchor)
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        self._queue.clear()


# ==================================================================
# 推理链路
# ==================================================================
class BackfatPipeline:
    """检测 -> 分割 -> 深度图 -> 回归。每一阶段单独计时。"""

    def __init__(self, yolo_path, fastsam_path, weights, device='cuda'):
        import torch
        self.torch = torch
        want_cuda = (device == 'cuda')
        self.device = torch.device(
            'cuda' if (want_cuda and torch.cuda.is_available()) else 'cpu')
        if want_cuda and self.device.type == 'cpu':
            print('[warn] CUDA 不可用，回落到 CPU。', file=sys.stderr)

        from ultralytics import YOLO, FastSAM
        self.yolo = YOLO(yolo_path)
        self.sam = FastSAM(fastsam_path)
        self.model = self._load_regressor(weights)
        self.weights_name = os.path.basename(weights) if weights else '(none)'
        self.timing = {}

    # -------------------------------------------------- 回归模型
    def _build_resnet(self):
        """与 train_depthmap.py 的 DepthMapResNet 结构完全一致，否则 state_dict 加载失败。"""
        import torch
        import torch.nn as nn
        from torchvision import models

        class DepthMapResNet(nn.Module):
            def __init__(self, dropout=0.3):
                super().__init__()
                r = models.resnet18(weights=None)
                self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                       padding=3, bias=False)
                self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
                self.layer1, self.layer2 = r.layer1, r.layer2
                self.layer3, self.layer4 = r.layer3, r.layer4
                self.avgpool = r.avgpool
                self.regressor = nn.Sequential(
                    nn.Dropout(dropout), nn.Linear(r.fc.in_features, 128),
                    nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 1))

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
                return self.regressor(torch.flatten(self.avgpool(x), 1))

        return DepthMapResNet()

    def _load_regressor(self, weights):
        m = self._build_resnet()
        if weights and os.path.exists(weights):
            # PyTorch >= 2.5 起 torch.load 默认 weights_only=True，会拒绝 numpy 标量，
            # 而本项目的 checkpoint 里存了 numpy 指标 —— 见 01_课题总览 §7.3。
            ck = self.torch.load(weights, map_location='cpu', weights_only=False)
            sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print('[warn] state_dict 部分不匹配 missing=%d unexpected=%d'
                      % (len(missing), len(unexpected)), file=sys.stderr)
        else:
            print('[warn] 未提供有效权重，回归输出无意义（仅用于链路联调）。',
                  file=sys.stderr)
        return m.to(self.device).eval()

    # -------------------------------------------------- 各阶段
    def detect(self, rgb):
        """取面积最大的框；取消类别过滤（COCO 无 pig，母猪常被判成 sheep）。
        猪贴近相机占满画面时检测器会失效，此时用全图 bbox 兜底。"""
        r = self.yolo.predict(rgb, conf=YOLO_CONF, verbose=False)[0]
        h, w = rgb.shape[:2]
        if r.boxes is None or len(r.boxes) == 0:
            return (0, 0, w, h), False
        xyxy = r.boxes.xyxy.cpu().numpy()          # 必须先 .cpu()，否则 argmax 报错
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        x1, y1, x2, y2 = xyxy[int(np.argmax(areas))].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:             # 框退化时同样兜底
            return (0, 0, w, h), False
        return (x1, y1, x2, y2), True

    def segment(self, rgb, bbox):
        r = self.sam.predict(rgb, bboxes=[list(bbox)], conf=FASTSAM_CONF,
                             verbose=False)[0]
        h, w = rgb.shape[:2]
        if r.masks is None or len(r.masks.data) == 0:
            m = np.zeros((h, w), np.uint8)
            m[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1     # 兜底：用 bbox 当掩膜
            return m, False
        md = r.masks.data.cpu().numpy()
        m = md[int(np.argmax(md.reshape(len(md), -1).sum(1)))]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_NEAREST)
        return (m > 0.5).astype(np.uint8), True

    def make_depthmap(self, depth_mm, mask, bbox):
        """掩膜 x 深度 -> 裁 bbox -> resize 256。全程 2D 域，不做三维反投影。"""
        d = depth_mm.astype(np.float32) * mask
        d[(d < NORM_MIN) | (d > 3000.0)] = 0.0
        x1, y1, x2, y2 = bbox
        crop = d[y1:y2, x1:x2]
        if crop.size == 0:
            crop = d
        return cv2.resize(crop, (DEPTHMAP_SIZE, DEPTHMAP_SIZE),
                          interpolation=cv2.INTER_LINEAR)

    def regress(self, depthmap):
        x = (depthmap - NORM_MIN) / (NORM_MAX - NORM_MIN)
        x = np.clip(x, 0, 1)
        x = cv2.resize(x, (MODEL_INPUT, MODEL_INPUT),
                       interpolation=cv2.INTER_LINEAR)
        t = self.torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(self.device)
        with self.torch.no_grad():
            return float(self.model(t).squeeze().cpu().numpy())

    # -------------------------------------------------- 单帧
    def run(self, rgb, depth_mm, grab_ms=0.0):
        """返回结果 dict；self.timing 记录各阶段耗时 ms。"""
        def _sync():
            if self.device.type == 'cuda':
                self.torch.cuda.synchronize()

        t0 = time.perf_counter()
        bbox, det_ok = self.detect(rgb)
        _sync()
        t1 = time.perf_counter()
        mask, seg_ok = self.segment(rgb, bbox)
        _sync()
        t2 = time.perf_counter()
        dm = self.make_depthmap(depth_mm, mask, bbox)
        t3 = time.perf_counter()
        backfat = self.regress(dm)
        _sync()
        t4 = time.perf_counter()

        self.timing = {
            'grab': grab_ms,
            'detect': (t1 - t0) * 1000.0,
            'segment': (t2 - t1) * 1000.0,
            'depthmap': (t3 - t2) * 1000.0,
            'regress': (t4 - t3) * 1000.0,
        }
        self.timing['total'] = sum(self.timing.values())
        return {'bbox': bbox, 'mask': mask, 'depthmap': dm, 'backfat': backfat,
                'det_ok': det_ok, 'seg_ok': seg_ok,
                'mask_ratio': float(mask.mean())}


# ==================================================================
# 数据源
# ==================================================================
class ReplaySource:
    """回放现场录制帧。目录结构见 D1 方案 §3.3：<dir>/color/*.png + <dir>/depth/*.png"""

    def __init__(self, root, loop=True):
        self.root, self.loop, self.i = root, loop, 0
        cdir = os.path.join(root, 'color')
        ddir = os.path.join(root, 'depth')
        if not os.path.isdir(cdir):
            raise SystemExit('回放目录缺少 color/ 子目录: %s' % root)
        ext = ('.png', '.jpg', '.jpeg')
        self.color = sorted(f for f in os.listdir(cdir) if f.lower().endswith(ext))
        self.ddir, self.cdir = ddir, cdir
        self.depth = (sorted(f for f in os.listdir(ddir) if f.lower().endswith(ext))
                      if os.path.isdir(ddir) else [])
        if not self.color:
            raise SystemExit('回放目录无图像: %s' % cdir)
        if not self.depth:
            print('[warn] 未找到 depth/ 帧，深度图与回归结果不可用。', file=sys.stderr)
        self.name = os.path.basename(os.path.normpath(root))

    def __len__(self):
        return len(self.color)

    def read(self):
        if self.i >= len(self.color):
            if not self.loop:
                return None
            self.i = 0
        t0 = time.perf_counter()
        rgb = cv2.imread(os.path.join(self.cdir, self.color[self.i]),
                         cv2.IMREAD_COLOR)
        if rgb is None:
            self.i += 1
            return self.read()
        if self.depth:
            k = min(self.i, len(self.depth) - 1)
            depth = cv2.imread(os.path.join(self.ddir, self.depth[k]),
                               cv2.IMREAD_UNCHANGED)
        else:
            depth = None
        if depth is None:
            depth = np.zeros(rgb.shape[:2], np.uint16)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        idx = self.i
        self.i += 1
        return rgb, depth.astype(np.float32), (time.perf_counter() - t0) * 1000.0, idx


class LiveSource:
    """Orbbec Femto Bolt 实时流。SDK 缺失时给出明确提示而不是静默失败。"""

    def __init__(self):
        try:
            from pyorbbecsdk import Config, Pipeline, OBSensorType
        except ImportError:
            raise SystemExit(
                '未安装 pyorbbecsdk，无法使用 --source live。\n'
                '现场请改用 --source replay <已录制帧目录> 进行演示与拍照。')
        self.pipe = Pipeline()
        cfg = Config()
        pl = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        cfg.enable_stream(pl.get_default_video_stream_profile())
        pl = self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        cfg.enable_stream(pl.get_default_video_stream_profile())
        self.pipe.start(cfg)
        self.name = 'live'
        self.i = 0

    def __len__(self):
        return 0

    def read(self):
        t0 = time.perf_counter()
        fs = self.pipe.wait_for_frames(200)
        if fs is None:
            return None
        cf, df = fs.get_color_frame(), fs.get_depth_frame()
        if cf is None or df is None:
            return None
        rgb = np.frombuffer(cf.get_data(), np.uint8).reshape(
            cf.get_height(), cf.get_width(), 3).copy()
        depth = np.frombuffer(df.get_data(), np.uint16).reshape(
            df.get_height(), df.get_width()).astype(np.float32)
        depth *= df.get_depth_scale()          # -> mm
        if depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        self.i += 1
        return rgb, depth, (time.perf_counter() - t0) * 1000.0, self.i


# ==================================================================
# 界面渲染
# ==================================================================
class Renderer:
    def __init__(self, td, meta, thresh=(BCS_THIN, BCS_FAT)):
        self.td, self.meta = td, meta
        self.thin, self.fat = thresh
        self.history = deque(maxlen=120)

    def grade(self, v):
        L = self.td.label
        if v < self.thin:
            return L('偏瘦', 'THIN'), COL_WARN
        if v > self.fat:
            return L('偏肥', 'FAT'), COL_BAD
        return L('正常', 'NORMAL'), COL_OK

    @staticmethod
    def _panel(img, x1, y1, x2, y2):
        cv2.rectangle(img, (x1, y1), (x2, y2), COL_PANEL, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), COL_LINE, 1)

    def render(self, rgb, res, timing, fps, src_name, frame_idx, n_frames):
        L, td = self.td.label, self.td
        cvs = np.full((CANVAS_H, CANVAS_W, 3), COL_BG, np.uint8)

        # ---------- 顶栏 ----------
        cv2.rectangle(cvs, (0, 0), (CANVAS_W, 46), COL_PANEL, -1)
        td.draw(cvs, (16, 10),
                L('母猪背膘非接触在线监测系统  —  边缘计算终端',
                  'Sow Backfat Non-contact Online Monitoring  -  Edge Terminal'),
                22, COL_TEXT)
        td.draw(cvs, (CANVAS_W - 16, 13),
                '%s | %s' % (self.meta['device'],
                             time.strftime('%Y-%m-%d %H:%M:%S')),
                15, COL_DIM, anchor='ra')

        # ---------- 左：RGB + 检测 + 掩膜 ----------
        LX, LY, LW, LH = 12, 58, 936, 620
        self._panel(cvs, LX, LY, LX + LW, LY + LH)
        vis = rgb.copy()
        m = res['mask']
        if m.any():
            vis[m > 0] = (0.45 * np.array(COL_ACC, np.float32)
                          + 0.55 * vis[m > 0]).astype(np.uint8)
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), COL_ACC, 2)
        cv2.putText(vis, 'sow %.2f' % res['mask_ratio'], (x1 + 4, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ACC, 2)

        s = min((LW - 16) / vis.shape[1], (LH - 40) / vis.shape[0])
        vis = cv2.resize(vis, (max(1, int(vis.shape[1] * s)),
                               max(1, int(vis.shape[0] * s))))
        ox = LX + (LW - vis.shape[1]) // 2
        oy = LY + 32 + (LH - 32 - vis.shape[0]) // 2
        cvs[oy:oy + vis.shape[0], ox:ox + vis.shape[1]] = vis
        td.draw(cvs, (LX + 12, LY + 7),
                L('① RGB 图像 + YOLO 检测框 + FastSAM 掩膜',
                  '(1) RGB + YOLO bbox + FastSAM mask'), 17, COL_DIM)

        # ---------- 左下：本次会话预测序列 ----------
        HX, HY, HW, HH = 12, LY + LH + 8, 936, CANVAS_H - (LY + LH + 8) - 12
        self._panel(cvs, HX, HY, HX + HW, HY + HH)
        td.draw(cvs, (HX + 12, HY + 6),
                L('④ 本次会话背膘预测序列（逐帧）',
                  '(4) Per-frame backfat prediction'), 17, COL_DIM)
        if len(self.history) > 1:
            h = list(self.history)
            lo, hi = min(h), max(h)
            if hi - lo < 2.0:
                mid = (lo + hi) / 2.0
                lo, hi = mid - 1.0, mid + 1.0
            px0, py0 = HX + 60, HY + 32
            pw, ph = HW - 80, HH - 48
            for gv, gc in ((self.thin, COL_WARN), (self.fat, COL_BAD)):
                if lo <= gv <= hi:
                    gy = int(py0 + ph - (gv - lo) / (hi - lo) * ph)
                    cv2.line(cvs, (px0, gy), (px0 + pw, gy), gc, 1, cv2.LINE_AA)
            pts = [(int(px0 + i / (len(h) - 1) * pw),
                    int(py0 + ph - (v - lo) / (hi - lo) * ph))
                   for i, v in enumerate(h)]
            cv2.polylines(cvs, [np.array(pts, np.int32)], False, COL_ACC, 2,
                          cv2.LINE_AA)
            td.draw(cvs, (HX + 8, py0 - 8), '%.1f' % hi, 13, COL_DIM)
            td.draw(cvs, (HX + 8, py0 + ph - 8), '%.1f' % lo, 13, COL_DIM)

        # ---------- 右上：深度图 ----------
        RX, RW = 960, CANVAS_W - 960 - 12
        self._panel(cvs, RX, 58, RX + RW, 330)
        dm = res['depthmap']
        # resize 的线性插值会在猪体边缘造出 0~300 之间的中间值，必须按 NORM_MIN 判有效，
        # 否则"深度范围"会显示成 1-1509 mm 这种假下界。
        valid = dm >= NORM_MIN
        norm = np.zeros(dm.shape, np.uint8)
        if valid.any():
            v = dm[valid]
            rng = float(v.max() - v.min())
            norm[valid] = np.clip((v - v.min()) / max(1e-6, rng) * 255,
                                  0, 255).astype(np.uint8)
        dvis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        dvis[~valid] = COL_PANEL
        dvis = cv2.resize(dvis, (232, 232), interpolation=cv2.INTER_NEAREST)
        cvs[84:316, RX + 14:RX + 246] = dvis
        td.draw(cvs, (RX + 12, 62),
                L('② 掩膜深度图 (256x256)', '(2) Masked depth map'), 17, COL_DIM)
        dmin, dmax = (float(dm[valid].min()), float(dm[valid].max())) \
            if valid.any() else (0.0, 0.0)
        info = [
            (L('有效像素', 'valid px'), '%.1f%%' % (100.0 * valid.mean())),
            (L('深度范围', 'range'), '%.0f-%.0f mm' % (dmin, dmax)),
            (L('检测', 'detect'),
             L('成功', 'ok') if res['det_ok'] else L('全图兜底', 'fallback')),
            (L('分割', 'segment'),
             L('成功', 'ok') if res['seg_ok'] else L('bbox兜底', 'fallback')),
        ]
        for i, (k, v) in enumerate(info):
            td.draw(cvs, (RX + 262, 96 + i * 30), k, 15, COL_DIM)
            td.draw(cvs, (RX + 382, 96 + i * 30), v, 15, COL_TEXT)

        # ---------- 右中：背膘读数 ----------
        self._panel(cvs, RX, 338, RX + RW, 520)
        td.draw(cvs, (RX + 12, 344),
                L('③ 背膘厚度回归输出', '(3) Backfat regression output'), 17, COL_DIM)
        bf = res['backfat']
        gtxt, gcol = self.grade(bf)
        td.draw(cvs, (RX + 24, 372), '%.1f' % bf, 70, COL_ACC)
        td.draw(cvs, (RX + 200, 420), 'mm', 24, COL_DIM)
        td.draw(cvs, (RX + 290, 378), L('体况', 'BCS'), 17, COL_DIM)
        td.draw(cvs, (RX + 290, 402), gtxt, 34, gcol)

        bx, by, bw, bh = RX + 24, 476, RW - 48, 20
        lo_s, hi_s = 8.0, 26.0
        for a, b, c in ((lo_s, self.thin, COL_WARN), (self.thin, self.fat, COL_OK),
                        (self.fat, hi_s, COL_BAD)):
            xa = int(bx + (a - lo_s) / (hi_s - lo_s) * bw)
            xb = int(bx + (b - lo_s) / (hi_s - lo_s) * bw)
            cv2.rectangle(cvs, (xa, by), (xb, by + bh), c, -1)
        mx = int(bx + (float(np.clip(bf, lo_s, hi_s)) - lo_s) / (hi_s - lo_s) * bw)
        cv2.line(cvs, (mx, by - 6), (mx, by + bh + 6), COL_TEXT, 3)
        td.draw(cvs, (bx, by + bh + 6), '%.0f' % lo_s, 13, COL_DIM)
        td.draw(cvs, (bx + bw, by + bh + 6), '%.0f' % hi_s, 13, COL_DIM, anchor='ra')

        # ---------- 右下：信息 + 时延 ----------
        self._panel(cvs, RX, 528, RX + RW, CANVAS_H - 12)
        td.draw(cvs, (RX + 12, 534),
                L('⑤ 会话信息与推理时延', '(5) Session info & latency'), 17, COL_DIM)
        src_txt = ('%s [%d/%d]' % (src_name, frame_idx + 1, n_frames)
                   if n_frames else '%s [%d]' % (src_name, frame_idx + 1))
        rows = [
            (L('耳标号', 'Ear tag'), self.meta['earid']),
            (L('饲喂站', 'Station'), self.meta['sta_id']),
            (L('妊娠日龄', 'Gest. day'), self.meta['gest_day']),
            (L('本次下料', 'Feed'), self.meta['feed']),
            (L('数据源', 'Source'), src_txt),
            (L('权重', 'Weights'), self.meta['weights']),
        ]
        y = 560
        for k, v in rows:
            td.draw(cvs, (RX + 20, y), k, 15, COL_DIM)
            td.draw(cvs, (RX + 150, y), str(v), 15, COL_TEXT)
            y += 24
        cv2.line(cvs, (RX + 16, y + 4), (RX + RW - 16, y + 4), COL_LINE, 1)
        y += 16

        stages = [(L('采集', 'Grab'), 'grab'), (L('检测', 'Detect'), 'detect'),
                  (L('分割', 'Segment'), 'segment'),
                  (L('深度图', 'DepthMap'), 'depthmap'),
                  (L('回归', 'Regress'), 'regress')]
        tot = max(1e-6, timing['total'])
        for k, key in stages:
            v = timing.get(key, 0.0)
            td.draw(cvs, (RX + 20, y), k, 15, COL_DIM)
            td.draw(cvs, (RX + 150, y), '%7.1f ms' % v, 15, COL_TEXT)
            bw2 = int((RW - 300) * v / tot)
            cv2.rectangle(cvs, (RX + 265, y + 5),
                          (RX + 265 + max(1, bw2), y + 16), COL_ACC, -1)
            y += 24
        td.draw(cvs, (RX + 20, y + 4), L('端到端', 'End-to-end'), 17, COL_TEXT)
        td.draw(cvs, (RX + 150, y + 4), '%7.1f ms' % tot, 17, COL_ACC)
        td.draw(cvs, (RX + 300, y + 4), '%.1f FPS' % fps, 17, COL_ACC)
        td.flush(cvs)          # 整帧文字一次性落到画布
        return cvs


# ==================================================================
def main():
    ap = argparse.ArgumentParser(
        description='母猪背膘边缘端在线监测演示（真实推理，界面数值即模型输出）')
    ap.add_argument('--source', default='replay', choices=['replay', 'live'])
    ap.add_argument('--dir', help='replay 模式的帧目录（含 color/ 与 depth/）')
    ap.add_argument('--weights', default='', help='背膘回归权重 .pth')
    ap.add_argument('--yolo', default='yolo11n.pt')
    ap.add_argument('--fastsam', default='FastSAM-s.pt')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--font', default=None, help='CJK 字体文件路径')
    ap.add_argument('--earid', default='-')
    ap.add_argument('--sta-id', default='-')
    ap.add_argument('--gest-day', default='-')
    ap.add_argument('--feed', default='-')
    ap.add_argument('--device-name', default='Jetson',
                    help='顶栏显示的平台名，如 "Jetson Orin NX 8GB"')
    ap.add_argument('--bcs-thresh', default='13,19',
                    help='体况三分级阈值 mm，形如 13,19')
    ap.add_argument('--fps-limit', type=float, default=8.0)
    ap.add_argument('--save-dir', default=None,
                    help='把渲染画面逐帧存 PNG（论文配图从这里取，可在实验室重生成）')
    ap.add_argument('--max-frames', type=int, default=0,
                    help='处理多少帧后自动退出，0 = 不限')
    ap.add_argument('--windowed', action='store_true', help='窗口模式（默认全屏）')
    ap.add_argument('--headless', action='store_true',
                    help='不开窗口，只按 --save-dir 出图（无显示器时用）')
    a = ap.parse_args()

    if a.source == 'replay' and not a.dir:
        ap.error('--source replay 需要 --dir 指定帧目录')
    if a.headless and not a.save_dir:
        ap.error('--headless 需要同时给 --save-dir')
    thin, fat = [float(x) for x in a.bcs_thresh.split(',')]

    src = ReplaySource(a.dir) if a.source == 'replay' else LiveSource()
    pipe = BackfatPipeline(a.yolo, a.fastsam, a.weights, a.device)
    td = TextDrawer(a.font)
    meta = {'earid': a.earid, 'sta_id': a.sta_id, 'gest_day': a.gest_day,
            'feed': a.feed, 'weights': pipe.weights_name,
            'device': a.device_name}
    rend = Renderer(td, meta, (thin, fat))

    csv_f = csv_w = None
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        csv_f = open(os.path.join(a.save_dir, 'predictions.csv'), 'w',
                     newline='', encoding='utf-8')
        csv_w = csv.writer(csv_f)
        csv_w.writerow(['frame', 'backfat_mm', 'grab_ms', 'detect_ms',
                        'segment_ms', 'depthmap_ms', 'regress_ms', 'total_ms'])

    win = 'backfat_edge_demo'
    if not a.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if not a.windowed:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)

    print('[info] device=%s  weights=%s  frames=%s'
          % (pipe.device, pipe.weights_name, len(src) or 'live'))
    if not a.headless:
        print('[info] 按 q 或 ESC 退出；按 s 保存当前画面。')

    fps, n = 0.0, 0
    try:
        while True:
            t_loop = time.perf_counter()
            got = src.read()
            if got is None:
                break
            rgb, depth, grab_ms, idx = got
            res = pipe.run(rgb, depth, grab_ms)
            rend.history.append(res['backfat'])

            canvas = rend.render(rgb, res, pipe.timing, fps,
                                 src.name, idx, len(src))
            # FPS 只统计"采集 + 推理 + 渲染"，不含 --save-dir 的 PNG 落盘开销，
            # 否则出图模式下的帧率会被磁盘 IO 严重拉低，读数失真。
            t_pipeline_done = time.perf_counter()
            if not a.headless:
                cv2.imshow(win, canvas)

            if csv_w:
                t = pipe.timing
                csv_w.writerow([idx, '%.3f' % res['backfat']]
                               + ['%.2f' % t[k] for k in
                                  ('grab', 'detect', 'segment', 'depthmap',
                                   'regress', 'total')])
                cv2.imwrite(os.path.join(a.save_dir, 'render_%04d.png' % idx),
                            canvas)

            if not a.headless:
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k == ord('s'):
                    fn = 'snapshot_%s.png' % time.strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(os.path.join(a.save_dir or '.', fn), canvas)
                    print('[info] 已保存', fn)

            dt = t_pipeline_done - t_loop
            if a.fps_limit > 0 and not a.headless:
                slack = 1.0 / a.fps_limit - (time.perf_counter() - t_loop)
                if slack > 0:
                    time.sleep(slack)
            n += 1
            # 第 1 帧含 CUDA/TensorRT warm-up，不计入均值，否则 FPS 长时间偏低
            if n > 1:
                fps = (1.0 / max(dt, 1e-6)) if n == 2 else \
                    0.9 * fps + 0.1 / max(dt, 1e-6)
            if a.max_frames and n >= a.max_frames:
                break
    finally:
        if csv_f:
            csv_f.close()
        cv2.destroyAllWindows()
    print('[info] 共处理 %d 帧' % n)


if __name__ == '__main__':
    main()
