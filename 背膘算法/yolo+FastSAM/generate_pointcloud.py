"""
基于 FastSAM 掩膜 + 深度图 + 相机内参 生成猪体点云 (.ply)
流程：FastSAM 分割 → mask 过滤深度图 → 反投影为 3D 点云 → 保存 PLY
"""
from ultralytics import YOLO, FastSAM
import cv2
import os
import gc
import numpy as np
import open3d as o3d
import torch


# --- 1. 支持中文路径 ---
def imread_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imread_depth_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


# --- 2. 相机内参（来自 intrinsic.json） ---
FX = 614.84063720703125
FY = 614.61328125
CX = 642.5406494140625
CY = 364.29312133789062
WIDTH = 1280
HEIGHT = 720
DEPTH_SCALE = 1000.0  # 深度值单位 mm → m
MIN_DEPTH = 0.3       # 最小有效深度 (m)
MAX_DEPTH = 3.0       # 最大有效深度 (m)

# --- 3. 路径配置 ---
color_dir = "G:/耿来硕数据/DK/nkyfirst/2-1/depth_cz_r_d_p/master/color/"
depth_dir = "G:/耿来硕数据/DK/nkyfirst/2-1/depth_cz_r_d_p/master/depth/"
output_dir = "G:/耿来硕数据/DK/nkyfirst/2-1/depth_cz_r_d_p/master/masked_pointclouds/"
os.makedirs(output_dir, exist_ok=True)

# --- 4. 加载模型 ---
print("正在加载模型...")
detect_model = YOLO('yolo11n.pt')
segment_model = FastSAM('FastSAM-s.pt')


def get_mask(img):
    """用 YOLO + FastSAM 获取猪体掩膜"""
    det_results = detect_model.predict(img, conf=0.15, device='cuda', verbose=False)[0]

    if len(det_results.boxes) == 0:
        h, w = img.shape[:2]
        margin = 0.05
        best_box = np.array([w * margin, h * margin, w * (1 - margin), h * (1 - margin)])
    else:
        areas = (det_results.boxes.xywh[:, 2] * det_results.boxes.xywh[:, 3]).cpu().numpy()
        max_idx = np.argmax(areas)
        best_box = det_results.boxes.xyxy[max_idx].cpu().numpy()

    seg_results = segment_model.predict(
        img, bboxes=best_box, device='cuda', retina_masks=True, conf=0.4, verbose=False
    )[0]

    if seg_results.masks is not None:
        mask = seg_results.masks.data[0].cpu().numpy().astype(np.uint8)
        # mask 分辨率可能和原图不同，需要 resize
        if mask.shape[:2] != (HEIGHT, WIDTH):
            mask = cv2.resize(mask, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
        return mask
    return None


def depth_to_pointcloud(color_img, depth_img, mask):
    """将掩膜区域的 depth + color 反投影为 3D 点云"""
    # 生成像素坐标网格
    v, u = np.where(mask == 1)  # v=行(y), u=列(x)
    z = depth_img[v, u].astype(np.float32) / DEPTH_SCALE  # mm → m

    # 过滤无效深度
    valid = (z > MIN_DEPTH) & (z < MAX_DEPTH)
    u, v, z = u[valid], v[valid], z[valid]

    # 反投影: 像素坐标 → 3D 坐标
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY

    # 组装点云
    points = np.stack([x, y, z], axis=-1)

    # 获取颜色 (BGR → RGB, 归一化到 0-1)
    colors = color_img[v, u][:, ::-1].astype(np.float32) / 255.0

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def batch_generate():
    color_list = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    print(f"共发现 {len(color_list)} 张图片\n")

    success, skip, fail = 0, 0, 0
    for fname in color_list:
        stem = os.path.splitext(fname)[0]  # "00002"
        save_path = os.path.join(output_dir, f"pig_{stem}.ply")

        # 断点续传：已生成的跳过
        if os.path.exists(save_path):
            skip += 1
            continue

        depth_path = os.path.join(depth_dir, f"{stem}.png")
        if not os.path.exists(depth_path):
            print(f"[-] {fname}: 找不到对应深度图，跳过")
            fail += 1
            continue

        color_img = imread_chinese(os.path.join(color_dir, fname))
        depth_img = imread_depth_chinese(depth_path)

        if color_img is None or depth_img is None:
            print(f"[-] {fname}: 图片读取失败")
            fail += 1
            continue

        # 获取掩膜
        mask = get_mask(color_img)
        if mask is None:
            print(f"[-] {fname}: 分割失败")
            fail += 1
            continue

        # 生成点云
        pcd = depth_to_pointcloud(color_img, depth_img, mask)

        if len(pcd.points) < 100:
            print(f"[-] {fname}: 有效点太少 ({len(pcd.points)})")
            fail += 1
            del pcd; gc.collect()
            continue

        # 保存
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"[+] {fname}: 生成点云 {len(pcd.points)} 个点")
        success += 1

        # 释放内存
        del pcd, color_img, depth_img, mask
        gc.collect()
        torch.cuda.empty_cache()

    if skip > 0:
        print(f"(跳过已存在: {skip})")
    print(f"\n完成! 成功: {success}, 失败: {fail}")
    print(f"点云保存在: {output_dir}")


if __name__ == "__main__":
    batch_generate()
