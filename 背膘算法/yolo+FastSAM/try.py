from ultralytics import YOLO, FastSAM
import cv2
import os
import numpy as np


# --- 1. 支持中文路径的读写函数（防 OpenCV 崩溃） ---
def imread_chinese(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_chinese(path, img):
    cv2.imencode('.jpg', img)[1].tofile(path)


# --- 2. 初始化模型 ---
print("正在加载模型...")
detect_model = YOLO('yolo11n.pt')
segment_model = FastSAM('FastSAM-s.pt')

# --- 3. 路径配置 ---
# 请确认这是你硬盘上实际的路径
input_dir = "G:/耿来硕数据/DK/nkyfirst/2-1/depth_cz_r_d_p/master/color/"
output_dir = "G:/耿来硕数据/DK/nkyfirst/2-1/depth_cz_r_d_p/master/detect_results/"
os.makedirs(output_dir, exist_ok=True)


def batch_process():
    if not os.path.exists(input_dir):
        print(f"找不到输入文件夹: {input_dir}")
        return

    img_list = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    print(f"共发现 {len(img_list)} 张图片，开始跑自动化流程...\n")

    for img_name in img_list:
        full_path = os.path.join(input_dir, img_name)
        img = imread_chinese(full_path)

        if img is None:
            print(f"图片读取失败: {img_name}")
            continue

        # --- A阶段：YOLO 开放视野检测 ---
        # 核心改动 1：去掉了 classes 限制，让它检测所有目标
        # 核心改动 2：conf 调低到 0.15，宁可错杀一千，不放过一个色块
        det_results = detect_model.predict(img, conf=0.15, device='cuda', verbose=False)[0]

        if len(det_results.boxes) == 0:
            # YOLO 检测失败时，猪往往占满画面，直接用中心区域作为 bbox
            h, w = img.shape[:2]
            margin = 0.05  # 留 5% 边距
            best_box = np.array([w * margin, h * margin, w * (1 - margin), h * (1 - margin)])
            print(f"[~] {img_name}: YOLO未检测到，用全图框兜底")
        else:
            # --- 核心改动 3：物理外挂——取面积最大的框 ---
            areas = (det_results.boxes.xywh[:, 2] * det_results.boxes.xywh[:, 3]).cpu().numpy()
            max_idx = np.argmax(areas)
            best_box = det_results.boxes.xyxy[max_idx].cpu().numpy()

        # --- B阶段：FastSAM 先验分割 ---
        seg_results = \
        segment_model.predict(img, bboxes=best_box, device='cuda', retina_masks=True, conf=0.4, verbose=False)[0]

        # --- 可视化与保存 ---
        # 画出 YOLO 给出的最大检测框（红色）
        cv2.rectangle(img, (int(best_box[0]), int(best_box[1])),
                      (int(best_box[2]), int(best_box[3])), (0, 0, 255), 2)

        # 叠加上 FastSAM 抠出的掩膜（半透明绿色）
        if seg_results.masks is not None:
            mask = seg_results.masks.data[0].cpu().numpy()
            # 创建一个绿色遮罩
            color_mask = np.zeros_like(img)
            color_mask[mask == 1] = [0, 255, 0]
            # 将遮罩和原图混合
            img = cv2.addWeighted(img, 1.0, color_mask, 0.4, 0)
            print(f"[+] {img_name}: 识别与分割成功")
        else:
            print(f"[!] {img_name}: YOLO 给了框，但 FastSAM 分割失败")

        # 保存结果
        save_path = os.path.join(output_dir, f"res_{img_name}")
        imwrite_chinese(save_path, img)

    print(f"\n全部处理完成！请去 {output_dir} 查看结果。")


if __name__ == "__main__":
    batch_process()