import json
import os

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics import YOLO

# ================== 配置 ==================
# BEST_PT = r"D:\git program\YOLOv11-Pano\ultralytics\runs\detect\runs\train\yolo11x_panorama_phase2\weights\best.pt"
BEST_PT = r"D:\git program\YOLOv11-Pano\ultralytics\runs\detect\runs\train\yolo11x_panorama\weights\best.pt"  # 这个模型的效果比上面的更好，所有选择这个
TEST_IMG_DIR = r"D:\BaiduNetdiskDownload\runingdataset\test\images"
TEST_LBL_DIR = r"D:\BaiduNetdiskDownload\runingdataset\test\labels"
CSV_PATH = r"E:\A研究生文献\近邻分析结果_带行人计数_complete.csv"

# 25m 测距默认参数（当 CSV 中某行缺失时使用）
DEFAULT_HEIGHT = 2.32
DEFAULT_PITCH = 2.0

# 推理与评估参数
CONF_THRESH = 0.25
IMGSZ = 1280


# ================== 辅助函数 ==================
def compute_25m_line(img_w, img_h, height, pitch):
    """计算 25 米边界线在每一列的 Y 坐标（越往下 Y 越大）."""
    alpha_deg = np.degrees(np.arctan(height / 25.0))
    x_coords = np.arange(img_w)
    theta_rel = ((x_coords - img_w / 2.0) / img_w) * 2.0 * np.pi
    total_angle = alpha_deg + pitch * np.cos(theta_rel)
    y_line = img_h / 2.0 + total_angle * (img_h / 180.0)
    y_line = np.clip(y_line, 0, img_h - 1)
    return y_line


def filter_boxes_by_25m(boxes_xyxy_conf, img_w, img_h, height, pitch):
    """过滤检测框，只保留底边中点在 25 米线下的框."""
    if len(boxes_xyxy_conf) == 0:
        return []
    y_25 = compute_25m_line(img_w, img_h, height, pitch)
    valid = []
    for box in boxes_xyxy_conf:
        x1, _y1, x2, y2, _conf = box
        bcx = int((x1 + x2) / 2)  # 底边中点 X
        by = int(y2)  # 底边 Y
        if 0 <= bcx < img_w and by >= y_25[bcx]:  # 在 25 米线下方 → 更近
            valid.append(box)
    return valid


'''
使用几何过滤后的结果也不好，不如原来的，然后我的调整都是deepseek进行调整的，后续可以使用Gemini试一下
def filter_boxes_by_25m(boxes_xyxy_conf, img_w, img_h, height, pitch):
    """过滤检测框：先进行几何约束，再进行25米线过滤"""
    if len(boxes_xyxy_conf) == 0:
        return []
    y_25 = compute_25m_line(img_w, img_h, height, pitch)
    valid = []
    for box in boxes_xyxy_conf:
        x1, y1, x2, y2, conf = box
        w = x2 - x1
        h = y2 - y1
        # 几何过滤：只保留宽高比在 (0.3, 3.0) 且面积大于 50 像素的框
        aspect_ratio = w / h if h > 0 else 0
        if not (0.3 < aspect_ratio < 3.0 and (w * h) > 50):
            continue
        # 25米过滤
        bcx = int((x1 + x2) / 2)
        by = int(y2)
        if 0 <= bcx < img_w and by >= y_25[bcx]:
            valid.append(box)
    return valid'''


def compute_detailed_metrics(gt_json_path, pred_json_path, conf_threshold):
    """计算精确率、召回率、F1 和准确率。 使用简化的 IoU 匹配，IoU 阈值设为 0.5。.
    """
    # 加载数据
    with open(gt_json_path) as f:
        gt_data = json.load(f)
    with open(pred_json_path) as f:
        pred_data = json.load(f)

    # 组织真值框
    gt_boxes = {}
    for ann in gt_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_boxes:
            gt_boxes[img_id] = []
        gt_boxes[img_id].append(ann["bbox"])

    # 按置信度排序预测框
    pred_annotations = sorted(pred_data["annotations"], key=lambda x: x["score"], reverse=True)

    TP = 0
    FP = 0
    matched_gt = set()  # 记录已匹配的真值框 (img_id, box_index)

    for pred in pred_annotations:
        if pred["score"] < conf_threshold:
            continue
        img_id = pred["image_id"]
        if img_id not in gt_boxes:
            FP += 1
            continue

        # 寻找当前图片中与预测框 IoU 最大且超过 0.5 的真值框
        max_iou = 0.0
        best_gt_idx = -1
        px1, py1, pw, ph = pred["bbox"]
        px2, py2 = px1 + pw, py1 + ph

        for idx, gt_bbox in enumerate(gt_boxes[img_id]):
            gx1, gy1, gw, gh = gt_bbox
            gx2, gy2 = gx1 + gw, gy1 + gh

            # 计算 IoU
            inter_x1 = max(px1, gx1)
            inter_y1 = max(py1, gy1)
            inter_x2 = min(px2, gx2)
            inter_y2 = min(py2, gy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            area_pred = pw * ph
            area_gt = gw * gh
            union = area_pred + area_gt - inter_area
            iou = inter_area / union if union > 0 else 0.0

            if iou > max_iou:
                max_iou = iou
                best_gt_idx = idx

        if max_iou >= 0.5 and (img_id, best_gt_idx) not in matched_gt:
            TP += 1
            matched_gt.add((img_id, best_gt_idx))
        else:
            FP += 1

    # 总真值框数
    total_gt = sum(len(v) for v in gt_boxes.values())
    FN = total_gt - TP

    # 计算指标
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / total_gt if total_gt > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = TP / (total_gt + FP) if (total_gt + FP) > 0 else 0.0

    return precision, recall, f1, accuracy, TP, FP, FN


# ================== 主程序 ==================
def main():
    # ---------- 1. 加载模型与参数表 ----------
    model = YOLO(BEST_PT)

    # 🎯 可靠的 CSV 读取与列名匹配
    try:
        df_params = pd.read_csv(CSV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df_params = pd.read_csv(CSV_PATH, encoding="gbk")

    if len(df_params.columns) < 5:
        df_params = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8")

    df_params.columns = df_params.columns.str.strip()

    height_col = None
    pitch_col = None
    pname_col = None

    for col in df_params.columns:
        col_lower = col.lower()
        if col_lower == "pname":
            pname_col = col
        # 精确匹配，兼容 'DeviceHeig' 这个拼写错误
        elif col_lower == "deviceheight" or col_lower == "deviceheig":
            height_col = col
        elif col_lower == "pitch":
            pitch_col = col

    if pname_col is None or height_col is None or pitch_col is None:
        print(f"警告：CSV 中缺少必要列。 pname: {pname_col}, height: {height_col}, pitch: {pitch_col}")
        print("请检查 CSV 文件是否存在这些确切的表头。将全部使用默认参数。")
        print("当前读取到的列名为：", list(df_params.columns))
        param_map = {}
    else:
        param_map = {}
        for _, row in df_params.iterrows():
            try:
                pname = str(row[pname_col]).strip()
                height = float(row[height_col])
                pitch = float(row[pitch_col])
                param_map[pname] = (height, pitch)
            except (ValueError, KeyError, TypeError):
                continue
        print(f"成功加载 {len(param_map)} 条相机参数。")

    # ---------- 2. 准备 COCO 数据结构 ----------
    pred_images, pred_anns = [], []
    gt_images, gt_anns = [], []
    pred_ann_id, gt_ann_id = 0, 0
    gt_counts, pred_counts = [], []

    img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"找到 {len(img_files)} 张测试图片，开始评估...")

    for img_id, fname in enumerate(img_files, 1):
        img_path = os.path.join(TEST_IMG_DIR, fname)
        pname = os.path.splitext(fname)[0]
        lbl_path = os.path.join(TEST_LBL_DIR, pname + ".txt")

        height, pitch = param_map.get(pname, (DEFAULT_HEIGHT, DEFAULT_PITCH))

        # ---------- 3. 模型推理 ----------
        results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESH, device=0, verbose=False)[
            0
        ]  # 没有进行增强推理TTA的效果会更好，也不需要动iou阈值，直接使用默认的0.7即可
        # results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESH, device=0, augment=True, verbose=False)[0]
        # results = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRESH, device=0, iou=0.8, verbose=False)[0]
        h_img, w_img = results.orig_shape

        if results.boxes is not None:
            boxes_xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            pred_raw = np.concatenate([boxes_xyxy, confs[:, None]], axis=1)
        else:
            pred_raw = np.empty((0, 5))

        pred_filt = filter_boxes_by_25m(pred_raw, w_img, h_img, height, pitch)

        # ---------- 4. 读取真值框并进行 25 米过滤 ----------
        gt_boxes_xywh = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    if cls != 0:
                        continue
                    x_abs = cx * w_img
                    y_abs = cy * h_img
                    bw_abs = bw * w_img
                    bh_abs = bh * h_img
                    x1 = x_abs - bw_abs / 2
                    y1 = y_abs - bh_abs / 2
                    gt_boxes_xywh.append([x1, y1, bw_abs, bh_abs])

        y_25 = compute_25m_line(w_img, h_img, height, pitch)
        gt_filt_xywh = []
        for box in gt_boxes_xywh:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            bcx, by = int((x1 + x2) / 2), int(y2)
            if 0 <= bcx < w_img and by >= y_25[bcx]:
                gt_filt_xywh.append(box)

        # ---------- 5. 存入 COCO JSON ----------
        pred_images.append({"id": img_id, "file_name": fname, "width": w_img, "height": h_img})
        gt_images.append({"id": img_id, "file_name": fname, "width": w_img, "height": h_img})

        for box in pred_filt:
            x1, y1, x2, y2, conf = box
            w, h = x2 - x1, y2 - y1
            pred_anns.append(
                {
                    "id": pred_ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(conf),
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            pred_ann_id += 1

        for box in gt_filt_xywh:
            x1, y1, w, h = box
            gt_anns.append(
                {
                    "id": gt_ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            gt_ann_id += 1

        gt_counts.append(len(gt_filt_xywh))
        pred_counts.append(len(pred_filt))
        print(f"  [{img_id}/{len(img_files)}] {fname}: GT={len(gt_filt_xywh)}, Pred={len(pred_filt)}")

    # ---------- 6. 保存 JSON 并调用 COCO 评估 ----------
    with open("pred_25m.json", "w") as f:
        json.dump({"images": pred_images, "annotations": pred_anns, "categories": [{"id": 1, "name": "person"}]}, f)
    with open("gt_25m.json", "w") as f:
        json.dump({"images": gt_images, "annotations": gt_anns, "categories": [{"id": 1, "name": "person"}]}, f)

    coco_gt = COCO("gt_25m.json")
    with open("pred_25m.json") as f:
        pred_data = json.load(f)
    coco_dt = coco_gt.loadRes(pred_data["annotations"])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ---------- 7. 计数误差 ----------
    gt_arr = np.array(gt_counts)
    pred_arr = np.array(pred_counts)
    mae = np.mean(np.abs(pred_arr - gt_arr))
    rmse = np.sqrt(np.mean((pred_arr - gt_arr) ** 2))
    print("\n========== 25m 内人数统计误差 ==========")
    print(f"MAE : {mae:.4f}  |  RMSE: {rmse:.4f}")

    # ---------- 8. 详细分类指标 ----------
    print(f"\n========== 详细分类指标 (置信度阈值 = {CONF_THRESH}) ==========")
    precision, recall, f1, accuracy, tp, fp, fn = compute_detailed_metrics("gt_25m.json", "pred_25m.json", CONF_THRESH)
    print(f"True Positive  (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"精确率   (Precision): {precision:.4f}")
    print(f"召回率   (Recall)   : {recall:.4f}")
    print(f"F1 分数  (F1-score) : {f1:.4f}")
    print(f"准确率   (Accuracy) : {accuracy:.4f}  (TP / (Total GT + FP))")


if __name__ == "__main__":
    main()
