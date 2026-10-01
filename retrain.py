import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # 1. 【绝对核心】：加载 last.pt。里面不仅有权重，还有 EMA（指数平滑）模型状态。
    # 绝不能用 best.pt，也不能用官方的 yolo11x.pt！
    model_path = r"D:\git program\YOLOv11-Pano\ultralytics\runs\detect\runs\train\yolo11x_panorama\weights\last.pt"

    model = YOLO(model_path)

    # 2. 启动极端平滑微调 (彻底锁死各项突变因素)
    model.train(
        # --- 基础配置 ---
        # ⚠️ 强烈建议这里写 data.yaml 的“绝对全路径”，彻底杜绝漂移到 coco8 的可能
        data="data.yaml",
        epochs=100,  # 在新阶段里再跑 100 轮 (等效 101-200)
        imgsz=1280,
        batch=4,
        rect=True,
        amp=True,
        device=0,
        workers=4,
        patience=10,
        project="runs/train",
        name="yolo11x_panorama_phase2",  # 存入新文件夹，绝不覆盖你前100轮的心血
        # ==========================================
        # 🔒 核心防断崖参数 (必须严格保持这样)
        # ==========================================
        # 1. 锁死学习率：用极小步长，代替丢失的优化器动量
        optimizer="AdamW",
        lr0=0.00001,  # 极小的十万分之一学习率，绝对不会震碎权重
        lrf=0.1,  # 最终衰减到百万分之一
        warmup_epochs=0.0,  # 【必须为0】绝对关闭预热！预热会让学习率短暂飙升，是断崖的罪魁祸首
        # 2. 锁死数据增强：与你第100轮结束时看到的数据环境完全一致
        mosaic=0.0,  # 【必须为0】关闭马赛克拼接
        mixup=0.0,  # 关闭混叠
        copy_paste=0.0,  # 关闭复制粘贴
        # 3. 明确模式
        resume=False,  # 明确告诉框架：这是一次全新的平滑微调，不读取损坏的旧日志
        exist_ok=True,
    )
