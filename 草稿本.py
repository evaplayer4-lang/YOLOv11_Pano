import matplotlib

matplotlib.use("Agg")  # 非交互式后端，安全无弹窗，只生成图片

import sys

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = r"D:\git program\YOLOv11-Pano\ultralytics\runs\detect\runs\train\yolo11x_panorama\results1.csv"

try:
    print("1. 正在读取 CSV...")
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    print(f"   读取成功，共 {len(df)} 行")

    required_cols = [
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "val/box_loss",
        "val/cls_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列: {col}")
    print("2. 所有必需列均存在")

    if df[required_cols].isnull().any().any():
        print("   警告：存在缺失值，将丢弃")
        df = df.dropna(subset=required_cols)

    print("3. 绘制损失曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Class Loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Class Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("custom_loss_curve.png", dpi=150, bbox_inches="tight")
    print("   -> 损失曲线已保存为 custom_loss_curve.png")

    print("4. 绘制 mAP 曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)
    plt.savefig("custom_map_curve.png", dpi=150, bbox_inches="tight")
    print("   -> mAP 曲线已保存为 custom_map_curve.png")

    print("5. 程序正常结束")

except Exception as e:
    print(f"\n错误: {e}")
    sys.exit(1)
