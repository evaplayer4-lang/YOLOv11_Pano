from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\git program\YOLOv11-Pano\ultralytics\yolo11x.pt")
    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=1280,
        batch=4,
        rect=True,
        amp=True,
        device=0,
        workers=4,          # 可以保留，加上保护后就能正常使用
        patience=10,
        project='runs/train',
        name='yolo11x_panorama',
        exist_ok=True
    )