#!/usr/bin/env python3
"""訓練 YOLOv12。

  python train.py --data datasets/fabric/data.yaml --model yolo12s.pt --imgsz 1024 --epochs 100

布匹影像為 2446x1000 的長條圖、瑕疵多為細長或很小，建議 imgsz >= 1024。
Colab T4 (15GB)：yolo12s + imgsz 1024 約可用 batch 8；記憶體不足時調小 batch 或用 -1 自動。
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo12s.pt", help="yolo12n/s/m/l/x.pt（預訓練）或 yolo12s.yaml（從頭訓練）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=float, default=8, help="-1 = 自動依 GPU 記憶體決定")
    ap.add_argument("--device", default=None, help="0 / cpu / 0,1")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="fabric")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="從 last.pt 續訓（--model 給 last.pt 路徑）")
    args = ap.parse_args()

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
        return

    batch = int(args.batch) if args.batch >= 1 else args.batch
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cos_lr=True,
        close_mosaic=10,
        # 布匹紋理上下、左右翻轉都合理
        fliplr=0.5,
        flipud=0.5,
        # 色差類瑕疵對顏色敏感，降低 HSV 增強
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.3,
        plots=True,
    )
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=batch, device=args.device)
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
