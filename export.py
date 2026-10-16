#!/usr/bin/env python3
"""匯出模型以便部署到邊緣裝置。

  python export.py --weights best.pt --format onnx          # 通用，ONNX Runtime / OpenCV DNN
  python export.py --weights best.pt --format engine --half # Jetson / NVIDIA GPU (TensorRT)，需在目標裝置上匯出
  python export.py --weights best.pt --format openvino      # Intel CPU / NPU
  python export.py --weights best.pt --format tflite        # Raspberry Pi / Android
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data", default=None, help="INT8 量化校正用的 data.yaml")
    args = ap.parse_args()

    path = YOLO(args.weights).export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        data=args.data,
        simplify=True,
    )
    print(f"已匯出：{path}")


if __name__ == "__main__":
    main()
