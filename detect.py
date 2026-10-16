#!/usr/bin/env python3
"""用訓練好的模型做偵測。

  python detect.py --weights runs/detect/fabric/weights/best.pt --source path/to/images_or_video
  python detect.py --weights best.pt --source 0                # USB 攝影機
  python detect.py --weights best.pt --source rtsp://...       # IP Cam 串流
  python detect.py --weights best.pt --source imgs/ --json out.json   # 另存 JSON 結果（方便給 IoT 端）
"""

import argparse
import json

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None, help="將偵測結果另存成 JSON")
    ap.add_argument("--show", action="store_true", help="即時顯示視窗（需桌面環境）")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        show=args.show,
        stream=True,
    )

    out = []
    for r in results:
        dets = [
            {
                "class": r.names[int(c)],
                "conf": round(float(s), 4),
                "bbox_xyxy": [round(float(v), 1) for v in b],
            }
            for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
        ]
        out.append({"image": r.path, "detections": dets})
        print(f"{r.path}: {len(dets)} 個瑕疵")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"結果已存到 {args.json}")


if __name__ == "__main__":
    main()
