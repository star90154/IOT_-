#!/usr/bin/env python3
"""Roboflow 上傳 / 下載。

上傳已轉好的 YOLO 資料集到 Roboflow（在網頁上檢視、修正或新增標註）：
  python tools/roboflow_sync.py upload --data datasets/fabric --workspace my-ws --project fabric-defect

在 Roboflow 產生版本後下載回來訓練：
  python tools/roboflow_sync.py download --workspace my-ws --project fabric-defect --version 1 --out datasets/rf

API key 從環境變數 ROBOFLOW_API_KEY 讀取（Colab 可放在左側「Secrets」）。
"""

import argparse
import os
import sys


def get_rf():
    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("請先設定環境變數 ROBOFLOW_API_KEY")
    return Roboflow(api_key=key)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload")
    up.add_argument("--data", required=True, help="YOLO 資料集目錄（含 data.yaml）")
    up.add_argument("--workspace", required=True)
    up.add_argument("--project", required=True)
    up.add_argument("--workers", type=int, default=8)

    dl = sub.add_parser("download")
    dl.add_argument("--workspace", required=True)
    dl.add_argument("--project", required=True)
    dl.add_argument("--version", type=int, required=True)
    dl.add_argument("--format", default="yolov8", help="YOLO txt 格式；yolov8/yolov11/yolov12 內容相同")
    dl.add_argument("--out", default="datasets/roboflow")

    args = ap.parse_args()
    ws = get_rf().workspace(args.workspace)

    if args.cmd == "upload":
        ws.upload_dataset(
            args.data,
            args.project,
            num_workers=args.workers,
            project_type="object-detection",
            dataset_format="yolov8",
        )
        print("上傳完成，請到 Roboflow 網頁檢查標註並 Generate 新版本")
    else:
        ds = ws.project(args.project).version(args.version).download(args.format, location=args.out)
        print(f"已下載到 {ds.location}，data.yaml 位於 {os.path.join(ds.location, 'data.yaml')}")


if __name__ == "__main__":
    main()
