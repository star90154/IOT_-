#!/usr/bin/env python3
"""把原始標註轉成 YOLO 格式並切分 train/val。

自動判斷標註格式：
  * 天池布匹瑕疵 JSON：[{"name": "xx.jpg", "defect_name": "破洞", "bbox": [x1, y1, x2, y2]}, ...]
  * COCO JSON：{"images": [...], "annotations": [...], "categories": [...]}
  * Pascal VOC XML 資料夾

用法：
  python tools/convert_to_yolo.py --src /path/to/guangdong_round1_train2 --out datasets/fabric
  python tools/convert_to_yolo.py --src ... --out ... --merge20 --normal-ratio 0.2
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 天池布匹瑕疵初賽官方計分時的 34 類 -> 20 類合併規則
MERGE20 = {
    "破洞": "破洞",
    "水渍": "污渍", "油渍": "污渍", "污渍": "污渍",
    "三丝": "三丝",
    "结头": "结头",
    "花板跳": "花板跳",
    "百脚": "百脚",
    "毛粒": "毛粒",
    "粗经": "粗经",
    "松经": "松经",
    "断经": "断经",
    "吊经": "吊经",
    "粗维": "粗维",
    "纬缩": "纬缩",
    "浆斑": "浆斑",
    "整经结": "整经结",
    "星跳": "星跳", "跳花": "星跳",
    "断氨纶": "断氨纶",
    "稀密档": "档类", "浪纹档": "档类", "色差档": "档类",
    "磨痕": "痕类", "轧痕": "痕类", "修痕": "痕类", "烧毛痕": "痕类",
    "死皱": "其他", "云织": "其他", "双纬": "其他", "双经": "其他",
    "跳纱": "其他", "筘路": "其他", "纬纱不良": "其他",
}


def index_images(src: Path) -> dict:
    """以檔名建立索引（遞迴搜尋），標註裡的檔名不必含路徑。"""
    idx = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name, p)
    return idx


def image_size(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def find_annotation(src: Path):
    jsons = [p for p in src.rglob("*.json") if not p.name.startswith(".")]
    if jsons:
        # 優先挑名字帶 anno 的
        jsons.sort(key=lambda p: ("anno" not in p.name.lower(), -p.stat().st_size))
        return "json", jsons[0]
    xmls = list(src.rglob("*.xml"))
    if xmls:
        return "voc", xmls
    raise SystemExit(f"在 {src} 找不到 .json 或 .xml 標註檔")


def load_records(src: Path, img_idx: dict):
    """回傳 {影像Path: [(類別名, x1, y1, x2, y2), ...]}"""
    kind, ann = find_annotation(src)
    recs = defaultdict(list)
    missing = 0

    if kind == "json":
        print(f"[info] 使用標註檔：{ann}")
        data = json.loads(ann.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "annotations" in data and "images" in data:  # COCO
            cats = {c["id"]: c["name"] for c in data.get("categories", [])}
            imgs = {i["id"]: i["file_name"] for i in data["images"]}
            for a in data["annotations"]:
                fn = Path(imgs[a["image_id"]]).name
                if fn not in img_idx:
                    missing += 1
                    continue
                x, y, w, h = a["bbox"]
                recs[img_idx[fn]].append((cats.get(a["category_id"], str(a["category_id"])), x, y, x + w, y + h))
        elif isinstance(data, list):  # 天池格式
            for a in data:
                fn = Path(a["name"]).name
                if fn not in img_idx:
                    missing += 1
                    continue
                x1, y1, x2, y2 = a["bbox"]
                recs[img_idx[fn]].append((str(a["defect_name"]), x1, y1, x2, y2))
        else:
            raise SystemExit("無法辨識的 JSON 標註格式")
    else:
        print(f"[info] 使用 VOC XML，共 {len(ann)} 個檔案")
        for x in ann:
            root = ET.parse(x).getroot()
            fn = Path(root.findtext("filename", "")).name
            if fn not in img_idx:
                missing += 1
                continue
            for o in root.iter("object"):
                b = o.find("bndbox")
                recs[img_idx[fn]].append((
                    o.findtext("name"),
                    float(b.findtext("xmin")), float(b.findtext("ymin")),
                    float(b.findtext("xmax")), float(b.findtext("ymax")),
                ))
    if missing:
        print(f"[warn] 有 {missing} 筆標註找不到對應影像，已略過")
    return recs


def to_yolo_lines(boxes, w, h, cls_map):
    lines = []
    for name, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0.0, min(float(x1), w)), max(0.0, min(float(x2), w))))
        y1, y2 = sorted((max(0.0, min(float(y1), h)), max(0.0, min(float(y2), h))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cls_map[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="解壓後的資料集根目錄")
    ap.add_argument("--out", required=True, type=Path, help="輸出 YOLO 資料集目錄")
    ap.add_argument("--val", type=float, default=0.2, help="驗證集比例")
    ap.add_argument("--merge20", action="store_true", help="套用天池 34->20 類合併")
    ap.add_argument("--normal-dir", default="normal_Images", help="無瑕疵影像資料夾名稱（當背景樣本）")
    ap.add_argument("--normal-ratio", type=float, default=0.1,
                    help="背景影像數量 = 瑕疵影像數 x 此比例；0 表示不用")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="複製影像（預設用 symlink 省空間）")
    args = ap.parse_args()

    random.seed(args.seed)
    img_idx = index_images(args.src)
    print(f"[info] 共找到 {len(img_idx)} 張影像")
    recs = load_records(args.src, img_idx)

    if args.merge20:
        unknown = {b[0] for bs in recs.values() for b in bs} - MERGE20.keys()
        if unknown:
            print(f"[warn] 以下類別不在合併表中，保持原名：{sorted(unknown)}")
        recs = {k: [(MERGE20.get(b[0], b[0]),) + tuple(b[1:]) for b in v] for k, v in recs.items()}

    counter = Counter(b[0] for bs in recs.values() for b in bs)
    names = sorted(counter)
    cls_map = {n: i for i, n in enumerate(names)}

    # 背景（無瑕疵）影像
    normals = []
    if args.normal_ratio > 0:
        normals = [p for p in img_idx.values() if args.normal_dir in p.parts and p not in recs]
        random.shuffle(normals)
        normals = normals[: int(len(recs) * args.normal_ratio)]

    items = list(recs.items()) + [(p, []) for p in normals]
    random.shuffle(items)
    n_val = int(len(items) * args.val)
    splits = {"val": items[:n_val], "train": items[n_val:]}

    if args.out.exists():
        shutil.rmtree(args.out)
    for split, its in splits.items():
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        for img, boxes in its:
            dst = args.out / "images" / split / img.name
            if args.copy:
                shutil.copy2(img, dst)
            else:
                dst.symlink_to(img.resolve())
            lines = to_yolo_lines(boxes, *image_size(img), cls_map) if boxes else []
            (args.out / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))

    yaml_lines = [f"path: {args.out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"[done] train={len(splits['train'])} val={len(splits['val'])} (含背景 {len(normals)})")
    print(f"[done] {len(names)} 類，框數：")
    for n in names:
        print(f"    {cls_map[n]:>2} {n:<8} {counter[n]}")
    print(f"[done] data.yaml -> {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
