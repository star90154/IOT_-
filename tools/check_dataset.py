#!/usr/bin/env python3
"""檢查 YOLO 資料集：標註格式、座標範圍、類別 ID、缺漏檔案、框尺寸分布。

用法：python tools/check_dataset.py datasets/fabric/data.yaml
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(data_yaml: str):
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(cfg.get("path", Path(data_yaml).parent))
    names = cfg["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    ok = True

    for split in ("train", "val"):
        img_dir = root / cfg[split]
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        cls_cnt, small, bad, empty, no_label = Counter(), 0, 0, 0, 0
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                no_label += 1
                continue
            rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()]
            if not rows:
                empty += 1
            for r in rows:
                if len(r) != 5 or int(r[0]) not in names:
                    bad += 1
                    continue
                c, *v = int(r[0]), *map(float, r[1:])
                if not all(0 <= x <= 1 for x in v):
                    bad += 1
                cls_cnt[c] += 1
                if v[2] * v[3] < 0.0001:  # 小於影像面積 0.01%
                    small += 1
        print(f"== {split}: {len(imgs)} 張影像, {sum(cls_cnt.values())} 個框")
        print(f"   背景(空標註) {empty}, 缺標註檔 {no_label}, 格式錯誤 {bad}, 極小框 {small}")
        for c in sorted(names):
            print(f"   {c:>2} {names[c]:<8} {cls_cnt[c]}")
        if bad or no_label:
            ok = False
    print("檢查通過" if ok else "有問題，請修正後再訓練")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
