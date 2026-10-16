# YOLOv12 布匹瑕疵偵測

用 YOLOv12（Ultralytics）在 Google Colab 上訓練布匹瑕疵偵測模型，可搭配 Roboflow 管理標註。

資料集：[Kaggle hcyl1025/guangdong-round1-train2](https://www.kaggle.com/datasets/hcyl1025/guangdong-round1-train2)
（天池「廣東工業智造」布匹瑕疵檢測初賽，影像 2446×1000）

## 快速開始（Colab）

1. 在 Colab 開啟 `notebooks/YOLOv12_Colab.ipynb`
   （GitHub 上打開這個檔案 → 網址的 `github.com` 換成 `githubtocolab.com`）
2. 執行階段選 **T4 GPU**
3. 由上往下執行各個 cell

## 專案結構

```
notebooks/YOLOv12_Colab.ipynb   Colab 一條龍流程
tools/convert_to_yolo.py        原始標註 → YOLO 格式 + 切分 train/val + 產生 data.yaml
tools/check_dataset.py          檢查標註（格式、座標範圍、類別分布、極小框）
tools/roboflow_sync.py          上傳到 Roboflow 修正標註 / 下載 Roboflow 版本
train.py                        訓練（含布匹適用的資料增強設定）
detect.py                       偵測圖片 / 影片 / 攝影機 / RTSP，可輸出 JSON
export.py                       匯出 ONNX / TensorRT / OpenVINO / TFLite
```

## 本機指令

```bash
pip install -r requirements.txt

# 1. 轉換（自動判斷天池 JSON / COCO / VOC XML）
python tools/convert_to_yolo.py --src /path/to/guangdong_round1_train2 --out datasets/fabric --merge20 --normal-ratio 0.1
python tools/check_dataset.py datasets/fabric/data.yaml

# 2. 訓練
python train.py --data datasets/fabric/data.yaml --model yolo12s.pt --imgsz 1024 --epochs 100 --batch 8

# 3. 偵測
python detect.py --weights runs/detect/fabric/weights/best.pt --source test_images/ --json pred.json

# 4. 匯出
python export.py --weights runs/detect/fabric/weights/best.pt --format onnx
```

## 類別

天池原始資料有 34 種瑕疵，`--merge20` 會套用官方計分時的合併規則：

| 合併後 | 原始類別 |
|---|---|
| 污渍 | 水渍、油渍、污渍 |
| 星跳 | 星跳、跳花 |
| 档类 | 稀密档、浪纹档、色差档 |
| 痕类 | 磨痕、轧痕、修痕、烧毛痕 |
| 其他 | 死皱、云织、双纬、双经、跳纱、筘路、纬纱不良 |
| 其餘 | 破洞、三丝、结头、花板跳、百脚、毛粒、粗经、松经、断经、吊经、粗维、纬缩、浆斑、整经结、断氨纶（各自一類） |

## Roboflow 標註流程（選用）

1. 先用 `convert_to_yolo.py` 轉好資料
2. `ROBOFLOW_API_KEY=xxx python tools/roboflow_sync.py upload --data datasets/fabric --workspace <ws> --project fabric-defect`
3. 在 Roboflow 網頁修正錯誤標註、加入新圖片並標註，然後 Generate 新版本
4. `python tools/roboflow_sync.py download --workspace <ws> --project fabric-defect --version 1 --out datasets/rf`
5. 用 `datasets/rf/data.yaml` 訓練

## 調參建議

- 瑕疵很小、又是細長形：`imgsz` 用 1024～1280；GPU 記憶體夠的話，模型改用 `yolo12m`
- 類別很不平衡（某些類只有幾十個框）：用 `--merge20`，或對少數類別補標註、做資料增強
- 誤報太多：調高 `--normal-ratio`（加入更多無瑕疵背景圖）或在偵測時調高 `--conf`
