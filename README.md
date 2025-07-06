# MATLAB2025

MATLAB EXPO 2025 用 - 魚検出システム (Fish Detection System)

質問等は松本(@m-shintaro)まで！

## 概要

このプロジェクトは、YOLO v4 (Tiny variant) を使用した魚検出システムです。水中画像から魚を自動検出するための MATLAB ベースの実装となっています。

## 必要な環境

- MATLAB R2021a 以降
- Computer Vision Toolbox
- Deep Learning Toolbox
- GPU (推奨): CUDA 対応の NVIDIA GPU

## プロジェクト構成

私たちが開発した魚検出 YOLO モデル(`FishDetector-TinyYOLOv4.mat`)は， [OneDrive](https://1drv.ms/u/c/729fc9ffb003509d/EbDqK5vTqglOhK7wzjSUjEcBaaj2lQkdI0_W8KuAYtSFCw?e=4XVojP) からダウンロードしてください．

```
MATLAB2025/
├── FishDetector-TinyYOLOv4.mat  # 学習済み検出モデル
├── TrainNet.m                    # モデル学習スクリプト
├── calculateAP.m                 # 基本的な精度評価
├── calculateInfo.m               # 詳細な精度評価
├── CalculateAdvanced.m           # GPU最適化・マルチ閾値評価
├── TrainModels/                  # その他の学習済みモデル
└── dataset/DeepFish/             # データセット
```

## 学習方法

### 1. データセットの準備

データセットは [DeepFish](https://alzayats.github.io/DeepFish/)を使用しています．
ここからダウンロードしたものを解凍し，そのまま dataset フォルダ(なければ作成し)に入れてください．

```
dataset/DeepFish/
├── Fish/
│   └── [ID]/
│       ├── train/  # 学習用画像(.jpg)とアノテーション(.txt)
│       └── valid/  # 検証用画像(.jpg)とアノテーション(.txt)
├── Nagative_samples/  # 魚が写っていない画像
└── classes.txt  # クラス定義ファイル
```

アノテーションファイル（.txt）の形式：

- YOLO 形式：`class_id x_center y_center width height`（正規化座標）
- 各値は 0-1 の範囲で正規化されています

### 2. モデルの学習

```matlab
% MATLABで以下を実行
TrainNet
```

学習パラメータ（TrainNet.m 内で設定）：

- 入力サイズ: 416×416×3
- バッチサイズ: 4
- エポック数: 50
- 学習率: 0.0001（20 エポックごとに 0.1 倍）
- 最適化手法: Adam
- データ拡張: 色相変換、ランダム反転、スケーリング

## モデルの検証方法

### 1. 基本的な精度評価（Average Precision）

```matlab
% 検出器とデータを読み込む
load('FishDetector-TinyYOLOv4.mat');  % detectorという変数名で読み込まれる
% 検証データを準備（TrainNet.mの該当部分を参照）
% ... validationDataの準備 ...

% APを計算
net = detector;  % calculateAP.mでは'net'という変数名を使用
calculateAP
```

### 2. 詳細な評価指標（精度、再現率、F1 スコア）

```matlab
% 検出器とデータを読み込む
load('FishDetector-TinyYOLOv4.mat');
net = detector;
% ... validationDataの準備 ...

% 詳細な評価を実行
calculateInfo
```

出力される指標：

- 平均適合率 (AP)
- 平均精度 (Mean Precision)
- 平均再現率 (Mean Recall)
- F1 スコア

### 3. マルチ閾値での最適化評価（GPU 対応）

```matlab
% 検出器とデータを読み込む
load('FishDetector-TinyYOLOv4.mat');
% ... validationDataの準備 ...

% 複数の検出閾値で評価
CalculateAdvanced
```

このスクリプトは複数の検出閾値（0.01 ～ 0.18）で評価を行い、最適な閾値を自動的に特定します。

### 4. 複数モデルの性能比較（RankAP）

```matlab
% 複数のモデルファイルを格納したフォルダを指定
modelsFolder = 'path/to/your/models';  % デフォルト: 'C:\Users\s-mat\AppData\Local\Temp\mat_models'
% ... validationDataの準備 ...

% モデルのAP値を計算してランキング表示
RankAP
```

このスクリプトは指定フォルダ内のすべての学習済みモデル（.mat ファイル）に対して：

- Average Precision (AP) を自動計算
- AP 値の降順でランキング表示
- 最も性能の良いモデルを特定

出力例：

```
model1.mat -> AP: 0.89234
model2.mat -> AP: 0.87512
model3.mat -> AP: 0.91045

--- Average Precision Ranking ---
1: model3.mat -> AP: 0.91045
2: model1.mat -> AP: 0.89234
3: model2.mat -> AP: 0.87512
```

## 画像から魚を検出する手順

### 1. 単一画像での検出

```matlab
% 学習済みモデルを読み込む
load('FishDetector-TinyYOLOv4.mat');

% 画像を読み込む
img = imread('your_fish_image.jpg');

% 検出を実行（閾値0.5）
[bboxes, scores, labels] = detect(detector, img, 'Threshold', 0.5);

% 検出結果を画像に描画
detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, scores);

% 結果を表示
figure
imshow(detectedImg)
title(sprintf('検出数: %d', size(bboxes, 1)));
```

### 2. 複数画像での検出

```matlab
% モデルを読み込む
load('FishDetector-TinyYOLOv4.mat');

% 画像フォルダを指定
imageFolder = 'path/to/your/images';
imds = imageDatastore(imageFolder);

% 各画像で検出を実行
while hasdata(imds)
    img = read(imds);
    [bboxes, scores, labels] = detect(detector, img, 'Threshold', 0.5);

    % 結果を表示または保存
    detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, scores);
    figure
    imshow(detectedImg)
    pause(1);  % 1秒待機
end
```

### 3. 検出パラメータの調整

```matlab
% 異なる閾値での検出
threshold = 0.3;  % より多くの検出（誤検出も増える可能性）
[bboxes, scores, labels] = detect(detector, img, 'Threshold', threshold);

% GPU使用の明示的な指定
[bboxes, scores, labels] = detect(detector, img, ...
    'Threshold', 0.5, ...
    'ExecutionEnvironment', 'gpu');
```

## トラブルシューティング

### GPU が使用されない場合

```matlab
% GPUの状態を確認
gpuDevice
% GPUをリセット
gpuDevice(1)
```

### メモリ不足の場合

- バッチサイズを小さくする（TrainNet.m 内の'MiniBatchSize'）
- 入力画像サイズを小さくする

### 検出精度が低い場合

- CalculateAdvanced.m を使用して最適な検出閾値を見つける
- より多くのエポック数で再学習する
- データ拡張のパラメータを調整する
