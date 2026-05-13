# BirdCLEF+ 2026 — コンペガイド & 実験スターター

> **競技URL**: https://www.kaggle.com/competitions/birdclef-2026/overview  
> **作成日**: 2026-05-13

---

## 目次

1. [コンペ概要](#1-コンペ概要)
2. [タスク詳細](#2-タスク詳細)
3. [データセット](#3-データセット)
4. [評価指標](#4-評価指標)
5. [制約条件](#5-制約条件)
6. [タイムライン](#6-タイムライン)
7. [アプローチ戦略](#7-アプローチ戦略)
8. [実験環境セットアップ](#8-実験環境セットアップ)
9. [ベースラインパイプライン実装](#9-ベースラインパイプライン実装)
10. [実験管理テンプレート](#10-実験管理テンプレート)
11. [参考リソース](#11-参考リソース)

---

## 1. コンペ概要

| 項目 | 内容 |
|------|------|
| **主催** | Cornell Lab of Ornithology / LifeCLEF |
| **テーマ** | 南米パンタナールの音響データから野生生物種を識別 |
| **対象地域** | パンタナール湿地帯（ブラジル・周辺国、150,000 km²超） |
| **問題種別** | 多ラベル分類（マルチラベル）|
| **識別対象種数** | **234種** |
| **賞金** | $50,000（Kaggle）+ $5,000（論文トラック） |

### 背景・動機

パンタナールは世界最大の熱帯湿地であり、650種以上の鳥類を含む膨大な生物多様性を誇る一方、定期的な現地調査が困難な地域です。1,000台の**パッシブ音響モニタリング（PAM）レコーダー**が連続稼働しており、その膨大な音声データを人手で確認することは不可能です。

本コンペでは、この音声データから**自動で種を検出・識別するMLモデル**を構築することで、保全活動を支援します。

---

## 2. タスク詳細

### 入力
- 野外収録された**連続音声（サウンドスケープ）**
- **5秒ウィンドウ**に分割して推論

### 出力
- 各5秒セグメントに対して、**234種それぞれの存在確率**（0〜1）

### 難しさのポイント
- **ドメインシフト**: 学習音声（Xeno-canto等）と推論音声（野外PAM）の音響特性の違い
- **クラス不均衡**: 希少種は学習サンプルが極めて少ない
- **マルチラベル**: 複数種が同時に鳴いている
- **背景ノイズ**: 雨音、風、虫音等の混入
- **CPU推論制約**: GPU使用不可、90分以内に推論完了が必須

---

## 3. データセット

### ファイル構成

```
birdclef-2026/
├── train_audio/               # 学習用音声（種別フォルダ分け）
│   └── {species_code}/
│       └── *.ogg
├── train_soundscapes/         # 野外収録サウンドスケープ（ラベル付き）
│   └── *.ogg
├── test_soundscapes/          # テスト用サウンドスケープ
│   └── *.ogg
├── train_metadata.csv         # 学習音声のメタデータ
├── train_soundscape_labels.csv # サウンドスケープラベル（今年は専門家アノテーション付き）
├── taxonomy.csv               # 種の分類情報
└── sample_submission.csv      # 提出フォーマットサンプル
```

### 各データの役割

| データ | 用途 | 特徴 |
|--------|------|------|
| `train_audio` | 主要学習データ | Xeno-canto等からの高品質録音、種ラベルあり |
| `train_soundscapes` | ドメイン適応・検証 | テストに最も近いドメイン。**今年は専門家アノテーション付き** |
| `test_soundscapes` | 推論対象 | ラベルなし、5秒ごとに予測 |
| `sample_submission.csv` | 提出形式確認 | 行=`{soundscape_id}_{seconds}`、列=234種 |

### Day 1 に確認すべき項目

- [ ] 各種のサンプル数分布（クラス不均衡の把握）
- [ ] 音声の録音時間・サンプリングレート
- [ ] `train_soundscapes` のアノテーション品質
- [ ] `taxonomy.csv` の分類体系
- [ ] `sample_submission.csv` の行形式

---

## 4. 評価指標

### **Macro-averaged ROC-AUC**

$$\text{Score} = \frac{1}{|C|} \sum_{c \in C} \text{AUC}(c)$$

- 各クラス（種）ごとにROC-AUCを計算し、**マクロ平均**を取る
- **真陽性が存在しないクラスはスキップ**される（テストセットで出現しない種は無視）
- 閾値に依存しないため、確率のランキング品質が重要

### 実装例（ローカル検証用）

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def macro_auc(y_true, y_pred):
    """真陽性が存在するクラスのみでmacro AUCを計算"""
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:  # 正例が存在するクラスのみ
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
    return np.mean(aucs)
```

---

## 5. 制約条件

| 制約 | 内容 |
|------|------|
| **実行環境** | **CPUノートブックのみ**（GPU不可） |
| **実行時間** | **90分以内** |
| **提出形式** | Kaggle Code Competition（推論コードを提出） |
| **外部データ** | 規定内で使用可（Xeno-canto、BirdNETモデル等） |

### CPU推論対策

- モデルサイズを抑える（EfficientNet-B0〜B2、MobileNet等）
- 量子化（INT8）やONNXへの変換を検討
- 推論バッチサイズの最適化
- 音声前処理のキャッシュ化

---

## 6. タイムライン

| 日付 | イベント |
|------|----------|
| 2026-03-11 | **コンペ開始** |
| 2026-05-27 | エントリー締切 / チームマージ締切 |
| **2026-06-03** | **最終提出締切** ← 残り約3週間！ |
| 2026-06-17 | 論文（Working Note）提出締切 |
| 2026-06-24 | 採択通知 |
| 2026-07-06 | Camera-ready締切 |

---

## 7. アプローチ戦略

### 全体フロー

```
音声ファイル
    ↓
[前処理] サンプリングレート統一 → 5秒クリップ分割
    ↓
[特徴抽出] Mel Spectrogram生成
    ↓
[モデル] CNN / EfficientNet / BirdNET等
    ↓
[後処理] Sigmoid → 確率出力 → アンサンブル
    ↓
提出 (5秒ウィンドウ × 234種)
```

### フェーズ1: データ理解（推奨 1〜2日）

1. データ量・クラス分布のEDA
2. サウンドスケープのドメイン確認（`train_soundscapes` vs `train_audio`）
3. `taxonomy.csv` から種の分類整理
4. 検証設計（録音場所・日付でGroupKFold推奨）

### フェーズ2: ベースライン構築

**推奨アーキテクチャ（過去実績）**

| モデル | 特徴 | CPU推論速度 |
|--------|------|-------------|
| EfficientNet-B0/B1 | 軽量・高精度バランス | ◎ |
| ResNet18/34 | シンプルで安定 | ◎ |
| MobileNetV3 | 超軽量 | ◎◎ |
| EfficientNet-B4 | 精度重視（CPU遅め） | △ |

**音声特徴量**

| 特徴量 | 推奨パラメータ |
|--------|--------------|
| Mel Spectrogram | n_mels=128, hop_length=320, n_fft=1024 |
| サンプリングレート | 32,000 Hz |
| 音声長 | 5秒（160,000サンプル） |

### フェーズ3: 精度向上テクニック

- **データ拡張**: MixUp、SpecAugment、時間軸シフト、ピッチシフト
- **疑似ラベル**: `test_soundscapes`へのpseudo-label付与
- **BirdNETの活用**: 事前学習済みモデルを転移学習のベースに
- **アンサンブル**: 複数モデルの確率の平均・最大値
- **閾値最適化**: 検証データでのpost-processing

### 検証設計（重要）

```python
# NG: ランダム分割（データリーク・ドメインシフトを見逃す）
# OK: 録音サイト・日付でのGroupKFold

from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = metadata['site_id']  # または recording_date
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    ...
```

---

### Kaggle APIでのデータダウンロード

```bash
# kaggle.jsonを~/.kaggle/に配置後
kaggle competitions download -c birdclef-2026
unzip birdclef-2026.zip -d data/raw/
```

---

## 8. ベースラインパイプライン実装

### `src/dataset.py`

```python
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

class BirdCLEFDataset(Dataset):
    """BirdCLEF 2026 学習用データセット"""
    
    SR = 32000          # サンプリングレート
    DURATION = 5        # クリップ長（秒）
    N_MELS = 128        # Mel bins
    N_FFT = 1024
    HOP_LENGTH = 320    # 160,000 / 500 frames
    
    def __init__(self, df, audio_dir, species_list, transform=None, is_train=True):
        self.df = df
        self.audio_dir = audio_dir
        self.species_list = species_list
        self.species2idx = {s: i for i, s in enumerate(species_list)}
        self.transform = transform
        self.is_train = is_train
        self.num_classes = len(species_list)
    
    def __len__(self):
        return len(self.df)
    
    def load_audio(self, path):
        """音声読み込み・5秒クリップ化"""
        audio, sr = librosa.load(path, sr=self.SR, mono=True)
        target_len = self.SR * self.DURATION
        
        if len(audio) < target_len:
            # パディング
            audio = np.pad(audio, (0, target_len - len(audio)))
        elif self.is_train:
            # 学習時はランダムクロップ
            start = np.random.randint(0, len(audio) - target_len)
            audio = audio[start:start + target_len]
        else:
            # 推論時は先頭から
            audio = audio[:target_len]
        
        return audio
    
    def audio_to_melspec(self, audio):
        """Mel Spectrogram変換"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SR,
            n_mels=self.N_MELS,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            fmin=20,
            fmax=16000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # 正規化 [-1, 1]
        mel_db = (mel_db + 80) / 80  # -80dBが下限の場合
        mel_db = np.clip(mel_db, 0, 1)
        return mel_db.astype(np.float32)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_dir, row['filename'])
        
        audio = self.load_audio(filepath)
        
        if self.transform:
            audio = self.transform(audio, sample_rate=self.SR)
        
        mel = self.audio_to_melspec(audio)
        # (1, H, W) → 3chに複製してImageNet事前学習モデルに対応
        mel = np.stack([mel, mel, mel], axis=0)
        
        # ラベル生成（マルチラベル）
        label = np.zeros(self.num_classes, dtype=np.float32)
        species = row['primary_label']
        if species in self.species2idx:
            label[self.species2idx[species]] = 1.0
        
        return torch.tensor(mel), torch.tensor(label)
```

### `src/model.py`

```python
import torch
import torch.nn as nn
import timm

class BirdCLEFModel(nn.Module):
    """EfficientNet-B0ベースのBirdCLEFモデル"""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=234, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # headを除去
            global_pool='avg',
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits  # BCEWithLogitsLossを使うためSigmoid不要


def get_model(cfg):
    return BirdCLEFModel(
        model_name=cfg['model_name'],
        num_classes=cfg['num_classes'],
        pretrained=cfg['pretrained'],
    )
```

### `src/train.py`

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc_score(y_true, y_pred):
    """コンペ評価指標：真陽性クラスのみでmacro AUC"""
    aucs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    auc = macro_auc_score(all_targets, all_preds)
    
    return total_loss / len(loader), auc


def run_training(model, train_loader, val_loader, cfg, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs']
    )
    
    best_auc = 0.0
    best_model_path = f"outputs/{cfg['model_name']}_best.pth"
    
    for epoch in range(cfg['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Best model saved (AUC: {best_auc:.4f})")
    
    print(f"\nBest Val AUC: {best_auc:.4f}")
    return best_model_path
```

### `configs/baseline.yaml`

```yaml
# ベースライン設定
model_name: efficientnet_b0
num_classes: 234
pretrained: true

# 学習設定
epochs: 30
batch_size: 32
lr: 1.0e-3
weight_decay: 1.0e-4

# データ
sample_rate: 32000
duration: 5
n_mels: 128
n_fft: 1024
hop_length: 320

# 検証
n_folds: 5
fold: 0

# 出力
output_dir: outputs/
```

### メインスクリプト（`train_baseline.py`）

```python
import yaml
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import BirdCLEFDataset
from src.model import get_model
from src.train import run_training

# 設定読み込み
with open('configs/baseline.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# メタデータ読み込み
metadata = pd.read_csv('data/raw/train_metadata.csv')
species_list = sorted(metadata['primary_label'].unique().tolist())
print(f"Total species: {len(species_list)}")

# Fold分割（録音IDでグループ化してリーク防止）
sgkf = StratifiedGroupKFold(n_splits=cfg['n_folds'])
metadata['fold'] = -1
for fold, (_, val_idx) in enumerate(
    sgkf.split(metadata, metadata['primary_label'], groups=metadata['filename'])
):
    metadata.loc[val_idx, 'fold'] = fold

train_df = metadata[metadata['fold'] != cfg['fold']].reset_index(drop=True)
val_df   = metadata[metadata['fold'] == cfg['fold']].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Dataset / DataLoader
train_ds = BirdCLEFDataset(train_df, 'data/raw/train_audio', species_list, is_train=True)
val_ds   = BirdCLEFDataset(val_df,   'data/raw/train_audio', species_list, is_train=False)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=4)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

# モデル学習
model = get_model(cfg).to(device)
best_path = run_training(model, train_loader, val_loader, cfg, device)
print(f"Best model: {best_path}")
```

---

## 9. 実験管理

### 実験チェックリスト

**フェーズ1（ベースライン）**
- [ ] EDA完了・データメモ作成
- [ ] Mel Spectrogram可視化確認
- [ ] ベースラインモデル学習・スコア確認
- [ ] 検証スコアとLBスコアの相関確認

**フェーズ2（精度向上）**
- [ ] SpecAugment追加
- [ ] MixUp / CutMix試行
- [ ] `train_soundscapes`を学習データに追加
- [ ] より大きなモデル（EfficientNet-B2/B3）
- [ ] BirdNET転移学習
- [ ] 疑似ラベル付与

**フェーズ3（CPU最適化）**
- [ ] 推論時間の計測（90分以内確認）
- [ ] モデル量子化（INT8）
- [ ] ONNXエクスポート
- [ ] バッチサイズ最適化
- [ ] アンサンブル（時間内に収まる組み合わせ）

---

## 10. 参考リソース

### 過去のBirdCLEF優勝解法（重要）
- [BirdCLEF 2024 Winner's Solution](https://www.kaggle.com/competitions/birdclef-2024/discussion)
- [BirdCLEF 2023 Top Solutions](https://www.kaggle.com/competitions/birdclef-2023/discussion)

### 参考ライブラリ・モデル
- [BirdNET](https://github.com/kahst/BirdNET-Analyzer) — Cornell Lab製の事前学習済み鳥類音響識別モデル
- [timm](https://github.com/huggingface/pytorch-image-models) — 豊富な事前学習済みCNNモデル
- [librosa](https://librosa.org/) — 音声特徴抽出
- [audiomentations](https://github.com/iver56/audiomentations) — 音声データ拡張

### Kaggleディスカッション
- [BirdCLEF+ 2026 Discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion)
- スターターノートブック・EDAノートブックを必ず確認

### 論文
- Kahl et al. (2021) "BirdNET: A deep learning solution for avian diversity monitoring" — 基礎論文
- SpecAugment (Park et al., 2019) — 音声拡張の定番手法

---

> **⚠️ 残り期間について**  
> 最終提出締切は **2026年6月3日** です（本ドキュメント作成時点から約3週間）。  
> 優先度：検証設計 > ベースライン構築 > CPU推論最適化 > 精度向上  
> まず動くパイプラインを素早く作り、提出してLBスコアを確認することが最優先です。
