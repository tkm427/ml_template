# CLAUDE.md

## プロジェクト概要

- docs/コンペティション名 にコンペティションの概要が書かれている。

---

## 実行環境

**基本的にすべての実行は Docker コンテナ内で行う。**

```bash
# 起動
docker compose up -d

# コンテナに入る
docker compose exec workspace bash

# スクリプト直接実行（コンテナに入らない場合）
docker compose exec workspace python scripts/train.py

# 停止
docker compose down

# ログ確認
docker compose logs -f
```

| サービス | URL |
|----------|-----|
| JupyterLab | http://localhost:8888 |
| MLflow UI | http://localhost:5000 |

---

## ディレクトリ構成ルール

```
bird_ml/
├── conf/
│   ├── config.yaml        # Hydra エントリポイント
│   ├── model/             # モデルアーキテクチャ設定
│   ├── data/              # データ・前処理設定
│   └── train/             # 学習設定（lr, epoch 等）
├── src/
│   ├── dataset.py         # Dataset クラス
│   ├── model.py           # モデル定義
│   ├── train.py           # 学習ループ
│   ├── inference.py       # 推論（CPU最適化）
│   └── utils/
│       └── mlflow_utils.py
├── scripts/               # 学習・推論実行スクリプト
├── notebooks/             # EDA・可視化のみ（本番コードを置かない）
├── data/
│   ├── raw/               # Kaggle 生データ（Git 管理外）
│   └── processed/         # 前処理済みデータ（Git 管理外）
└── outputs/               # チェックポイント・submission（Git 管理外）
```

---

## 設定管理（Hydra）

- すべての実験パラメータは `conf/` 以下の YAML で管理する
- 実験ごとにファイルを書き換えず、コマンドラインオーバーライドで変更する

```bash
# 例: model と lr を変えて実行
python scripts/train.py model=effnet_b2 train.lr=5e-4
```

`conf/config.yaml` の基本構成:

```yaml
defaults:
  - model: effnet_b0
  - data: default
  - train: default
  - _self_

mlflow:
  experiment: baseline
  run_name: ???  # 実行時に必ず指定
```

---

## 実験管理規約（MLflow）

### 命名規則（厳守）

| 種別 | 形式 | 例 |
|------|------|----|
| Experiment 名 | `{phase}` | `baseline`, `augmentation`, `model_search`, `cpu_opt` |
| Run 名 | `YYYYMMDD_{model}_{変更点}` | `20260513_effnet_b0_baseline` |

phase の選択肢: `baseline` / `augmentation` / `model_search` / `soundscape` / `pseudo_label` / `cpu_opt` / `ensemble`

### 必須ログ項目

```python
# パラメータ（Hydra cfg を丸ごとログ）
mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

# メトリクス（エポックごと）
mlflow.log_metric("val_auc", val_auc, step=epoch)
mlflow.log_metric("val_loss", val_loss, step=epoch)
mlflow.log_metric("train_loss", train_loss, step=epoch)

# アーティファクト
mlflow.log_artifact("outputs/best_model.pth")
mlflow.log_artifact("outputs/submission.csv")
```

---

## 検証設計

```python
# StratifiedGroupKFold（録音ファイル名でグループ化）
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
groups = metadata['filename']  # または site_id
```

- **ランダム split 禁止**（ドメインシフト・データリーク防止）
- LB 提出後は必ず「ローカル val AUC」と「LB スコア」の相関をメモする

---

## Claudeとの作業ルール

1. **実装前に必ず設計を提案し、承認を得てから実装する**
2. 動作確認済みのベースラインコードは上書きしない（新しい config または module ファイルを作る）
3. 新しいモデルを追加するときは CPU 推論時間を計測してコメントに残す
4. コメントは日本語で書いてよい

---

## コード品質

```bash
# Lint（コンテナ外でも実行可）
uv run ruff check src/

# 自動修正
uv run ruff check --fix src/
```

- 型ヒント推奨（`def train(...) -> float:`）
- docstring は基本不要。WHY が自明でない箇所だけ1行コメント

---

## パッケージ管理（uv）

```bash
# パッケージ追加後、コンテナ再ビルドが必要
uv add <package>
git add uv.lock pyproject.toml
docker compose up --build -d
```

---

## キー制約（常に意識）

- 提出は **CPU ノートブック**（GPU 不可）
- 推論時間 **90分以内**（複数モデルのアンサンブルは時間計測必須）
- 出力は **sigmoid 確率**（softmax でなく BCEWithLogitsLoss を使う）
- 提出フォーマット: 行 = `{soundscape_id}_{seconds}`、列 = 234種
