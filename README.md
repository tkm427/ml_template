# Kaggle Competition Project

DockerとW&Bを用いたKaggleコンペ用ローカル環境。

## 技術スタック

| ツール | 用途 |
|---|---|
| [gcr.io/kaggle-gpu-images/python](https://github.com/Kaggle/docker-python) | ベースイメージ（GPU・主要ライブラリ同梱） |
| [Docker Compose](https://docs.docker.com/compose/) | コンテナ管理 |
| [uv](https://github.com/astral-sh/uv) | Pythonパッケージ・バージョン管理 |
| [Weights & Biases](https://wandb.ai/) | 実験管理 |
| [Tailscale](https://tailscale.com/) | リモートアクセス（VPN） |

## ディレクトリ構成

```
kaggle_project/
├── conf/
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── train/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils/wandb_utils.py  # W&Bヘルパー
├── scripts/
│   ├── train.py               # 学習エントリポイント（Hydra）
│   ├── make_submission.py     # 推論 + submission.csv 生成
│   └── analyze_*.py           # 分析スクリプト（再利用可能）
├── notebooks/                 # EDA・可視化のみ（本番コード禁止）
├── wandb/                  # W&B実験ログ（Git管理外）
├── data/
│   ├── raw/                   # Kaggle 生データ（Git 管理外）
│   └── processed/             # 前処理済みデータ（Git 管理外）
├── docs/
│   └── {competition}/
│       ├── competition_overview.md   # コンペ仕様（静的）
│       ├── strategy.md               # 戦略・時間配分・週次ふりかえり
│       ├── experiments.md            # 実験インデックス・現在のフォーカス
│       ├── experiments/              # 1 実験 1 ファイル
│       │   └── exp001_baseline.md
│       ├── eda/                      # データ理解の発見
│       │   └── class_distribution.md
│       ├── research/                 # 競合・先行研究の調査
│       │   ├── past_solutions.md
│       │   ├── discussions.md
│       │   ├── public_notebooks.md
│       │   └── papers.md
│       └── postmortems/              # 重要な失敗の深掘り
│           └── pm001_aug_failed.md
└── outputs/                   # チェックポイント・submission（Git 管理外）
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd kaggle-project
```

### 2. 環境変数の設定

`.env` を作成し、Kaggle APIキーと W&B APIキーを設定する。
Kaggle APIキーは [Kaggle Account Settings](https://www.kaggle.com/settings/account) から、
W&B APIキーは [W&B Authorize](https://wandb.ai/authorize) から取得できる。

```bash
cp .env.example .env
```

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
WANDB_API_KEY=your_wandb_api_key
```

### 3. lockfileの生成

```bash
# uvがない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

uv lock
```

### 4. Dockerコンテナの起動

```bash
docker compose up --build -d
```

### 5. 動作確認

| サービス | URL |
|---|---|
| Jupyter Lab | http://localhost:8888 |
| W&B ダッシュボード | https://wandb.ai/\<entity\>/\<project\> |

## データのダウンロード

```bash
docker compose exec workspace bash

# コンテナ内で実行
kaggle competitions download -c <competition-name> -p data/raw/
unzip data/raw/<competition-name>.zip -d data/raw/
```

## パッケージ管理（uv）

```bash
# パッケージの追加
uv add <package-name>

# 開発用パッケージの追加
uv add --group dev <package-name>

# lockfileの更新後はコミットする
git add uv.lock pyproject.toml
```

コンテナを再ビルドすると新しいパッケージが反映される。

```bash
docker compose up --build -d
```

## 学習の中断・再開

学習には時間がかかるため、任意のタイミングで中断し、あとから続きを再開できる。

### 中断する

学習中に `Ctrl+C` を押すと、その直前のエポックのチェックポイントが保存されて終了する。

```
Epoch 010/030 | train_loss: 0.1234 | val_loss: 0.1100 | val_auc: 0.9500
^C
中断しました。outputs/resume_20260518_effnet_b0_specaugment_fold0.ckpt から再開できます。
```

チェックポイントには以下が保存される:
- モデルの重み
- Optimizer / Scheduler の状態（学習率スケジュールが正確に再現される）
- 完了済みエポック番号
- ベスト val AUC
- W&B の Run ID（同じ Run にメトリクスが続けて記録される）

### 再開する

**同じ `wandb.run_name` で再実行するだけ**。チェックポイントが自動検出されて続きから始まる。

```bash
docker compose exec workspace python scripts/train.py \
  wandb.run_name=20260518_effnet_b0_specaugment fold=0
# → Resume: epoch 10 から再開 (best_auc=0.9500) と表示される
```

### 最初からやり直す

チェックポイントファイルを削除してから実行する。

```bash
rm outputs/resume_20260518_effnet_b0_specaugment_fold0.ckpt
docker compose exec workspace python scripts/train.py \
  wandb.run_name=20260518_effnet_b0_specaugment fold=0
```

> チェックポイントのファイル名形式: `outputs/resume_{run_name}_fold{N}.ckpt`

---

## 実験管理（W&B）

`src/utils/wandb_utils.py` のヘルパーを使って実験をログする。

```python
from src.utils.wandb_utils import start_run, log_cv_results

with start_run(project="my-competition", run_name="lgbm-baseline"):
    wandb.config.update({"n_estimators": 1000, "learning_rate": 0.05})

    scores = cross_validate(model, X, y)
    log_cv_results(scores)

    artifact = wandb.Artifact("submission", type="submission")
    artifact.add_file("outputs/submission.csv")
    wandb.log_artifact(artifact)
```

実験結果はW&Bのダッシュボードで確認できる → https://wandb.ai/\<entity\>/\<project\>

## .gitignore の対象

以下はGit管理外とする。

- `data/raw/`, `data/processed/` : データファイル
- `wandb/` : 実験ログ
- `outputs/` : 予測結果
- `.env` : APIキー
