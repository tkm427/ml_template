# Kaggle Competition Project

DockerとMLflowを用いたKaggleコンペ用ローカル環境。

## 技術スタック

| ツール | 用途 |
|---|---|
| [gcr.io/kaggle-gpu-images/python](https://github.com/Kaggle/docker-python) | ベースイメージ（GPU・主要ライブラリ同梱） |
| [Docker Compose](https://docs.docker.com/compose/) | コンテナ管理 |
| [uv](https://github.com/astral-sh/uv) | Pythonパッケージ・バージョン管理 |
| [MLflow](https://mlflow.org/) | 実験管理 |
| [Tailscale](https://tailscale.com/) | リモートアクセス（VPN） |

## ディレクトリ構成

```
kaggle-project/
├── docker-compose.yml
├── Dockerfile
├── .env                    # Kaggle API Key（Git管理外）
├── .gitignore
├── pyproject.toml          # uv依存管理
├── uv.lock
├── README.md
│
├── data/
│   ├── raw/                # Kaggleからダウンロードした生データ（Git管理外）
│   ├── processed/          # 前処理済みデータ（Git管理外）
│   └── external/           # 外部データ
│
├── notebooks/              # 探索・分析用Jupyter Notebook
│
├── src/
│   ├── config.py           # 設定・定数
│   ├── features/           # 特徴量エンジニアリング
│   ├── models/             # 学習・推論
│   └── utils/
│       └── mlflow_utils.py # MLflowヘルパー
│
├── mlruns/                 # MLflow実験ログ（Git管理外）
├── mlartifacts/            # MLflowアーティファクト（Git管理外）
├── outputs/                # 予測結果・submission
└── scripts/
    ├── download_data.sh
    └── run_train.sh
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd kaggle-project
```

### 2. 環境変数の設定

`.env` を作成し、Kaggle APIキーを設定する。
APIキーは [Kaggle Account Settings](https://www.kaggle.com/settings/account) から取得できる。

```bash
cp .env.example .env
```

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
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
| MLflow UI | http://localhost:5000 |

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

## 実験管理（MLflow）

`src/utils/mlflow_utils.py` のヘルパーを使って実験をログする。

```python
from src.utils.mlflow_utils import start_run, log_cv_results

with start_run(experiment_name="my-competition", run_name="lgbm-baseline"):
    mlflow.log_params({"n_estimators": 1000, "learning_rate": 0.05})

    scores = cross_validate(model, X, y)
    log_cv_results(scores)

    mlflow.log_artifact("outputs/submission.csv")
```

実験結果はMLflow UIで確認できる → http://localhost:5000

## リモートアクセス（Tailscale）

WSLなどリモートマシン上で環境を動かす場合、TailscaleのIPでアクセスする。

```bash
# WSL側でTailscale IPを確認
tailscale ip -4
```

| サービス | URL |
|---|---|
| Jupyter Lab | http://\<Tailscale-IP\>:8888 |
| MLflow UI | http://\<Tailscale-IP\>:5000 |

## コンテナの操作

```bash
# 起動
docker compose up -d

# 停止
docker compose down

# ログ確認
docker compose logs -f

# コンテナに入る
docker compose exec workspace bash
```

## .gitignore の対象

以下はGit管理外とする。

- `data/raw/`, `data/processed/` : データファイル
- `mlruns/`, `mlartifacts/` : 実験ログ
- `outputs/` : 予測結果
- `.env` : APIキー
