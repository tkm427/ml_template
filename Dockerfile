FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock* ./

# --no-install-project: プロジェクト自体をビルドせず依存だけ入れる
# UV_SYSTEM_PYTHON=1 でシステム Python へインストール
RUN uv pip install --system -r pyproject.toml || \
    pip install hydra-core omegaconf wandb scikit-learn

COPY . .