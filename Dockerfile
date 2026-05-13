FROM gcr.io/kaggle-gpu-images/python:latest

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock* ./

# --no-install-project: プロジェクト自体をビルドせず依存だけ入れる
RUN uv sync --frozen --no-install-project
COPY . .