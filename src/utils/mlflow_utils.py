import mlflow
import os
from contextlib import contextmanager

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)

@contextmanager
def start_run(experiment_name: str, run_name: str, tags: dict = {}):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run

def log_cv_results(scores: list[float], metric_name: str = "cv_score"):
    """Cross-validationの結果をまとめてログ"""
    for i, score in enumerate(scores):
        mlflow.log_metric(f"{metric_name}_fold{i}", score)
    mlflow.log_metric(f"{metric_name}_mean", sum(scores) / len(scores))