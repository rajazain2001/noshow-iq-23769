from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from noshow_iq.db import insert_training_run
from noshow_iq.preprocess import PreprocessResult, preprocess_dataset


@dataclass(frozen=True)
class ModelArtifact:
    model: Any
    feature_columns: list[str]
    trained_at_utc: str


@dataclass(frozen=True)
class EvaluationMetrics:
    per_class: dict[int, dict[str, float]]
    macro_avg: dict[str, float]
    weighted_avg: dict[str, float]
    confusion_matrix: list[list[int]]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_model(artifact: ModelArtifact, path: str = "model.pkl") -> str:
    out_path = str(Path(path))
    joblib.dump(
        {"model": artifact.model, "feature_columns": artifact.feature_columns, "trained_at_utc": artifact.trained_at_utc},
        out_path,
    )
    return out_path


def load_model(path: str = "model.pkl") -> ModelArtifact:
    payload = joblib.load(str(Path(path)))
    return ModelArtifact(
        model=payload["model"],
        feature_columns=list(payload["feature_columns"]),
        trained_at_utc=str(payload.get("trained_at_utc") or _utc_now_iso()),
    )


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series) -> EvaluationMetrics:
    y_true = np.asarray(y).astype(int)
    y_pred = np.asarray(model.predict(X)).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    per_class: dict[int, dict[str, float]] = {}
    for idx, label in enumerate([0, 1]):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": float(support[idx]),
        }

    macro_avg = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }

    weights = support / support.sum() if support.sum() else np.array([0.5, 0.5])
    weighted_avg = {
        "precision": float(np.sum(precision * weights)),
        "recall": float(np.sum(recall * weights)),
        "f1": float(np.sum(f1 * weights)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return EvaluationMetrics(
        per_class=per_class,
        macro_avg=macro_avg,
        weighted_avg=weighted_avg,
        confusion_matrix=cm.tolist(),
    )


def train(
    csv_path: str,
    *,
    model_path: str = "model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
    imbalance_technique: str = "class_weight=balanced",
    log_training_to_mongo: bool = True,
) -> dict[str, Any]:
    prep: PreprocessResult = preprocess_dataset(csv_path)
    if prep.y is None:
        raise ValueError("Target column not found after preprocessing (expected `No-show` / `no_show`).")

    X_train, X_test, y_train, y_test = train_test_split(
        prep.X,
        prep.y,
        test_size=test_size,
        random_state=random_state,
        stratify=prep.y,
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    artifact = ModelArtifact(model=model, feature_columns=prep.feature_columns, trained_at_utc=_utc_now_iso())
    saved_to = save_model(artifact, model_path)

    mongo_logged = False
    if log_training_to_mongo:
        metrics_doc = asdict(metrics)
        metrics_doc["per_class"] = {str(k): v for k, v in metrics_doc.get("per_class", {}).items()}
        doc = {
            "timestamp": artifact.trained_at_utc,
            "training_size": int(len(X_train)),
            "imbalance_technique": imbalance_technique,
            "metrics": metrics_doc,
        }
        try:
            mongo_logged = insert_training_run(doc)
        except Exception:
            mongo_logged = False

    return {
        "model_path": saved_to,
        "trained_at_utc": artifact.trained_at_utc,
        "metrics": asdict(metrics),
        "mongo_logged": mongo_logged,
    }


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X)).astype(int)


