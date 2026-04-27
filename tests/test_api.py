from __future__ import annotations

import tempfile
from pathlib import Path

import mongomock
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from noshow_iq.api import AppSettings, create_app
from noshow_iq.model import ModelArtifact, save_model


def _write_dummy_model(path: str) -> None:
    # Minimal model that supports predict_proba and uses our 9 features.
    feature_columns = [
        "age",
        "scholarship",
        "hypertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "sms_received",
        "days_in_advance",
        "appointment_weekday",
    ]
    X = pd.DataFrame(
        [
            [10, 0, 0, 0, 0, 0, 0, 0, 1],
            [80, 1, 1, 1, 0, 0, 1, 20, 4],
            [30, 0, 0, 0, 0, 0, 1, 2, 2],
            [65, 1, 1, 0, 1, 0, 0, 10, 5],
        ],
        columns=feature_columns,
    )
    y = [0, 1, 0, 1]
    model = LogisticRegression(max_iter=500).fit(X, y)
    artifact = ModelArtifact(model=model, feature_columns=feature_columns, trained_at_utc="2026-01-01T00:00:00+00:00")
    save_model(artifact, path)


def test_health_ok_without_mongo() -> None:
    app = create_app(settings=AppSettings(mongo_uri=None))
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_requires_model() -> None:
    mongo = mongomock.MongoClient()
    app = create_app(settings=AppSettings(mongo_uri="mongodb://ignored", model_path="missing.pkl"), mongo_client=mongo)
    client = TestClient(app)
    r = client.post("/predict", json={"Age": 10})
    assert r.status_code == 503


def test_predict_inserts_prediction_document() -> None:
    mongo = mongomock.MongoClient()
    with tempfile.TemporaryDirectory() as td:
        model_path = str(Path(td) / "model.pkl")
        _write_dummy_model(model_path)

        app = create_app(settings=AppSettings(mongo_uri="mongodb://ignored", model_path=model_path), mongo_client=mongo)
        client = TestClient(app)

        payload = {"Age": 20, "ScheduledDay": "2016-04-29T18:38:08Z", "AppointmentDay": "2016-04-29T00:00:00Z"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "risk_level" in body
        assert "probability" in body
        assert "recommendation" in body

        docs = list(mongo["noshow_iq"]["predictions"].find({}))
        assert len(docs) == 1
        assert docs[0]["raw_input"]["Age"] == 20


def test_history_returns_last_20() -> None:
    mongo = mongomock.MongoClient()
    db = mongo["noshow_iq"]
    for i in range(25):
        db["predictions"].insert_one({"timestamp": f"2026-01-01T00:00:{i:02d}Z", "risk_level": "low", "probability": 0.1})

    app = create_app(settings=AppSettings(mongo_uri="mongodb://ignored"), mongo_client=mongo)
    client = TestClient(app)
    r = client.get("/history")
    assert r.status_code == 200
    assert len(r.json()) == 20


def test_stats_aggregation_shape() -> None:
    mongo = mongomock.MongoClient()
    db = mongo["noshow_iq"]
    db["predictions"].insert_one({"timestamp": "2026-01-01T00:00:00Z", "risk_level": "high", "probability": 0.9})
    db["predictions"].insert_one({"timestamp": "2026-01-02T00:00:00Z", "risk_level": "low", "probability": 0.2})
    db["training_runs"].insert_one({"timestamp": "2026-02-01T00:00:00Z"})

    app = create_app(settings=AppSettings(mongo_uri="mongodb://ignored"), mongo_client=mongo)
    client = TestClient(app)
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {
        "total_predictions",
        "high_risk_count",
        "low_risk_count",
        "average_probability",
        "last_trained",
    }
    assert data["total_predictions"] == 2
    assert data["high_risk_count"] == 1
    assert data["low_risk_count"] == 1
    assert data["last_trained"] == "2026-02-01T00:00:00Z"


def test_predict_requires_mongo_configured() -> None:
    with tempfile.TemporaryDirectory() as td:
        model_path = str(Path(td) / "model.pkl")
        _write_dummy_model(model_path)
        app = create_app(settings=AppSettings(mongo_uri=None, model_path=model_path))
        client = TestClient(app)
        r = client.post("/predict", json={"Age": 20})
        assert r.status_code == 500

