from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import JSONResponse
from pymongo import MongoClient

from noshow_iq.db import aggregate_stats, fetch_prediction_history, get_client, get_db, insert_prediction
from noshow_iq.model import load_model
from noshow_iq.preprocess import preprocess_record


@dataclass(frozen=True)
class AppSettings:
    mongo_uri: str | None
    db_name: str = "noshow_iq"
    model_path: str = "artifacts/model.pkl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _risk_from_probability(p: float) -> str:
    if p >= 0.7:
        return "high"
    if p >= 0.4:
        return "medium"
    return "low"


def _recommendation_from_risk(risk: str) -> str:
    if risk == "high":
        return "Call patient and confirm the appointment; consider overbooking."
    if risk == "medium":
        return "Send a reminder SMS/call and confirm availability."
    return "Standard reminder is sufficient."


def create_app(
    *,
    settings: AppSettings | None = None,
    mongo_client: MongoClient | None = None,
) -> FastAPI:
    s = settings or AppSettings(mongo_uri=os.getenv("MONGO_URI"))

    app = FastAPI(title="NoShowIQ")

    client: MongoClient | None = mongo_client
    if client is None:
        if not s.mongo_uri:
            client = None
        else:
            client = get_client(s.mongo_uri)

    db = get_db(client, s.db_name) if client is not None else None

    @app.get("/")
    def root() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "noshow_iq_dashboard.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        mongo_ok = False
        if client is not None:
            try:
                client.admin.command("ping")
                mongo_ok = True
            except Exception:
                mongo_ok = False
        return {"status": "ok", "mongo": mongo_ok}

    @app.post("/predict")
    def predict(payload: dict[str, Any]) -> JSONResponse:
        try:
            artifact = load_model(s.model_path)
        except Exception:
            raise HTTPException(status_code=503, detail="Model not available. Train and save model.pkl first.")

        X = preprocess_record(payload)
        X = X.reindex(columns=artifact.feature_columns, fill_value=0)

        proba = None
        if hasattr(artifact.model, "predict_proba"):
            probs = artifact.model.predict_proba(X)
            proba = float(probs[0, 1])
        else:
            pred = int(artifact.model.predict(X)[0])
            proba = float(pred)

        risk = _risk_from_probability(proba)
        rec = _recommendation_from_risk(risk)

        doc = {
            "timestamp": _utc_now_iso(),
            "raw_input": payload,
            "cleaned_features": X.iloc[0].to_dict(),
            "risk_level": risk,
            "probability": proba,
            "recommendation": rec,
        }

        if db is None:
            raise HTTPException(status_code=500, detail="MongoDB not configured. Set MONGO_URI.")

        insert_prediction(db, doc)

        return JSONResponse(
            {
                "risk_level": risk,
                "probability": proba,
                "recommendation": rec,
            }
        )

    @app.get("/history")
    def history() -> list[dict[str, Any]]:
        if db is None:
            raise HTTPException(status_code=500, detail="MongoDB not configured. Set MONGO_URI.")
        return fetch_prediction_history(db, limit=20)

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        if db is None:
            raise HTTPException(status_code=500, detail="MongoDB not configured. Set MONGO_URI.")
        out = aggregate_stats(db)
        return out

    return app


app = create_app()

