from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


def get_mongo_uri(explicit_uri: str | None = None) -> str | None:
    uri = explicit_uri or os.getenv("MONGO_URI")
    if not uri:
        return None
    return uri


def get_client(mongo_uri: str) -> MongoClient:
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)


def get_db(client: MongoClient, db_name: str = "noshow_iq") -> Database:
    return client[db_name]


def get_collection(db: Database, name: str) -> Collection:
    return db[name]


def insert_prediction(
    db: Database,
    document: dict[str, Any],
    *,
    collection_name: str = "predictions",
) -> Any:
    return get_collection(db, collection_name).insert_one(document).inserted_id


def fetch_prediction_history(
    db: Database,
    *,
    limit: int = 20,
    collection_name: str = "predictions",
) -> list[dict[str, Any]]:
    cur = (
        get_collection(db, collection_name)
        .find({}, sort=[("timestamp", -1)])
        .limit(int(limit))
    )
    docs: list[dict[str, Any]] = list(cur)
    for d in docs:
        d["_id"] = str(d.get("_id"))
    return docs


def aggregate_stats(
    db: Database,
    *,
    predictions_collection: str = "predictions",
    training_runs_collection: str = "training_runs",
) -> dict[str, Any]:
    """
    Must be MongoDB aggregation only (no Python computation of stats).
    """
    pred_coll = get_collection(db, predictions_collection)
    train_coll = get_collection(db, training_runs_collection)

    pred_pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}
                },
                "low_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}
                },
                "average_probability": {"$avg": "$probability"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "total_predictions": 1,
                "high_risk_count": 1,
                "low_risk_count": 1,
                "average_probability": 1,
            }
        },
    ]

    train_pipeline = [
        {"$group": {"_id": None, "last_trained": {"$max": "$timestamp"}}},
        {"$project": {"_id": 0, "last_trained": 1}},
    ]

    pred_res = list(pred_coll.aggregate(pred_pipeline))
    train_res = list(train_coll.aggregate(train_pipeline))

    out: dict[str, Any] = {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": None,
        "last_trained": None,
    }
    if pred_res:
        out.update(pred_res[0])
    if train_res:
        out.update(train_res[0])
    return out


def insert_training_run(
    document: dict[str, Any],
    *,
    mongo_uri: str | None = None,
    db_name: str = "noshow_iq",
    collection_name: str = "training_runs",
) -> bool:
    uri = get_mongo_uri(mongo_uri)
    if not uri:
        return False

    client = get_client(uri)
    try:
        client.admin.command("ping")
        client[db_name][collection_name].insert_one(document)
        return True
    finally:
        client.close()
