from __future__ import annotations

import os
from typing import Any, Optional

from pymongo import MongoClient


def get_mongo_uri(explicit_uri: str | None = None) -> str | None:
    uri = explicit_uri or os.getenv("MONGO_URI")
    if not uri:
        return None
    return uri


def get_client(mongo_uri: str) -> MongoClient:
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)


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

