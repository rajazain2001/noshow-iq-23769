from __future__ import annotations

import argparse
from typing import Any

import httpx


def _predict_payload() -> dict[str, Any]:
    return {
        "Age": 20,
        "ScheduledDay": "2016-04-29T18:38:08Z",
        "AppointmentDay": "2016-04-29T00:00:00Z",
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
        "Gender": "F",
    }


def smoke_test(base_url: str) -> bool:
    url = base_url.rstrip("/")
    endpoints = {
        "health": f"{url}/health",
        "predict": f"{url}/predict",
        "stats": f"{url}/stats",
    }

    payload = _predict_payload()

    with httpx.Client(timeout=30.0) as client:
        health = client.get(endpoints["health"])
        if health.status_code != 200:
            print("FAIL: /health", health.status_code, health.text)
            return False
        if health.json().get("status") != "ok":
            print("FAIL: /health wrong status", health.json())
            return False

        pred = client.post(endpoints["predict"], json=payload)
        if pred.status_code != 200:
            print("FAIL: /predict", pred.status_code, pred.text)
            return False
        p_json = pred.json()
        if not all(k in p_json for k in ("risk_level", "probability", "recommendation")):
            print("FAIL: /predict wrong shape", p_json)
            return False

        stats = client.get(endpoints["stats"])
        if stats.status_code != 200:
            print("FAIL: /stats", stats.status_code, stats.text)
            return False
        s_json = stats.json()
        required = {
            "total_predictions",
            "high_risk_count",
            "low_risk_count",
            "average_probability",
            "last_trained",
        }
        if not required.issubset(set(s_json.keys())):
            print("FAIL: /stats wrong shape", s_json)
            return False

    print("PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", help="Base URL like https://<space>.hf.space")
    args = parser.parse_args()

    ok = smoke_test(args.base_url)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

