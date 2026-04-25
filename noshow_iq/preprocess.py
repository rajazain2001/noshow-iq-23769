from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _norm_col(name: str) -> str:
    """
    Normalize a column name so messy Kaggle headers map consistently.
    Example: "No-show" -> "noshow", "ScheduledDay" -> "scheduledday".
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


RAW_TO_CANONICAL = {
    "patientid": "patient_id",
    "appointmentid": "appointment_id",
    "gender": "gender",
    "scheduledday": "scheduled_day",
    "appointmentday": "appointment_day",
    "age": "age",
    "neighbourhood": "neighbourhood",
    "scholarship": "scholarship",
    "hipertension": "hypertension",
    "diabetes": "diabetes",
    "alcoholism": "alcoholism",
    "handcap": "handicap",
    "smsreceived": "sms_received",
    "noshow": "no_show",
}


CANONICAL_FEATURES = [
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


@dataclass(frozen=True)
class PreprocessResult:
    X: pd.DataFrame
    y: pd.Series | None
    feature_columns: list[str]


def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    Load Kaggle raw CSV. Keep everything as-is; cleaning happens later.
    """
    return pd.read_csv(csv_path)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map messy raw columns (incl. misspellings like Hipertension/Handcap/No-show)
    into stable canonical names used by the rest of the project.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        n = _norm_col(str(col))
        if n in RAW_TO_CANONICAL:
            rename_map[col] = RAW_TO_CANONICAL[n]
    out = df.rename(columns=rename_map).copy()
    return out


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _safe_int(series: pd.Series, *, fill: int = 0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(fill).astype(int)
    return s


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset with defensive rules:
    - Invalid ages -> NaN -> filled with median valid age
    - Parse dates for scheduled/appointment
    - Coerce binary/int fields
    """
    df = canonicalize_columns(df_raw)

    if "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        age = age.where((age >= 0) & (age <= 115), np.nan)
        median_age = float(np.nanmedian(age.to_numpy())) if np.isfinite(np.nanmedian(age.to_numpy())) else 0.0
        df["age"] = age.fillna(median_age).round().astype(int)

    if "scheduled_day" in df.columns:
        df["scheduled_day"] = _to_datetime(df["scheduled_day"])
    if "appointment_day" in df.columns:
        df["appointment_day"] = _to_datetime(df["appointment_day"])

    for col in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]:
        if col in df.columns:
            df[col] = _safe_int(df[col], fill=0)

    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.upper().str.strip()
        df["gender"] = df["gender"].where(df["gender"].isin(["M", "F"]), "U")

    return df


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering required by the exam:
    - days_in_advance: days between booking and appointment
    Additional feature:
    - appointment_weekday: 0=Mon ... 6=Sun
    """
    df = df_clean.copy()

    if "scheduled_day" in df.columns and "appointment_day" in df.columns:
        sched = df["scheduled_day"].dt.floor("D")
        appt = df["appointment_day"].dt.floor("D")
        days = (appt - sched).dt.days
        days = days.where(days.notna(), 0)
        df["days_in_advance"] = days.clip(lower=0).astype(int)
        df["appointment_weekday"] = appt.dt.weekday.fillna(0).astype(int)
    else:
        df["days_in_advance"] = 0
        df["appointment_weekday"] = 0

    return df


def extract_target(df: pd.DataFrame) -> pd.Series | None:
    """
    Target in Kaggle dataset is typically column `No-show` (canonical: `no_show`)
    with values Yes/No, meaning did not show up.
    Returns 1 for no-show, 0 for show.
    """
    if "no_show" not in df.columns:
        return None
    s = df["no_show"].astype(str).str.strip().str.lower()
    y = (s == "yes").astype(int)
    return y


def build_features(df: pd.DataFrame, *, feature_columns: Iterable[str] = CANONICAL_FEATURES) -> pd.DataFrame:
    out: dict[str, Any] = {}
    for col in feature_columns:
        if col not in df.columns:
            out[col] = 0
        else:
            out[col] = df[col]
    X = pd.DataFrame(out)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def preprocess_dataset(csv_path: str) -> PreprocessResult:
    """
    End-to-end preprocessing for training/evaluation.
    """
    df_raw = load_raw_csv(csv_path)
    df_clean = clean_dataframe(df_raw)
    df_feat = engineer_features(df_clean)
    y = extract_target(df_feat)
    X = build_features(df_feat)
    return PreprocessResult(X=X, y=y, feature_columns=list(X.columns))


def preprocess_record(record: dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess a single appointment JSON record into a 1-row feature frame.
    Accepts raw Kaggle-like keys (including 'No-show') and/or canonical keys.
    """
    df = pd.DataFrame([record])
    df_clean = clean_dataframe(df)
    df_feat = engineer_features(df_clean)
    X = build_features(df_feat)
    return X


def dataset_expected_path() -> str:
    return "data/raw/KaggleV2-May-2016.csv"

