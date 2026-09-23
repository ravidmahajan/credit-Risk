from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import settings


PROTECTED_COLUMNS = ["race", "gender", "ethnicity"]
NUMERIC_FEATURES = [
    "loan_amount",
    "loan_to_value_ratio",
    "income",
    "debt_to_income_ratio",
    "property_value",
    "loan_term",
    "tract_minority_population_percent",
    "tract_to_msa_income_percentage",
    "ffiec_msa_md_median_family_income",
]
CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "lien_status",
    "occupancy_type",
    "preapproval",
    "construction_method",
    "derived_loan_product_type",
    "derived_dwelling_category",
    "applicant_credit_score_type",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_COLUMNS = FEATURE_COLUMNS + PROTECTED_COLUMNS + ["approved", "action_taken", "loan_id"]

NA_VALUES = {"", "NA", "N/A", "Exempt", "nan", "None", "9999"}


def load_hmda_frame(path: Path | str = settings.data_path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    if {"race", "gender", "approved", *FEATURE_COLUMNS}.issubset(raw.columns):
        frame = raw.copy()
    else:
        frame = normalize_hmda(raw)

    frame = frame.dropna(subset=["approved"]).reset_index(drop=True)
    for column in CATEGORICAL_FEATURES + PROTECTED_COLUMNS:
        frame[column] = frame[column].fillna("Not Available").astype(str)
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[MODEL_COLUMNS].copy()


def normalize_hmda(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame()
    frame["loan_id"] = raw.index.map(lambda idx: f"hmda-2023-il-warren-{idx + 1:04d}")
    frame["action_taken"] = _numeric(raw.get("action_taken"))
    frame = frame[frame["action_taken"].isin([1, 3])].copy()
    source = raw.loc[frame.index]

    frame["approved"] = (frame["action_taken"] == 1).astype(int)
    frame["loan_amount"] = _numeric(source.get("loan_amount"))
    frame["loan_to_value_ratio"] = _numeric(source.get("loan_to_value_ratio"))
    frame["income"] = _numeric(source.get("income"))
    frame["debt_to_income_ratio"] = source.get("debt_to_income_ratio", pd.Series(index=source.index)).map(_parse_dti)
    frame["property_value"] = _numeric(source.get("property_value"))
    frame["loan_term"] = _numeric(source.get("loan_term"))
    frame["tract_minority_population_percent"] = _numeric(source.get("tract_minority_population_percent"))
    frame["tract_to_msa_income_percentage"] = _numeric(source.get("tract_to_msa_income_percentage"))
    frame["ffiec_msa_md_median_family_income"] = _numeric(source.get("ffiec_msa_md_median_family_income"))

    frame["loan_type"] = _category(source.get("loan_type"))
    frame["loan_purpose"] = _category(source.get("loan_purpose"))
    frame["lien_status"] = _category(source.get("lien_status"))
    frame["occupancy_type"] = _category(source.get("occupancy_type"))
    frame["preapproval"] = _category(source.get("preapproval"))
    frame["construction_method"] = _category(source.get("construction_method"))
    frame["derived_loan_product_type"] = _category(source.get("derived_loan_product_type"))
    frame["derived_dwelling_category"] = _category(source.get("derived_dwelling_category"))
    frame["applicant_credit_score_type"] = _category(source.get("applicant_credit_score_type"))

    frame["race"] = _category(source.get("derived_race"))
    frame["gender"] = _category(source.get("derived_sex"))
    frame["ethnicity"] = _category(source.get("derived_ethnicity"))
    return frame


def build_sample_applicant(frame: pd.DataFrame) -> dict[str, Any]:
    if "approved" in frame and (frame["approved"] == 0).any():
        candidate = frame[frame["approved"] == 0].sort_values("loan_to_value_ratio", ascending=False).iloc[0]
        sample = {column: _json_value(candidate[column]) for column in FEATURE_COLUMNS}
        for column in PROTECTED_COLUMNS:
            sample[column] = _json_value(candidate[column])
        return sample

    sample: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        sample[column] = float(frame[column].median())
    for column in CATEGORICAL_FEATURES:
        mode = frame[column].mode(dropna=True)
        sample[column] = str(mode.iloc[0]) if not mode.empty else "Not Available"
    sample.update({"race": "Black or African American", "gender": "Female", "ethnicity": "Not Hispanic or Latino"})
    return sample


def _numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = series.astype(str).str.replace(",", "", regex=False).replace(list(NA_VALUES), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _category(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.replace(list(NA_VALUES), np.nan).fillna("Not Available").astype(str)


def _parse_dti(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    text = str(value).strip()
    if text in NA_VALUES:
        return float("nan")
    if text == "<20%":
        return 15.0
    if text == ">60%":
        return 65.0
    if text == "20%-<30%":
        return 25.0
    if text == "30%-<36%":
        return 33.0
    if text == "50%-60%":
        return 55.0
    if "-" in text:
        parts = text.replace("%", "").replace("<", "").split("-")
        nums = [float(part) for part in parts if part.strip().replace(".", "", 1).isdigit()]
        if nums:
            return float(sum(nums) / len(nums))
    try:
        return float(text.replace("%", ""))
    except ValueError:
        return float("nan")


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
