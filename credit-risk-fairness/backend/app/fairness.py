from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


REFERENCE_GROUPS = {
    "race": "White",
    "gender": "Male",
    "ethnicity": "Not Hispanic or Latino",
}


@dataclass(frozen=True)
class BiasPolicy:
    max_disparate_impact_deviation: float = float(os.getenv("BIAS_MAX_DI_DEVIATION", "0.65"))
    min_group_count: int = int(os.getenv("MIN_FAIRNESS_GROUP_COUNT", "5"))


def audit_demographics(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("race", "gender"),
    y_true_col: str = "approved",
    y_pred_col: str = "predicted_approved",
    policy: BiasPolicy = BiasPolicy(),
) -> dict[str, Any]:
    slices = []
    for group_column in group_columns:
        slices.extend(
            audit_group(
                frame,
                group_column=group_column,
                y_true_col=y_true_col,
                y_pred_col=y_pred_col,
                min_group_count=policy.min_group_count,
            )
        )

    max_di_deviation = max(
        (abs(1.0 - item["disparate_impact_ratio"]) for item in slices if item["disparate_impact_ratio"] is not None),
        default=0.0,
    )
    return {
        "policy": {
            "max_disparate_impact_deviation": policy.max_disparate_impact_deviation,
            "min_group_count": policy.min_group_count,
            "interpretation": "Build fails when abs(1 - disparate_impact_ratio) exceeds the configured threshold.",
        },
        "summary": {
            "evaluated_slices": len(slices),
            "max_disparate_impact_deviation": round(max_di_deviation, 4),
            "passes_bias_regression": bool(max_di_deviation <= policy.max_disparate_impact_deviation),
        },
        "slices": slices,
    }


def audit_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    y_true_col: str,
    y_pred_col: str,
    min_group_count: int,
) -> list[dict[str, Any]]:
    eligible = frame.dropna(subset=[group_column, y_true_col, y_pred_col]).copy()
    eligible = eligible.groupby(group_column).filter(lambda part: len(part) >= min_group_count)
    if eligible.empty:
        return []

    reference = REFERENCE_GROUPS.get(group_column)
    if reference not in set(eligible[group_column]):
        reference = str(eligible[group_column].value_counts().idxmax())
    ref_part = eligible[eligible[group_column] == reference]
    ref_metrics = _rates(ref_part[y_true_col].to_numpy(), ref_part[y_pred_col].to_numpy())

    rows = []
    for group_value, part in eligible.groupby(group_column):
        metrics = _rates(part[y_true_col].to_numpy(), part[y_pred_col].to_numpy())
        approval_rate = metrics["selection_rate"]
        ref_rate = ref_metrics["selection_rate"]
        disparate_impact = _safe_ratio(approval_rate, ref_rate)
        rows.append(
            {
                "dimension": group_column,
                "group": str(group_value),
                "reference_group": str(reference),
                "n": int(len(part)),
                "approval_rate": _round_or_none(approval_rate),
                "observed_approval_rate": _round_or_none(float(np.mean(part[y_true_col]))),
                "statistical_parity_difference": _round_or_none(approval_rate - ref_rate),
                "disparate_impact_ratio": _round_or_none(disparate_impact),
                "equalized_odds": {
                    "tpr_difference": _round_or_none(metrics["tpr"] - ref_metrics["tpr"]),
                    "fpr_difference": _round_or_none(metrics["fpr"] - ref_metrics["fpr"]),
                    "max_difference": _round_or_none(
                        max(abs(metrics["tpr"] - ref_metrics["tpr"]), abs(metrics["fpr"] - ref_metrics["fpr"]))
                    ),
                },
                "four_fifths_rule": _status_from_di(disparate_impact),
            }
        )
    return sorted(rows, key=lambda row: (row["dimension"], row["group"]))


def _rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return {
        "selection_rate": float(np.mean(y_pred)) if len(y_pred) else float("nan"),
        "tpr": _safe_ratio(tp, tp + fn),
        "fpr": _safe_ratio(fp, fp + tn),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def _round_or_none(value: float) -> float | None:
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), 4)


def _status_from_di(value: float) -> str:
    if math.isnan(value):
        return "insufficient-reference"
    if 0.8 <= value <= 1.25:
        return "pass"
    return "review"

