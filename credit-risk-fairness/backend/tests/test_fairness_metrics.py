from __future__ import annotations

import pandas as pd

from app.fairness import BiasPolicy, audit_demographics


def test_fairness_metrics_include_core_measures() -> None:
    frame = pd.DataFrame(
        {
            "race": ["White", "White", "Black or African American", "Black or African American"],
            "gender": ["Male", "Female", "Male", "Female"],
            "approved": [1, 0, 1, 0],
            "predicted_approved": [1, 1, 0, 0],
        }
    )

    report = audit_demographics(frame, policy=BiasPolicy(max_disparate_impact_deviation=1.0, min_group_count=1))

    assert report["summary"]["evaluated_slices"] == 4
    race_slice = next(item for item in report["slices"] if item["dimension"] == "race" and item["group"] == "Black or African American")
    assert "statistical_parity_difference" in race_slice
    assert "disparate_impact_ratio" in race_slice
    assert "equalized_odds" in race_slice

