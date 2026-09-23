from __future__ import annotations

from app.modeling import CreditRiskModel


def test_bias_regression_disparate_impact_gate() -> None:
    model = CreditRiskModel()
    model.load_or_train()
    report = model.audit()["fairness"]

    assert report["summary"]["passes_bias_regression"], (
        "Disparate impact regression gate failed: "
        f"max deviation={report['summary']['max_disparate_impact_deviation']} "
        f"limit={report['policy']['max_disparate_impact_deviation']}"
    )

