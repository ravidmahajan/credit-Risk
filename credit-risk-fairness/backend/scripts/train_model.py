from __future__ import annotations

import json
from pathlib import Path

from app.modeling import CreditRiskModel


def main() -> None:
    model = CreditRiskModel()
    model.load_or_train()
    out = Path("backend/artifacts")
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_summary.json").write_text(json.dumps(model.audit(), indent=2), encoding="utf-8")
    print(json.dumps(model.runtime.training_summary, indent=2))


if __name__ == "__main__":
    main()

