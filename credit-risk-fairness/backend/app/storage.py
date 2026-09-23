from __future__ import annotations

from dataclasses import dataclass, field
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc
from typing import Any

from .config import settings

try:  # pragma: no cover - optional in unit tests.
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[misc, assignment]


@dataclass
class PredictionStore:
    """Prediction persistence with Postgres when configured and memory as a local fallback."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    engine: Engine | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if settings.database_url and create_engine is not None:
            self.engine = create_engine(settings.database_url, pool_pre_ping=True)
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        create table if not exists credit_predictions (
                          prediction_id text primary key,
                          created_at timestamptz not null,
                          decision text not null,
                          probability double precision not null,
                          payload jsonb not null,
                          response jsonb not null
                        )
                        """
                    )
                )

    def save(self, record: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        stored = {
            **record,
            "payload": payload,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.records[record["prediction_id"]] = stored
        if self.engine is not None:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        insert into credit_predictions (prediction_id, created_at, decision, probability, payload, response)
                        values (:prediction_id, :created_at, :decision, :probability, cast(:payload as jsonb), cast(:response as jsonb))
                        on conflict (prediction_id) do update set
                          created_at = excluded.created_at,
                          decision = excluded.decision,
                          probability = excluded.probability,
                          payload = excluded.payload,
                          response = excluded.response
                        """
                    ),
                    {
                        "prediction_id": record["prediction_id"],
                        "created_at": stored["created_at"],
                        "decision": record["decision"],
                        "probability": record["probability"],
                        "payload": _json(payload),
                        "response": _json(record),
                    },
                )
        return stored

    def get(self, prediction_id: str) -> dict[str, Any] | None:
        if prediction_id in self.records:
            return self.records[prediction_id]
        if self.engine is not None:
            with self.engine.begin() as connection:
                row = (
                    connection.execute(
                        text("select response, payload, created_at from credit_predictions where prediction_id = :prediction_id"),
                        {"prediction_id": prediction_id},
                    )
                    .mappings()
                    .first()
                )
                if row:
                    return {**row["response"], "payload": row["payload"], "created_at": row["created_at"].isoformat()}
        return self.records.get(prediction_id)


def _json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value)
