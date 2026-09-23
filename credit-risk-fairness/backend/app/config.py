from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "hmda_il_warren_2023_originated_denied.csv"


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Explainable Credit Risk & Fairness Auditing API")
    model_version: str = os.getenv("MODEL_VERSION", "hmda-xgb-v0.1.0")
    data_path: Path = Path(os.getenv("HMDA_DATA_PATH", str(DEFAULT_DATA_PATH)))
    database_url: str = os.getenv("DATABASE_URL", "")
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5174,http://localhost:5174").split(",")
        if origin.strip()
    )
    approval_threshold_floor: float = float(os.getenv("APPROVAL_THRESHOLD_FLOOR", "0.35"))
    approval_threshold_ceiling: float = float(os.getenv("APPROVAL_THRESHOLD_CEILING", "0.72"))
    max_di_deviation: float = float(os.getenv("BIAS_MAX_DI_DEVIATION", "0.65"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    min_group_count: int = int(os.getenv("MIN_FAIRNESS_GROUP_COUNT", "5"))


settings = Settings()

