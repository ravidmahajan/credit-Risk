from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApplicantPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    loan_amount: float = Field(..., ge=0)
    loan_to_value_ratio: float | None = Field(None, ge=0)
    income: float | None = None
    debt_to_income_ratio: float | None = None
    property_value: float | None = None
    loan_term: float | None = None
    loan_type: str | None = None
    loan_purpose: str | None = None
    lien_status: str | None = None
    occupancy_type: str | None = None
    race: str | None = None
    gender: str | None = None
    ethnicity: str | None = None


class PredictionResponse(BaseModel):
    prediction_id: str
    decision: str
    approved: bool
    probability: float
    threshold: float
    latency_ms: float
    model_version: str
    model_backend: str
    protected_attributes: dict[str, str]
    explanation_method: str
    shap_vector: list[dict[str, Any]]
    top_factors: list[dict[str, Any]]
    force_plot: dict[str, Any]

