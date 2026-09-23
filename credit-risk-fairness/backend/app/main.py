from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .config import settings
from .modeling import CreditRiskModel
from .schemas import ApplicantPayload, PredictionResponse
from .storage import PredictionStore


REQUEST_LATENCY = Histogram("credit_api_request_latency_seconds", "API request latency", ["method", "path"])
PREDICTIONS = Counter("credit_api_predictions_total", "Prediction count by decision", ["decision"])

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Real-time credit risk predictions with structured SHAP/LIME explanations and demographic fairness audits.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CreditRiskModel()
store = PredictionStore()


@app.on_event("startup")
def startup() -> None:
    model.load_or_train()


@app.middleware("http")
async def instrument_requests(request: Any, call_next: Any) -> Any:
    start = time.perf_counter()
    response = await call_next(request)
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(time.perf_counter() - start)
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_backend": model.runtime.backend, "model_version": settings.model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ApplicantPayload) -> dict[str, Any]:
    record = model.predict(payload.model_dump(exclude_none=True))
    store.save(record, payload.model_dump(exclude_none=True))
    PREDICTIONS.labels(record["decision"]).inc()
    return record


@app.get("/audit/fairness")
def fairness_audit() -> dict[str, Any]:
    return model.audit()


@app.get("/explain/{prediction_id}")
def explain(prediction_id: str) -> dict[str, Any]:
    record = store.get(prediction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction id not found in the in-memory store.")
    return {
        "prediction_id": prediction_id,
        "explanation": {
            "shap_vector": record["shap_vector"],
            "force_plot": record["force_plot"],
            "top_factors": record["top_factors"],
        },
        "counterfactual": model.counterfactual(record, record.get("payload")),
    }


@app.get("/demo/state")
def demo_state() -> dict[str, Any]:
    return model.demo_state()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

