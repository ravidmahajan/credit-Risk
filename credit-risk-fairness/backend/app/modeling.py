from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import settings
from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_sample_applicant, load_hmda_frame
from .fairness import BiasPolicy, audit_demographics

try:  # pragma: no cover - exercised when the optional production dependency is installed.
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

try:  # pragma: no cover
    import shap
except Exception:  # pragma: no cover
    shap = None  # type: ignore[assignment]


@dataclass
class ModelRuntime:
    pipeline: Pipeline | None = None
    frame: pd.DataFrame | None = None
    audit_report: dict[str, Any] = field(default_factory=dict)
    sample_applicant: dict[str, Any] = field(default_factory=dict)
    threshold: float = 0.5
    backend: str = "untrained"
    feature_names: list[str] = field(default_factory=list)
    baseline_row: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    shap_explainer: Any = None


class CreditRiskModel:
    def __init__(self) -> None:
        self.runtime = ModelRuntime()

    def load_or_train(self) -> None:
        frame = load_hmda_frame(settings.data_path)
        self.fit(frame)

    def fit(self, frame: pd.DataFrame) -> None:
        X = frame[FEATURE_COLUMNS]
        y = frame["approved"].astype(int)
        stratify_key = _build_stratify_key(frame)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=settings.random_state,
            stratify=stratify_key,
        )

        pipeline, backend = _build_pipeline()
        pipeline.fit(X_train, y_train)
        valid_probability = pipeline.predict_proba(X_valid)[:, 1]
        threshold = _calibrate_threshold(y_valid.to_numpy(), valid_probability)

        scored = frame.copy()
        probabilities = pipeline.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
        scored["predicted_probability"] = probabilities
        scored["predicted_approved"] = (probabilities >= threshold).astype(int)

        feature_names = _feature_names(pipeline)
        cv_summary = _cross_validation_summary(pipeline, X, y)
        auc = roc_auc_score(y_valid, valid_probability) if len(np.unique(y_valid)) > 1 else float("nan")
        sample = build_sample_applicant(frame)
        baseline = _baseline_row(frame)
        audit = audit_demographics(
            scored,
            policy=BiasPolicy(
                max_disparate_impact_deviation=settings.max_di_deviation,
                min_group_count=settings.min_group_count,
            ),
        )

        self.runtime = ModelRuntime(
            pipeline=pipeline,
            frame=scored,
            audit_report=audit,
            sample_applicant=sample,
            threshold=threshold,
            backend=backend,
            feature_names=feature_names,
            baseline_row=baseline,
            training_summary={
                "rows": int(len(frame)),
                "positive_rate": round(float(y.mean()), 4),
                "validation_auc": round(float(auc), 4) if not np.isnan(auc) else None,
                "threshold": round(float(threshold), 4),
                "cross_validation": cv_summary,
                "feature_policy": "Protected columns are preserved for auditing and excluded from model features.",
            },
            shap_explainer=_make_shap_explainer(pipeline, backend),
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        row = self._row_from_payload(payload)
        probability = float(self._pipeline.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
        decision = "approve" if probability >= self.runtime.threshold else "deny"
        explanation = self.explain_row(row, probability)
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "prediction_id": f"pred_{uuid.uuid4().hex[:16]}",
            "decision": decision,
            "approved": decision == "approve",
            "probability": round(probability, 6),
            "threshold": round(self.runtime.threshold, 6),
            "latency_ms": round(latency_ms, 3),
            "model_version": settings.model_version,
            "model_backend": self.runtime.backend,
            "protected_attributes": {
                "race": str(payload.get("race", "Not Provided")),
                "gender": str(payload.get("gender", "Not Provided")),
                "ethnicity": str(payload.get("ethnicity", "Not Provided")),
            },
            **explanation,
        }

    def explain_row(self, row: pd.DataFrame, probability: float | None = None) -> dict[str, Any]:
        probability = float(self._pipeline.predict_proba(row[FEATURE_COLUMNS])[:, 1][0]) if probability is None else probability
        contributions, method = self._local_contributions(row)
        sorted_contributions = sorted(contributions, key=lambda item: abs(item["value"]), reverse=True)
        base_probability = float(self._pipeline.predict_proba(pd.DataFrame([self.runtime.baseline_row])[FEATURE_COLUMNS])[:, 1][0])
        return {
            "explanation_method": method,
            "shap_vector": sorted_contributions,
            "top_factors": sorted_contributions[:6],
            "force_plot": {
                "base_value": round(base_probability, 6),
                "output_value": round(float(probability), 6),
                "link": "identity_probability",
                "features": sorted_contributions,
            },
        }

    def counterfactual(self, prediction: dict[str, Any] | None, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        original_payload = payload or self.runtime.sample_applicant
        row = self._row_from_payload(original_payload)
        current_probability = float(self._pipeline.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
        target_approval = current_probability < self.runtime.threshold
        candidate = row.iloc[0].to_dict()
        steps = []

        edits = [
            ("loan_to_value_ratio", -5.0, 60.0, "Lower LTV"),
            ("debt_to_income_ratio", -4.0, 24.0, "Lower DTI"),
            ("income", 8.0, None, "Increase income"),
            ("loan_amount", -7.5, 25000.0, "Lower requested amount"),
        ]
        if not target_approval:
            edits = [
                ("loan_to_value_ratio", 5.0, 115.0, "Raise LTV"),
                ("debt_to_income_ratio", 4.0, 70.0, "Raise DTI"),
                ("income", -8.0, 5.0, "Reduce income"),
                ("loan_amount", 7.5, None, "Raise requested amount"),
            ]

        best_probability = current_probability
        for _ in range(6):
            search_rows = []
            metadata = []
            for feature, delta, bound, label in edits:
                trial = dict(candidate)
                for _step in range(1, 9):
                    value = float(trial.get(feature) or self.runtime.baseline_row.get(feature) or 0)
                    if feature in {"income", "loan_amount"}:
                        next_value = value * (1 + delta / 100)
                    else:
                        next_value = value + delta
                    if bound is not None:
                        next_value = max(next_value, bound) if delta < 0 else min(next_value, bound)
                    if abs(next_value - value) < 1e-9:
                        continue
                    trial[feature] = round(next_value, 3)
                    search_rows.append(dict(trial))
                    metadata.append((feature, trial[feature], label))
            if not search_rows:
                break

            probabilities = self._pipeline.predict_proba(pd.DataFrame(search_rows)[FEATURE_COLUMNS])[:, 1]
            if target_approval:
                best_index = int(np.argmax(probabilities))
                improves = float(probabilities[best_index]) > best_probability + 0.0001
            else:
                best_index = int(np.argmin(probabilities))
                improves = float(probabilities[best_index]) < best_probability - 0.0001
            if not improves:
                break

            best_probability = float(probabilities[best_index])
            candidate = search_rows[best_index]
            feature, new_value, label = metadata[best_index]
            if steps and steps[-1]["feature"] == feature:
                steps[-1] = {"feature": feature, "new_value": new_value, "probability": round(best_probability, 6), "rationale": label}
            else:
                steps.append({"feature": feature, "new_value": new_value, "probability": round(best_probability, 6), "rationale": label})

            if (target_approval and best_probability >= self.runtime.threshold) or (
                not target_approval and best_probability < self.runtime.threshold
            ):
                return {
                    "target_decision": "approve" if target_approval else "deny",
                    "original_probability": round(current_probability, 6),
                    "counterfactual_probability": round(best_probability, 6),
                    "changes": steps,
                    "status": "target-reached",
                    "interpretation": "These what-if edits crossed the model decision threshold.",
                }

        final_probability = best_probability
        return {
            "target_decision": "approve" if target_approval else "deny",
            "original_probability": round(current_probability, 6),
            "counterfactual_probability": round(final_probability, 6),
            "changes": steps,
            "status": "target-not-reached",
            "interpretation": "The search found the closest improvement it could, but it did not cross the decision threshold.",
        }

    def audit(self) -> dict[str, Any]:
        return {
            "dataset": {
                "source": "Official FFIEC HMDA Data Browser CSV",
                "scope": "2023 Warren County, IL; action_taken in {1 originated, 3 denied}",
                "rows": self.runtime.training_summary.get("rows", 0),
            },
            "model": {
                "version": settings.model_version,
                "backend": self.runtime.backend,
                "threshold": self.runtime.threshold,
                "training": self.runtime.training_summary,
            },
            "fairness": self.runtime.audit_report,
        }

    def demo_state(self) -> dict[str, Any]:
        prediction = self.predict(self.runtime.sample_applicant)
        return {
            "audit": self.audit(),
            "sampleApplicant": self.runtime.sample_applicant,
            "samplePrediction": prediction,
            "counterfactual": self.counterfactual(prediction, self.runtime.sample_applicant),
        }

    @property
    def _pipeline(self) -> Pipeline:
        if self.runtime.pipeline is None:
            raise RuntimeError("Model is not trained.")
        return self.runtime.pipeline

    def _row_from_payload(self, payload: dict[str, Any]) -> pd.DataFrame:
        row = dict(self.runtime.baseline_row or self.runtime.sample_applicant)
        row.update(payload)
        for column in NUMERIC_FEATURES:
            row[column] = _as_float(row.get(column), self.runtime.baseline_row.get(column, 0.0))
        for column in CATEGORICAL_FEATURES:
            row[column] = str(row.get(column) or self.runtime.baseline_row.get(column) or "Not Available")
        return pd.DataFrame([row])

    def _local_contributions(self, row: pd.DataFrame) -> tuple[list[dict[str, Any]], str]:
        if shap is not None and self.runtime.backend == "xgboost" and self.runtime.shap_explainer is not None:
            try:  # pragma: no cover - optional dependency path.
                transformed = self._pipeline.named_steps["features"].transform(row[FEATURE_COLUMNS])
                values = self.runtime.shap_explainer.shap_values(transformed)
                vector = values[0] if np.asarray(values).ndim == 2 else values[1][0]
                return _collapse_encoded_contributions(vector, self.runtime.feature_names), "shap_tree"
            except Exception:
                pass

        rows = [dict(self.runtime.baseline_row)]
        for feature in FEATURE_COLUMNS:
            perturbed = dict(self.runtime.baseline_row)
            perturbed[feature] = row.iloc[0][feature]
            rows.append(perturbed)
        batch_probabilities = self._pipeline.predict_proba(pd.DataFrame(rows)[FEATURE_COLUMNS])[:, 1]
        baseline_probability = float(batch_probabilities[0])
        actual_probability = float(self._pipeline.predict_proba(row[FEATURE_COLUMNS])[:, 1][0])
        contributions = []
        for feature in FEATURE_COLUMNS:
            feature_probability = float(batch_probabilities[len(contributions) + 1])
            contributions.append(
                {
                    "feature": feature,
                    "value": round(feature_probability - baseline_probability, 6),
                    "feature_value": _jsonable(row.iloc[0][feature]),
                }
            )
        residual = actual_probability - baseline_probability - sum(item["value"] for item in contributions)
        contributions.append({"feature": "interaction_residual", "value": round(residual, 6), "feature_value": "model interactions"})
        return contributions, "feature_perturbation_lime_fallback"


def _build_pipeline() -> tuple[Pipeline, str]:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    classifier: Any
    backend: str
    if XGBClassifier is not None:
        classifier = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=2,
            random_state=settings.random_state,
            tree_method="hist",
        )
        backend = "xgboost"
    else:
        classifier = GradientBoostingClassifier(n_estimators=140, learning_rate=0.05, max_depth=3, random_state=settings.random_state)
        backend = "sklearn-gradient-boosting"

    return (
        Pipeline(
            steps=[
                (
                    "features",
                    ColumnTransformer(
                        transformers=[
                            ("num", numeric, NUMERIC_FEATURES),
                            ("cat", categorical, CATEGORICAL_FEATURES),
                        ]
                    ),
                ),
                ("classifier", classifier),
            ]
        ),
        backend,
    )


def _build_stratify_key(frame: pd.DataFrame) -> pd.Series:
    combo = frame["approved"].astype(str) + "_" + frame["race"].astype(str)
    if combo.value_counts().min() >= 2:
        return combo
    return frame["approved"].astype(int)


def _calibrate_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(settings.approval_threshold_floor, settings.approval_threshold_ceiling, 76):
        y_pred = (probabilities >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue
        score = f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _cross_validation_summary(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    min_class = int(y.value_counts().min())
    splits = min(5, min_class)
    if splits < 2:
        return {"folds": 0, "roc_auc_mean": None, "roc_auc_std": None}
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=settings.random_state)
    scores = cross_val_score(clone(pipeline), X, y, scoring="roc_auc", cv=cv)
    return {"folds": splits, "roc_auc_mean": round(float(np.mean(scores)), 4), "roc_auc_std": round(float(np.std(scores)), 4)}


def _baseline_row(frame: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        row[column] = float(frame[column].median())
    for column in CATEGORICAL_FEATURES:
        row[column] = str(frame[column].mode(dropna=True).iloc[0])
    return row


def _feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return [name.replace("num__", "").replace("cat__", "") for name in pipeline.named_steps["features"].get_feature_names_out()]
    except Exception:
        return FEATURE_COLUMNS


def _make_shap_explainer(pipeline: Pipeline, backend: str) -> Any:
    if shap is None or backend != "xgboost":
        return None
    try:  # pragma: no cover - optional dependency path.
        return shap.TreeExplainer(pipeline.named_steps["classifier"])
    except Exception:
        return None


def _collapse_encoded_contributions(values: np.ndarray, feature_names: list[str]) -> list[dict[str, Any]]:
    totals: dict[str, float] = {}
    for name, value in zip(feature_names, values, strict=False):
        base = name.split("_", 1)[0] if name not in FEATURE_COLUMNS else name
        totals[base] = totals.get(base, 0.0) + float(value)
    return [{"feature": key, "value": round(value, 6), "feature_value": "encoded"} for key, value in totals.items()]


def _as_float(value: Any, default: Any) -> float:
    try:
        if value in {None, "", "NA"}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
