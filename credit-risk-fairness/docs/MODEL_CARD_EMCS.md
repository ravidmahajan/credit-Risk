# Model Card: EMCS Format

## E: Evidence

**Dataset:** Public HMDA Data Browser loan-level CSV from FFIEC/CFPB.

**Demo scope:** 2023 Warren County, Illinois applications with `action_taken` in `{1 originated, 3 denied}`.

**Rows:** 230 in the checked-in fixture.

**Target:** `approved = 1` when HMDA `action_taken == 1`; `approved = 0` when `action_taken == 3`.

**Protected attributes preserved for audit:** `derived_race`, `derived_sex`, `derived_ethnicity`.

**Protected attributes excluded from model features:** race, gender, and ethnicity are not used as predictors in the default feature set.

## M: Model

**Primary estimator:** XGBoost binary classifier when the production dependency is installed.

**Local fallback:** scikit-learn `GradientBoostingClassifier`, used when `xgboost` is unavailable.

**Feature pipeline:** median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.

**Calibration:** validation-set F1 threshold search constrained by `APPROVAL_THRESHOLD_FLOOR` and `APPROVAL_THRESHOLD_CEILING`.

**Validation:** stratified train/test split and stratified cross-validation. The fixture run produced mean CV ROC AUC near `0.7789`.

## C: Context

**Intended use:** Auditable demonstration service for credit-risk ML architecture, API behavior, explanation payloads, and fairness monitoring.

**Not intended use:** Real lending decisions without institution-specific validation, adverse action review, model governance approval, legal review, and ongoing monitoring.

**Latency objective:** `<200ms` p95 prediction latency after model warm-up. SHAP is used for XGBoost deployments; the fallback explanation path batches perturbations to stay low-latency.

**Operational interfaces:** FastAPI OpenAPI docs, React dashboard, Prometheus metrics, Grafana datasource, Docker Compose, Kubernetes manifests.

## S: Safety, Fairness, And Stewardship

**Fairness metrics:** Statistical Parity Difference, Equalized Odds, and Disparate Impact Ratio across race and gender slices.

**Bias regression gate:** CI fails when `abs(1 - disparate_impact_ratio)` exceeds `BIAS_MAX_DI_DEVIATION`.

**Current fixture audit:** max disparate impact deviation `0.1682`, passing the configured `0.65` gate.

**Known limitations:** Small county-level demo data is useful for UI/API verification, not representative production validation. Some HMDA fields are categorical codes and should be mapped to institution-readable labels before stakeholder review.

**Monitoring:** Request latency histogram and prediction decision counter are exported at `/metrics`. Prediction records persist to PostgreSQL when `DATABASE_URL` is configured.

**Review cadence:** Recompute fairness audits when data vintage, geography, model features, or threshold policy changes.

