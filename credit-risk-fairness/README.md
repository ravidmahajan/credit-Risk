# Explainable Credit Risk & Fairness Auditing API

Production-oriented FastAPI + React project for real-time credit risk scoring, structured explanations, and demographic fairness audits.

## What Is Included

- FastAPI REST API with OpenAPI docs at `/docs`
- HMDA data pipeline that preserves protected columns for audit while excluding them from model features
- XGBoost classifier when `xgboost` is installed, with a scikit-learn gradient boosting fallback for local environments
- Threshold calibration and stratified cross-validation
- SHAP TreeExplainer path with a vectorized LIME-style perturbation fallback
- Fairness metrics: Statistical Parity Difference, Equalized Odds, Disparate Impact Ratio
- Prometheus metrics at `/metrics`
- Postgres persistence for predictions when `DATABASE_URL` is set
- React dashboard with live API mode and offline demo mode
- Docker Compose stack for API, dashboard, PostgreSQL, Prometheus, and Grafana
- Kubernetes deployment manifests
- CI bias regression test that fails when `abs(1 - disparate_impact_ratio)` exceeds `BIAS_MAX_DI_DEVIATION`
- Companion notebook and EMCS model card

## Data Source

The checked-in demo dataset is a public HMDA loan-level CSV from the FFIEC/CFPB Data Browser:

```text
https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?counties=17187&years=2023&actions_taken=1,3
```

Scope: Warren County, Illinois, 2023, originated applications (`action_taken=1`) and denied applications (`action_taken=3`).

HMDA public loan-level data are modified to protect applicant and borrower privacy. For a larger live pull:

```bash
cd credit-risk-fairness
python backend/scripts/bootstrap_hmda.py --state IL --year 2023 --out backend/data/hmda_live.csv
```

Then run with:

```bash
HMDA_DATA_PATH=backend/data/hmda_live.csv uvicorn app.main:app --app-dir backend --reload
```

## Local Run

Backend:

```bash
cd credit-risk-fairness
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Dashboard:

```bash
npm run credit:dev
```

Open `http://127.0.0.1:5174`.

The dashboard falls back to a static demo state if the API is not running.

## API Surface

```http
POST /predict
GET /audit/fairness
GET /explain/{prediction_id}
GET /metrics
GET /health
```

Example prediction body:

```json
{
  "loan_amount": 45000,
  "loan_to_value_ratio": 182.62,
  "income": 45,
  "debt_to_income_ratio": 43,
  "property_value": 25000,
  "loan_term": 120,
  "loan_type": "1",
  "loan_purpose": "2",
  "lien_status": "1",
  "occupancy_type": "1",
  "race": "White",
  "gender": "Male",
  "ethnicity": "Hispanic or Latino"
}
```

## Docker Stack

```bash
cd credit-risk-fairness/infra
docker compose up --build
```

- Dashboard: `http://127.0.0.1:8080`
- API docs: `http://127.0.0.1:8000/docs`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3001` (`admin` / `credit`)

## Verification

```bash
cd credit-risk-fairness
python -m pytest
npm run credit:build
```

The current local fallback backend scored ten demo predictions at roughly 42-66 ms each after training on this machine. The API exposes Prometheus histograms so p95 can be monitored in Grafana under production load.

