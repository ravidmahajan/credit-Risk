import { FormEvent, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  FileText,
  Gauge,
  GitBranch,
  Info,
  RefreshCw,
  Scale,
  Send,
  Server,
  ShieldCheck,
} from "lucide-react";
import { demoState } from "./demoState";
import type { Counterfactual, DemoState, FairnessSlice, PredictionResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
const numberFields = ["loan_amount", "loan_to_value_ratio", "income", "debt_to_income_ratio", "property_value", "loan_term"] as const;
const selectFields = {
  race: [
    { value: "White", label: "White" },
    { value: "Black or African American", label: "Black or African American" },
    { value: "Asian", label: "Asian" },
    { value: "Joint", label: "Joint" },
    { value: "Race Not Available", label: "Race not available" },
  ],
  gender: [
    { value: "Male", label: "Male" },
    { value: "Female", label: "Female" },
    { value: "Joint", label: "Joint" },
    { value: "Sex Not Available", label: "Sex not available" },
  ],
  loan_type: [
    { value: "1", label: "1 - Conventional" },
    { value: "2", label: "2 - FHA-insured" },
    { value: "3", label: "3 - VA-guaranteed" },
    { value: "4", label: "4 - USDA RHS/FSA" },
  ],
  loan_purpose: [
    { value: "1", label: "1 - Home purchase" },
    { value: "2", label: "2 - Home improvement" },
    { value: "31", label: "31 - Refinancing" },
    { value: "32", label: "32 - Cash-out refinancing" },
    { value: "4", label: "4 - Other purpose" },
    { value: "5", label: "5 - Not applicable" },
  ],
  applicant_credit_score_type: [
    { value: "1", label: "1 - Equifax Beacon 5.0" },
    { value: "2", label: "2 - Experian/FICO v2" },
    { value: "3", label: "3 - TransUnion FICO Classic 04" },
    { value: "7", label: "7 - More than one model" },
    { value: "8", label: "8 - Other credit scoring model" },
    { value: "9", label: "9 - Not applicable" },
  ],
} as const;

const fieldHelp: Record<string, string> = {
  loan_amount: "Requested mortgage amount in dollars.",
  loan_to_value_ratio: "Use percent points. 80 means the loan is 80% of the property value; 182 means the debt is above the property value.",
  income: "HMDA reports applicant income in thousands of dollars. 45 means about $45,000.",
  debt_to_income_ratio: "Use percent points. 43 means debt payments are 43% of monthly income. 1 is allowed but means an unusually low 1%, not 100%.",
  property_value: "Property value relied on in the credit decision, in dollars.",
  loan_term: "Loan term in months. 360 means a 30-year mortgage.",
  loan_type: "HMDA code for whether the loan is conventional or government-backed.",
  loan_purpose: "HMDA code for what the borrower is trying to do with the loan.",
  applicant_credit_score_type: "HMDA code for the scoring model used, not the credit score itself.",
  race: "Preserved for fairness audit, excluded from model features.",
  gender: "Preserved for fairness audit, excluded from model features.",
};

const dataDictionary = [
  ["Loan-to-value ratio", "Percent of property value covered by secured debt. Enter 80.05 for 80.05%, not 0.8005."],
  ["Debt-to-income ratio", "Percent of monthly income used for monthly debt. Enter 43 for 43%, not 0.43."],
  ["Loan type 2", "Federal Housing Administration insured (FHA)."],
  ["Loan purpose 2", "Home improvement."],
  ["Applicant credit score type 8", "Other credit scoring model. It describes the model source, not the actual score."],
  ["Structured explanation", "A machine-readable explanation: top factors, contribution values, and force-plot JSON for the prediction."],
  ["feature_perturbation_lime_fallback", "Local fallback explainer. It changes one feature at a time from a baseline applicant to estimate that feature's effect when SHAP is unavailable."],
  ["Base probability", "The model score for a typical baseline applicant before applying this applicant's feature values."],
  ["Threshold", "The calibrated cutoff. Probability above the threshold means approve; below it means deny."],
  ["Counterfactual", "The smallest suggested changes that would move this prediction toward the opposite decision."],
  ["Bias regression gate", "A CI guardrail that fails the build when disparate-impact drift exceeds the configured threshold."],
] as const;

const sourceLinks = [
  {
    label: "Official HMDA Data Browser CSV used here",
    href: "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?counties=17187&years=2023&actions_taken=1,3",
  },
  {
    label: "CFPB 2023 HMDA summary",
    href: "https://www.consumerfinance.gov/data-research/hmda/summary-of-2023-data-on-mortgage-lending/",
  },
  {
    label: "2023 HMDA reporting guide",
    href: "https://www.ffiec.gov/hmda/pdf/2023Guide.pdf",
  },
];

function App() {
  const [state, setState] = useState<DemoState>(demoState);
  const [applicant, setApplicant] = useState<Record<string, string | number>>(demoState.sampleApplicant);
  const [prediction, setPrediction] = useState<PredictionResponse>(demoState.samplePrediction);
  const [counterfactual, setCounterfactual] = useState<Counterfactual>(demoState.counterfactual);
  const [source, setSource] = useState<"live" | "demo">("demo");
  const [busy, setBusy] = useState(false);
  const [infoTab, setInfoTab] = useState<"why" | "charts" | "data" | "guide">("why");
  const [hasUnscoredChanges, setHasUnscoredChanges] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(apiUrl("/demo/state"))
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json() as Promise<DemoState>;
      })
      .then((payload) => {
        if (cancelled) return;
        setState(payload);
        setApplicant(payload.sampleApplicant);
        setPrediction(payload.samplePrediction);
        setCounterfactual(payload.counterfactual);
        setSource("live");
        setHasUnscoredChanges(false);
      })
      .catch(() => setSource("demo"));
    return () => {
      cancelled = true;
    };
  }, []);

  const raceSlices = useMemo(() => state.audit.fairness.slices.filter((slice) => slice.dimension === "race"), [state]);
  const genderSlices = useMemo(() => state.audit.fairness.slices.filter((slice) => slice.dimension === "gender"), [state]);
  const decisionTone = prediction.approved ? "good" : "bad";

  async function submitPrediction(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(apiUrl("/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(applicant),
      });
      if (!response.ok) throw new Error("Prediction request failed");
      const nextPrediction = (await response.json()) as PredictionResponse;
      setPrediction(nextPrediction);
      setSource("live");
      setHasUnscoredChanges(false);
      const explanation = await fetch(apiUrl(`/explain/${nextPrediction.prediction_id}`));
      if (explanation.ok) {
        const body = (await explanation.json()) as { counterfactual: Counterfactual };
        setCounterfactual(body.counterfactual);
      }
    } catch {
      setPrediction(demoState.samplePrediction);
      setCounterfactual(demoState.counterfactual);
      setSource("demo");
      setHasUnscoredChanges(false);
    } finally {
      setBusy(false);
    }
  }

  function updateApplicant(field: string, value: string | number) {
    setApplicant((current) => ({ ...current, [field]: value }));
    setHasUnscoredChanges(true);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Credit Risk Fairness Console">
          <span className="brand-mark">CR</span>
          <span>
            <strong>Credit Risk Fairness Console</strong>
            <small>HMDA explainability and audit service</small>
          </span>
        </a>
        <div className={`status-pill ${source === "live" ? "live" : ""}`}>
          <Server size={16} />
          {source === "live" ? "Live API" : "Demo State"}
        </div>
      </header>

      <section id="top" className="workspace">
        <div className="page-heading">
          <div>
            <p className="eyebrow">Production ML service</p>
            <h1>Explainable credit decisions with real-time fairness telemetry</h1>
          </div>
          <div className={`decision-tile ${decisionTone}`}>
            <span>Decision</span>
            <strong>{prediction.decision}</strong>
            <small>{asPercent(prediction.probability)} approval probability</small>
          </div>
        </div>

        <section className="metric-grid" aria-label="Model metrics">
          <Metric
            icon={<Database />}
            label="HMDA rows"
            value={state.audit.dataset.rows.toLocaleString()}
            description="Number of public mortgage application records used in this demo training set."
          />
          <Metric
            icon={<Gauge />}
            label="P95 target"
            value="<200ms"
            sub={`${prediction.latency_ms.toFixed(1)}ms last`}
            description="Latency goal: 95% of prediction requests should finish under 200 milliseconds."
          />
          <Metric
            icon={<GitBranch />}
            label="CV ROC AUC"
            value={valueOrDash(state.audit.model.training.cross_validation.roc_auc_mean)}
            description="Cross-validation ranking score. Closer to 1.0 means better separation of approvals and denials."
          />
          <Metric
            icon={<ShieldCheck />}
            label="DI max deviation"
            value={state.audit.fairness.summary.max_disparate_impact_deviation.toFixed(4)}
            sub={state.audit.fairness.summary.passes_bias_regression ? "gate passing" : "review"}
            description="Largest fairness drift from equal approval rates across audited demographic groups."
          />
        </section>

        <div className="main-grid">
          <section className="panel input-panel">
            <PanelTitle icon={<Send />} title="POST /predict" meta={prediction.model_backend} />
            <form onSubmit={submitPrediction} className="predict-form">
              {numberFields.map((field) => (
                <label key={field}>
                  <span>{labelize(field)}</span>
                  <input
                    type="number"
                    value={applicant[field] ?? ""}
                    min={0}
                    max={field === "debt_to_income_ratio" ? 100 : undefined}
                    step="0.01"
                    onChange={(event) => updateApplicant(field, Number(event.target.value))}
                  />
                  <small>{fieldHelp[field]}</small>
                </label>
              ))}
              {Object.entries(selectFields).map(([field, options]) => (
                <label key={field}>
                  <span>{labelize(field)}</span>
                  <select value={String(applicant[field] ?? "")} onChange={(event) => updateApplicant(field, event.target.value)}>
                    {options.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <small>{fieldHelp[field]}</small>
                </label>
              ))}
              <button className="primary-button" disabled={busy} type="submit">
                {busy ? <RefreshCw className="spin" size={18} /> : <Send size={18} />}
                Score applicant
              </button>
              {hasUnscoredChanges ? <p className="pending-note">Applicant values changed. Click Score applicant to update the decision and explanation.</p> : null}
            </form>
          </section>

          <section className="panel explanation-panel">
            <PanelTitle icon={<Activity />} title="Structured explanation" meta={humanMethod(prediction.explanation_method)} />
            <p className="panel-note">
              This is the model's reason payload: the approval score, cutoff threshold, top feature effects, and force-plot JSON returned by the API.
            </p>
            <div className="probability-band">
              <span>Base {asPercent(prediction.force_plot.base_value)}</span>
              <div className="probability-track">
                <i style={{ width: `${Math.max(2, prediction.probability * 100)}%` }} />
                <b style={{ left: `${prediction.threshold * 100}%` }} />
              </div>
              <span>Threshold {asPercent(prediction.threshold)}</span>
            </div>
            <div className="factor-list">
              {prediction.top_factors.map((factor) => (
                <ExplanationBar key={factor.feature} factor={factor} />
              ))}
            </div>
            <DecisionNarrative prediction={prediction} hasUnscoredChanges={hasUnscoredChanges} />
            <pre className="json-block">{JSON.stringify(prediction.force_plot, null, 2)}</pre>
          </section>
        </div>

        <section className="panel info-panel">
          <PanelTitle icon={<Info />} title="How to read this result" meta="Decision, data, and HMDA field guide" />
          <div className="tabbar" role="tablist" aria-label="Result information">
            <button className={infoTab === "why" ? "active" : ""} type="button" onClick={() => setInfoTab("why")}>
              <Activity size={16} />
              Why decision
            </button>
            <button className={infoTab === "charts" ? "active" : ""} type="button" onClick={() => setInfoTab("charts")}>
              <BarChart3 size={16} />
              Charts
            </button>
            <button className={infoTab === "data" ? "active" : ""} type="button" onClick={() => setInfoTab("data")}>
              <Database size={16} />
              Training data
            </button>
            <button className={infoTab === "guide" ? "active" : ""} type="button" onClick={() => setInfoTab("guide")}>
              <FileText size={16} />
              Field guide
            </button>
          </div>
          {infoTab === "why" ? <WhyDecision prediction={prediction} hasUnscoredChanges={hasUnscoredChanges} /> : null}
          {infoTab === "charts" ? <ChartsPanel applicant={applicant} prediction={prediction} state={state} hasUnscoredChanges={hasUnscoredChanges} /> : null}
          {infoTab === "data" ? <TrainingDataCard state={state} /> : null}
          {infoTab === "guide" ? <FieldGuide /> : null}
        </section>

        <section className="panel fairness-panel">
          <PanelTitle icon={<Scale />} title="GET /audit/fairness" meta={state.audit.dataset.scope} />
          <div className="fairness-legend" aria-label="Fairness metric definitions">
            <div>
              <strong>n</strong>
              <span>Number of records in that demographic slice.</span>
            </div>
            <div>
              <strong>SPD</strong>
              <span>Statistical Parity Difference: group approval rate minus reference group approval rate. Closer to 0 is better.</span>
            </div>
            <div>
              <strong>DI</strong>
              <span>Disparate Impact ratio: group approval rate divided by reference group approval rate. Closer to 1 is better.</span>
            </div>
            <div>
              <strong>EO</strong>
              <span>Equalized Odds gap: largest TPR/FPR error-rate difference versus the reference group. Closer to 0 is better.</span>
            </div>
          </div>
          <div className="fairness-grid">
            <FairnessTable title="Race slices" slices={raceSlices} />
            <FairnessTable title="Gender slices" slices={genderSlices} />
          </div>
        </section>

        <div className="main-grid secondary">
          <section className="panel">
            <PanelTitle icon={<BarChart3 />} title="Counterfactual" meta="GET /explain/{id}" />
            <div className="counterfactual-head">
              <strong>{asPercent(counterfactual.original_probability)}</strong>
              <span>to</span>
              <strong>{asPercent(counterfactual.counterfactual_probability)}</strong>
              <small>{counterfactual.target_decision} target</small>
            </div>
            <p className={`counterfactual-status ${counterfactual.status === "target-reached" ? "pass" : "warn"}`}>
              {counterfactual.status === "target-reached" ? "Target reached" : "Target not reached"}:{" "}
              {counterfactual.interpretation ?? "This is a what-if estimate, not a guaranteed lending recommendation."}
            </p>
            <ol className="change-list">
              {counterfactual.changes.length ? counterfactual.changes.slice(0, 6).map((change, index) => (
                <li key={`${change.feature}-${index}`}>
                  <span>{labelize(change.feature)}</span>
                  <strong>{change.new_value}</strong>
                  <small>{asPercent(change.probability)}</small>
                </li>
              )) : (
                <li>
                  <span>No beneficial single-step change found</span>
                  <strong>--</strong>
                  <small>{asPercent(counterfactual.original_probability)}</small>
                </li>
              )}
            </ol>
          </section>

          <section className="panel">
            <PanelTitle icon={<ShieldCheck />} title="Bias regression gate" meta="CI policy" />
            <div className={`gate ${state.audit.fairness.summary.passes_bias_regression ? "pass" : "fail"}`}>
              {state.audit.fairness.summary.passes_bias_regression ? <CheckCircle2 /> : <AlertTriangle />}
              <div>
                <strong>{state.audit.fairness.summary.passes_bias_regression ? "Passing" : "Review required"}</strong>
                <span>
                  max |1 - DI| = {state.audit.fairness.summary.max_disparate_impact_deviation}; limit ={" "}
                  {state.audit.fairness.policy.max_disparate_impact_deviation}
                </span>
                <small>
                  DI is disparate impact. This checks the worst fairness drift from equal approval rates across audited groups.
                </small>
              </div>
            </div>
            <dl className="model-card-strip">
              <div>
                <dt>Model</dt>
                <dd>{state.audit.model.version}</dd>
                <small>Version label: HMDA data, XGBoost target model, release 0.1.0.</small>
              </div>
              <div>
                <dt>Validation AUC</dt>
                <dd>{valueOrDash(state.audit.model.training.validation_auc)}</dd>
                <small>How well the model ranks approvals above denials on the validation split.</small>
              </div>
              <div>
                <dt>Positive rate</dt>
                <dd>{asPercent(state.audit.model.training.positive_rate)}</dd>
                <small>Share of training records labeled originated/approved.</small>
              </div>
            </dl>
            <p className="ci-note">
              CI policy means the automated GitHub Actions test fails if this fairness drift goes above the configured limit.
            </p>
          </section>
        </div>
      </section>
    </main>
  );
}

function Metric({ icon, label, value, sub, description }: { icon: ReactNode; label: string; value: string; sub?: string; description?: string }) {
  return (
    <div className="metric-card">
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
      {sub ? <small>{sub}</small> : null}
      {description ? <p>{description}</p> : null}
    </div>
  );
}

function PanelTitle({ icon, title, meta }: { icon: ReactNode; title: string; meta: string }) {
  return (
    <div className="panel-title">
      <span>{icon}</span>
      <h2>{title}</h2>
      <small>{meta}</small>
    </div>
  );
}

function ExplanationBar({ factor }: { factor: { feature: string; value: number; feature_value: string | number } }) {
  const width = Math.min(100, Math.abs(factor.value) * 210);
  const direction = factor.value >= 0 ? "positive" : "negative";
  return (
    <div className={`factor ${direction}`}>
      <div>
        <strong>{labelize(factor.feature)}</strong>
        <span>{String(factor.feature_value)}</span>
      </div>
      <div className="factor-track">
        <i style={{ width: `${width}%` }} />
      </div>
      <b>{factor.value > 0 ? "+" : ""}{factor.value.toFixed(4)}</b>
    </div>
  );
}

function DecisionNarrative({ prediction, hasUnscoredChanges }: { prediction: PredictionResponse; hasUnscoredChanges: boolean }) {
  const positive = prediction.top_factors.filter((factor) => factor.value > 0).slice(0, 2);
  const negative = prediction.top_factors.filter((factor) => factor.value < 0).slice(0, 3);
  const dominant = prediction.approved ? positive : negative;

  return (
    <div className="decision-copy">
      <strong>{prediction.approved ? "Approval drivers" : "Denial drivers"}</strong>
      {hasUnscoredChanges ? <em>Showing the last scored result. Score the applicant again to refresh these drivers.</em> : null}
      <p>
        The model compares the approval probability ({asPercent(prediction.probability)}) with the calibrated threshold ({asPercent(prediction.threshold)}).
        {prediction.approved ? " The strongest positive factors pushed the score above that threshold." : " The strongest negative factors kept the score below that threshold."}
      </p>
      <div>
        {dominant.map((factor) => (
          <span key={factor.feature}>{labelize(factor.feature)}</span>
        ))}
      </div>
    </div>
  );
}

function WhyDecision({ prediction, hasUnscoredChanges }: { prediction: PredictionResponse; hasUnscoredChanges: boolean }) {
  const negatives = prediction.top_factors.filter((factor) => factor.value < 0).slice(0, 4);
  const positives = prediction.top_factors.filter((factor) => factor.value > 0).slice(0, 4);

  return (
    <div className="tab-content split-explain">
      {hasUnscoredChanges ? <p className="tab-alert">You changed applicant inputs. This tab updates after you click Score applicant.</p> : null}
      <div>
        <h3>Reduced approval probability</h3>
        <ul>
          {negatives.map((factor) => (
            <li key={factor.feature}>
              <strong>{labelize(factor.feature)}</strong>
              <span>
                {String(factor.feature_value)} changed probability by {factor.value.toFixed(4)}
              </span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3>Increased approval probability</h3>
        <ul>
          {positives.map((factor) => (
            <li key={factor.feature}>
              <strong>{labelize(factor.feature)}</strong>
              <span>
                {String(factor.feature_value)} changed probability by +{factor.value.toFixed(4)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ChartsPanel({
  applicant,
  prediction,
  state,
  hasUnscoredChanges,
}: {
  applicant: Record<string, string | number>;
  prediction: PredictionResponse;
  state: DemoState;
  hasUnscoredChanges: boolean;
}) {
  return (
    <div className="tab-content charts-panel">
      {hasUnscoredChanges ? <p className="tab-alert">Input gauges update as you type. Probability and factor charts update after scoring.</p> : null}
      <div className="gauge-grid">
        <GaugeCard
          label="Approval probability"
          value={prediction.probability}
          max={1}
          display={asPercent(prediction.probability)}
          marker={prediction.threshold}
          help={`Decision threshold is ${asPercent(prediction.threshold)}.`}
        />
        <GaugeCard
          label="Loan-to-value ratio"
          value={toNumber(applicant.loan_to_value_ratio)}
          max={220}
          display={`${toNumber(applicant.loan_to_value_ratio).toFixed(2)}%`}
          marker={80 / 220}
          help="Lower LTV usually means more collateral cushion."
        />
        <GaugeCard
          label="Debt-to-income ratio"
          value={toNumber(applicant.debt_to_income_ratio)}
          max={100}
          display={`${toNumber(applicant.debt_to_income_ratio).toFixed(2)}%`}
          marker={43 / 100}
          help="Lower DTI usually means more income capacity."
        />
      </div>
      <div className="chart-grid">
        <TopFactorChart factors={prediction.top_factors} />
        <FairnessApprovalChart slices={state.audit.fairness.slices} />
      </div>
    </div>
  );
}

function GaugeCard({
  label,
  value,
  max,
  display,
  marker,
  help,
}: {
  label: string;
  value: number;
  max: number;
  display: string;
  marker?: number;
  help: string;
}) {
  const width = clamp((value / max) * 100, 0, 100);
  const markerLeft = marker === undefined ? undefined : clamp(marker * 100, 0, 100);
  return (
    <div className="gauge-card">
      <div>
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      <div className="gauge-track">
        <i style={{ width: `${width}%` }} />
        {markerLeft !== undefined ? <b style={{ left: `${markerLeft}%` }} /> : null}
      </div>
      <p>{help}</p>
    </div>
  );
}

function TopFactorChart({ factors }: { factors: PredictionResponse["top_factors"] }) {
  const max = Math.max(...factors.map((factor) => Math.abs(factor.value)), 0.01);
  return (
    <div className="mini-chart">
      <h3>Top factor impact</h3>
      {factors.slice(0, 6).map((factor) => (
        <div className="impact-row" key={factor.feature}>
          <span>{labelize(factor.feature)}</span>
          <div>
            <i className={factor.value >= 0 ? "positive" : "negative"} style={{ width: `${(Math.abs(factor.value) / max) * 100}%` }} />
          </div>
          <strong>{factor.value > 0 ? "+" : ""}{factor.value.toFixed(4)}</strong>
        </div>
      ))}
    </div>
  );
}

function FairnessApprovalChart({ slices }: { slices: FairnessSlice[] }) {
  const chartSlices = slices.filter((slice) => slice.approval_rate !== null);
  return (
    <div className="mini-chart">
      <h3>Training-data approval rates</h3>
      {chartSlices.map((slice) => (
        <div className="approval-row" key={`${slice.dimension}-${slice.group}`}>
          <span>
            {slice.group}
            <small>{slice.dimension}, n={slice.n}</small>
          </span>
          <div>
            <i style={{ width: `${(slice.approval_rate ?? 0) * 100}%` }} />
          </div>
          <strong>{asPercent(slice.approval_rate)}</strong>
        </div>
      ))}
    </div>
  );
}

function TrainingDataCard({ state }: { state: DemoState }) {
  return (
    <div className="tab-content data-card">
      <div className="data-summary">
        <div>
          <span>Rows</span>
          <strong>{state.audit.dataset.rows}</strong>
        </div>
        <div>
          <span>Target</span>
          <strong>Originated vs denied</strong>
        </div>
        <div>
          <span>Protected columns</span>
          <strong>Race, gender, ethnicity</strong>
        </div>
      </div>
      <p>
        The model is trained on a public HMDA loan-level sample: 2023 Warren County, Illinois applications where `action_taken`
        is either `1` loan originated or `3` application denied. Demographic columns are kept for fairness auditing but excluded
        from model features.
      </p>
      <div className="training-chart">
        <div>
          <span>Originated / approved</span>
          <strong>{asPercent(state.audit.model.training.positive_rate)}</strong>
          <i style={{ width: `${state.audit.model.training.positive_rate * 100}%` }} />
        </div>
        <div>
          <span>Denied</span>
          <strong>{asPercent(1 - state.audit.model.training.positive_rate)}</strong>
          <i style={{ width: `${(1 - state.audit.model.training.positive_rate) * 100}%` }} />
        </div>
      </div>
      <div className="source-list">
        {sourceLinks.map((link) => (
          <a key={link.href} href={link.href} target="_blank" rel="noreferrer">
            {link.label}
          </a>
        ))}
      </div>
    </div>
  );
}

function FieldGuide() {
  return (
    <div className="tab-content dictionary">
      <p className="static-note">This tab is a static dictionary. It does not change with applicant input.</p>
      {dataDictionary.map(([term, description]) => (
        <div key={term}>
          <strong>{term}</strong>
          <span>{description}</span>
        </div>
      ))}
    </div>
  );
}

function FairnessTable({ title, slices }: { title: string; slices: FairnessSlice[] }) {
  return (
    <div className="fairness-table">
      <h3>{title}</h3>
      <div className="table-head">
        <span>Group</span>
        <span>SPD</span>
        <span>DI</span>
        <span>EO</span>
      </div>
      {slices.map((slice) => (
        <div className="table-row" key={`${slice.dimension}-${slice.group}`}>
          <span>
            <strong>{slice.group}</strong>
            <small>n={slice.n}</small>
          </span>
          <MetricCell value={slice.statistical_parity_difference} signed />
          <MetricCell value={slice.disparate_impact_ratio} />
          <MetricCell value={slice.equalized_odds.max_difference} />
        </div>
      ))}
    </div>
  );
}

function MetricCell({ value, signed = false }: { value: number | null; signed?: boolean }) {
  if (value === null) return <span className="metric-cell muted">--</span>;
  const warning = signed ? Math.abs(value) > 0.1 : value < 0.8 || value > 1.25;
  return <span className={`metric-cell ${warning ? "warn" : "ok"}`}>{signed && value > 0 ? "+" : ""}{value.toFixed(4)}</span>;
}

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

function asPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function valueOrDash(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(4);
}

function labelize(value: string) {
  return value.replace(/_/g, " ");
}

function toNumber(value: string | number | undefined) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function humanMethod(value: string) {
  if (value === "feature_perturbation_lime_fallback") return "LIME-style fallback";
  if (value === "shap_tree") return "SHAP TreeExplainer";
  return labelize(value);
}

export default App;
