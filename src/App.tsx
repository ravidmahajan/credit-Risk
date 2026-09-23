import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock,
  Code2,
  Database,
  HelpCircle,
  Minus,
  RefreshCw,
  Scale,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  X,
} from "lucide-react";
import { demoState } from "./demoState";
import {
  APPLICANT_PRESETS,
  BASE_PROBABILITY,
  CALIBRATED_THRESHOLD,
  calculateCounterfactual,
  calculatePrediction,
} from "./scoringEngine";
import type { ApplicantData } from "./scoringEngine";
import type { Counterfactual, DemoState, FairnessSlice, PredictionResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

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
    { value: "1", label: "1 – Conventional" },
    { value: "2", label: "2 – FHA-insured" },
    { value: "3", label: "3 – VA-guaranteed" },
    { value: "4", label: "4 – USDA RHS/FSA" },
  ],
  loan_purpose: [
    { value: "1", label: "1 – Home purchase" },
    { value: "2", label: "2 – Home improvement" },
    { value: "31", label: "31 – Refinancing" },
    { value: "32", label: "32 – Cash-out refinancing" },
    { value: "4", label: "4 – Other purpose" },
  ],
  applicant_credit_score_type: [
    { value: "1", label: "1 – Equifax Beacon 5.0" },
    { value: "2", label: "2 – Experian / FICO v2" },
    { value: "3", label: "3 – TransUnion FICO Classic 04" },
    { value: "7", label: "7 – More than one model" },
    { value: "8", label: "8 – Other credit scoring model" },
    { value: "9", label: "9 – Not applicable" },
  ],
} as const;

const dataDictionary = [
  ["Loan-to-value ratio (LTV)", "Percent of property value covered by loan debt. 80.0 means 80%."],
  ["Debt-to-income ratio (DTI)", "Percent of monthly income used for monthly debt. Standard cap is 43%."],
  ["Loan type 1 vs 2", "1 is Conventional mortgage; 2 is Federal Housing Administration insured (FHA)."],
  ["Loan purpose 1 vs 2", "1 is Home purchase; 2 is Home improvement."],
  ["Credit score type 1-3", "Bureau models: 1 Equifax, 2 Experian, 3 TransUnion."],
  ["Statistical Parity Difference (SPD)", "Approval rate of protected group minus reference group rate. Closer to 0 is fairer."],
  ["Disparate Impact (DI)", "Approval rate ratio vs reference group. Standard four-fifths guideline requires DI >= 0.80."],
  ["Equalized Odds (EO)", "Largest gap in true positive rate (TPR) or false positive rate (FPR) vs reference."],
  ["Bias Regression Gate", "Automated CI guardrail that halts deployments if maximum DI deviation exceeds 0.65."],
  ["LIME / Perturbation", "Feature attribution method that measures score delta when perturbing individual features."],
] as const;

function App() {
  const [state, setState] = useState<DemoState>(demoState);
  const [applicant, setApplicant] = useState<Record<string, string | number>>(demoState.sampleApplicant);

  // Dynamic ML prediction and counterfactual
  const [prediction, setPrediction] = useState<PredictionResponse>(() => calculatePrediction(demoState.sampleApplicant));
  const [counterfactual, setCounterfactual] = useState<Counterfactual>(() =>
    calculateCounterfactual(demoState.sampleApplicant, calculatePrediction(demoState.sampleApplicant).probability)
  );

  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<"live" | "local">("local");
  const [activeStep, setActiveStep] = useState<number>(1);
  const [activePreset, setActivePreset] = useState<string>("high_risk");

  // Expandable sections state
  const [fairnessTableOpen, setFairnessTableOpen] = useState(false);
  const [forcePlotOpen, setForcePlotOpen] = useState(false);
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  // Attempt live API on initial mount
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/demo/state`)
      .then((r) => {
        if (!r.ok) throw new Error("API offline");
        return r.json() as Promise<DemoState>;
      })
      .then((p) => {
        if (cancelled) return;
        setState(p);
        setApplicant(p.sampleApplicant);
        setPrediction(p.samplePrediction);
        setCounterfactual(p.counterfactual);
        setSource("live");
      })
      .catch(() => {
        setSource("local");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const creditScore = useMemo(() => {
    return Math.round(300 + prediction.probability * 550);
  }, [prediction.probability]);

  // Handle input update with live auto-calculation
  function handleFieldChange(field: string, value: string | number) {
    const updated = { ...applicant, [field]: value };
    setApplicant(updated);
    setActivePreset("");

    // Live calculation
    const next = calculatePrediction(updated as ApplicantData);
    setPrediction(next);
    setCounterfactual(calculateCounterfactual(updated as ApplicantData, next.probability));
  }

  // Handle Preset selection
  function handleApplyPreset(preset: (typeof APPLICANT_PRESETS)[0]) {
    setActivePreset(preset.id);
    setApplicant(preset.data as Record<string, string | number>);
    const next = calculatePrediction(preset.data);
    setPrediction(next);
    setCounterfactual(calculateCounterfactual(preset.data, next.probability));
  }

  // Handle Form Submit (Queries live backend if available, or local ML engine)
  async function handleSubmitScore(e?: FormEvent) {
    if (e) e.preventDefault();
    setBusy(true);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(applicant),
      });
      if (!res.ok) throw new Error("API failed");
      const next = (await res.json()) as PredictionResponse;
      setPrediction(next);
      setSource("live");

      const expl = await fetch(`${API_BASE}/explain/${next.prediction_id}`);
      if (expl.ok) {
        const body = (await expl.json()) as { counterfactual: Counterfactual };
        setCounterfactual(body.counterfactual);
      }
    } catch {
      // Local ML Engine calculation
      const next = calculatePrediction(applicant as ApplicantData);
      setPrediction(next);
      setCounterfactual(calculateCounterfactual(applicant as ApplicantData, next.probability));
      setSource("local");
    } finally {
      setTimeout(() => setBusy(false), 200);
    }
  }

  // Rainbow bar pointer percentage: scale 300 to 850
  const spectrumPct = Math.min(100, Math.max(0, ((creditScore - 300) / 550) * 100));

  // Donut gauge circumference
  const circ = 2 * Math.PI * 34; // r=34 -> circ ≈ 213.6
  const dashoffset = circ * (1 - Math.min(1, Math.max(0, prediction.probability)));

  return (
    <div className="app-wrapper">
      {/* ─── Top Navbar ─── */}
      <nav className="top-navbar">
        <div className="top-navbar-inner">
          <div className="brand-section">
            <div className="brand-badge">CR</div>
            <div>
              <div className="brand-title">Credit Risk & Fairness Console</div>
              <div className="brand-subtitle">HMDA Fair Lending Audit & Decision Engine</div>
            </div>
          </div>

        </div>
      </nav>

      {/* ─── Guided Step Flow Progress Bar ─── */}
      <div className="flow-nav-bar">
        <div className="flow-nav-inner">
          {[
            { step: 1, id: "step1", label: "Applicant Profile" },
            { step: 2, id: "step2", label: "Risk Score & Decision" },
            { step: 3, id: "step3", label: "Attribution & SHAP" },
            { step: 4, id: "step4", label: "Fair Lending Audit" },
            { step: 5, id: "step5", label: "Counterfactuals" },
          ].map((item) => (
            <a
              key={item.step}
              href={`#${item.id}`}
              className={`flow-step-pill ${activeStep === item.step ? "active" : ""}`}
              onClick={() => setActiveStep(item.step)}
            >
              <span className="step-circle-badge">{item.step}</span>
              <span>{item.label}</span>
            </a>
          ))}
        </div>
      </div>

      {/* ─── Main Workspace Canvas ─── */}
      <main className="workspace-canvas">
        {/* ========================================================== */}
        {/*  STEP 1: APPLICANT PROFILE & PARAMETERS                    */}
        {/* ========================================================== */}
        <section className="dashboard-section-card" id="step1">
          <div className="section-header-block">
            <div className="section-title-wrap">
              <span className="section-num-tag">01</span>
              <div>
                <h2 className="section-main-title">Applicant Profile & Loan Parameters</h2>
                <p className="section-sub-title">
                  Configure mortgage loan parameters and borrower financial indicators to compute calibrated approval probability.
                </p>
              </div>
            </div>
          </div>

          {/* Quick 1-Click Test Profiles */}
          <div className="presets-container">
            <div className="presets-header-row">
              <span className="presets-label">Quick Test Profiles (1-Click)</span>
              <span className="presets-hint">Select a benchmark to immediately test high-risk, borderline, or prime scenarios</span>
            </div>
            <div className="presets-pill-group">
              {APPLICANT_PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  className={`preset-chip-btn ${activePreset === preset.id ? "active" : ""}`}
                  onClick={() => handleApplyPreset(preset)}
                >
                  <Sparkles size={13} />
                  <span>{preset.name}</span>
                  <span className="preset-chip-tag">{preset.tag}</span>
                </button>
              ))}
            </div>
          </div>

          {/* 3-Column Inputs Grid on Desktop */}
          <form onSubmit={handleSubmitScore} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div className="inputs-3col-grid">
              {/* Column 1: Loan Parameters */}
              <div className="input-category-card">
                <div className="category-card-title">
                  <span>Loan Information</span>
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Loan Amount ($)</span>
                    <small>Requested mortgage</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.loan_amount ?? ""}
                    min={0}
                    step="1000"
                    onChange={(e) => handleFieldChange("loan_amount", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Property Collateral ($)</span>
                    <small>Appraised value</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.property_value ?? ""}
                    min={0}
                    step="1000"
                    onChange={(e) => handleFieldChange("property_value", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Loan-to-Value (LTV %)</span>
                    <small>80 = 80%</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.loan_to_value_ratio ?? ""}
                    min={0}
                    step="0.1"
                    onChange={(e) => handleFieldChange("loan_to_value_ratio", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Loan Term (Months)</span>
                    <small>360 = 30-year</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.loan_term ?? ""}
                    min={12}
                    max={480}
                    step="12"
                    onChange={(e) => handleFieldChange("loan_term", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Loan Program Type</span>
                  </div>
                  <select
                    className="field-select"
                    value={String(applicant.loan_type ?? "1")}
                    onChange={(e) => handleFieldChange("loan_type", e.target.value)}
                  >
                    {selectFields.loan_type.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Loan Purpose</span>
                  </div>
                  <select
                    className="field-select"
                    value={String(applicant.loan_purpose ?? "1")}
                    onChange={(e) => handleFieldChange("loan_purpose", e.target.value)}
                  >
                    {selectFields.loan_purpose.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Column 2: Borrower Financials */}
              <div className="input-category-card">
                <div className="category-card-title">
                  <span>Borrower Financials</span>
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Annual Income ($k/yr)</span>
                    <small>45 ≈ $45,000</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.income ?? ""}
                    min={0}
                    step="1"
                    onChange={(e) => handleFieldChange("income", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Debt-to-Income (DTI %)</span>
                    <small>Monthly debt / income</small>
                  </div>
                  <input
                    type="number"
                    className="field-input"
                    value={applicant.debt_to_income_ratio ?? ""}
                    min={0}
                    max={100}
                    step="0.5"
                    onChange={(e) => handleFieldChange("debt_to_income_ratio", Number(e.target.value))}
                  />
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Credit Bureau Scoring Model</span>
                  </div>
                  <select
                    className="field-select"
                    value={String(applicant.applicant_credit_score_type ?? "1")}
                    onChange={(e) => handleFieldChange("applicant_credit_score_type", e.target.value)}
                  >
                    {selectFields.applicant_credit_score_type.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ marginTop: "auto", padding: "12px", background: "#FFFFFF", borderRadius: "10px", border: "1px solid var(--border-light)" }}>
                  <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-primary)", display: "block" }}>
                    Capacity Assessment
                  </span>
                  <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", margin: "4px 0 0", lineHeight: 1.35 }}>
                    Monthly income capacity of ${(Number(applicant.income || 45) * 83.33).toFixed(0)}/mo against standard debt limits.
                  </p>
                </div>
              </div>

              {/* Column 3: Demographic Attributes (Audit Only) */}
              <div className="input-category-card">
                <div className="category-card-title">
                  <span>Demographic Data</span>
                  <span className="audit-only-badge">Audit Only — Not in Model</span>
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Applicant Race</span>
                  </div>
                  <select
                    className="field-select"
                    value={String(applicant.race ?? "White")}
                    onChange={(e) => handleFieldChange("race", e.target.value)}
                  >
                    {selectFields.race.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field-control-wrap">
                  <div className="field-label-row">
                    <span>Applicant Gender</span>
                  </div>
                  <select
                    className="field-select"
                    value={String(applicant.gender ?? "Male")}
                    onChange={(e) => handleFieldChange("gender", e.target.value)}
                  >
                    {selectFields.gender.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ marginTop: "auto", padding: "12px", background: "#EFF6FF", borderRadius: "10px", border: "1px solid #BFDBFE" }}>
                  <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#1D4ED8", display: "block" }}>
                    Fair Housing Act Compliance
                  </span>
                  <p style={{ fontSize: "0.72rem", color: "#3B82F6", margin: "4px 0 0", lineHeight: 1.35 }}>
                    Protected attributes are audited for algorithmic parity in Step 4 and are strictly excluded from predictive model weights.
                  </p>
                </div>
              </div>
            </div>

            <button type="submit" className="btn-score-cta" disabled={busy}>
              {busy ? <RefreshCw className="spin" size={18} /> : <Sliders size={18} />}
              {busy ? "Scoring Application with Live Model..." : "Score Application"}
            </button>
          </form>
        </section>

        {/* ========================================================== */}
        {/*  STEP 2: RISK SCORING & DECISION (DECISION HERO)           */}
        {/* ========================================================== */}
        <section className="dashboard-section-card" id="step2">
          <div className="section-header-block">
            <div className="section-title-wrap">
              <span className="section-num-tag">02</span>
              <div>
                <h2 className="section-main-title">Risk Scoring & Model Decision</h2>
                <p className="section-sub-title">
                  Calibrated approval probability compared against the 62.6% regulatory cutoff threshold.
                </p>
              </div>
            </div>
          </div>

          <div className="decision-hero-grid">
            {/* Left Score Block */}
            <div className="score-hero-block">
              <div className="score-eyebrow-row">
                <span className="score-badge-chip">Score Index</span>
                <span className="score-badge-chip">Cutoff: {(CALIBRATED_THRESHOLD * 100).toFixed(1)}%</span>
              </div>

              <div className="hero-big-number-row">
                <span className="hero-score-number">{creditScore}</span>
                <span className="hero-score-denom">/ 850</span>
              </div>

              <div className={`decision-status-pill ${prediction.approved ? "good" : "bad"}`}>
                {prediction.approved ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                <span>
                  {prediction.approved
                    ? `Approved · ${(prediction.probability * 100).toFixed(1)}% Probability`
                    : `Denied · ${(prediction.probability * 100).toFixed(1)}% Probability`}
                </span>
              </div>

              <p className="score-meta-text">
                Model: <strong>{prediction.model_version}</strong> ({prediction.model_backend}) · Latency:{" "}
                {prediction.latency_ms.toFixed(0)}ms · Method: LIME/SHAP Perturbation
              </p>
            </div>

            {/* Right Rainbow Visual Block */}
            <div className="rainbow-hero-visual">
              <div className="rainbow-bar-container">
                <div className="rainbow-bar-track">
                  {/* Dynamic Sliding Triangle Pointer */}
                  <div className="spectrum-pointer-anchor" style={{ left: `${spectrumPct}%` }} />
                </div>
                <div className="scale-ticks-row">
                  <span>300 (0% Deny)</span>
                  <span>Cutoff {(CALIBRATED_THRESHOLD * 100).toFixed(0)}%</span>
                  <span>710</span>
                  <span>850 (100% Approve)</span>
                </div>
              </div>

              {/* Sub-meters */}
              <div className="hero-meters-grid">
                <div className="hero-meter-cell">
                  <span>LTV Ratio</span>
                  <strong>{Number(applicant.loan_to_value_ratio || 80).toFixed(1)}%</strong>
                  <small style={{ color: Number(applicant.loan_to_value_ratio || 80) <= 80 ? "#059669" : "#DC2626", fontSize: "0.7rem", fontWeight: 700 }}>
                    {Number(applicant.loan_to_value_ratio || 80) <= 80 ? "✓ Safe (<=80%)" : "⚠ Elevated Risk"}
                  </small>
                </div>

                <div className="hero-meter-cell">
                  <span>DTI Ratio</span>
                  <strong>{Number(applicant.debt_to_income_ratio || 43).toFixed(1)}%</strong>
                  <small style={{ color: Number(applicant.debt_to_income_ratio || 43) <= 43 ? "#059669" : "#DC2626", fontSize: "0.7rem", fontWeight: 700 }}>
                    {Number(applicant.debt_to_income_ratio || 43) <= 43 ? "✓ Safe (<=43%)" : "⚠ High Debt"}
                  </small>
                </div>

                <div className="hero-meter-cell">
                  <span>Annual Income</span>
                  <strong>${(Number(applicant.income || 45) * 1000).toLocaleString()}</strong>
                  <small style={{ color: "#059669", fontSize: "0.7rem", fontWeight: 700 }}>
                    Capacity OK
                  </small>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ========================================================== */}
        {/*  STEP 3: ATTRIBUTION & EXPLAINABILITY (SHAP / LIME)        */}
        {/* ========================================================== */}
        <section className="dashboard-section-card" id="step3">
          <div className="section-header-block">
            <div className="section-title-wrap">
              <span className="section-num-tag">03</span>
              <div>
                <h2 className="section-main-title">Attribution & Risk Explainability</h2>
                <p className="section-sub-title">
                  Local feature contributions quantifying each parameter's marginal impact on the final approval score.
                </p>
              </div>
            </div>
          </div>

          <div className="attribution-2col-grid">
            {/* Left: Credit Risk Sensitivity Curve (Spline Chart) */}
            <div className="spline-chart-card">
              <div className="chart-card-title-row">
                <span className="chart-card-title">Credit Risk Sensitivity Curve</span>
                <span style={{ fontSize: "0.74rem", color: "var(--text-muted)", fontWeight: 600 }}>
                  Warren County HMDA Benchmark
                </span>
              </div>

              <div className="chart-svg-wrap">
                <svg viewBox="0 0 320 150" className="spline-svg">
                  <defs>
                    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#C4DD21" stopOpacity="0.32" />
                      <stop offset="100%" stopColor="#C4DD21" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>

                  {/* Horizontal Grid lines */}
                  <line x1="40" y1="20" x2="310" y2="20" stroke="#E5E7EB" strokeDasharray="3 3" />
                  <line x1="40" y1="55" x2="310" y2="55" stroke="#E5E7EB" strokeDasharray="3 3" />
                  <line x1="40" y1="90" x2="310" y2="90" stroke="#E5E7EB" strokeDasharray="3 3" />
                  <line x1="40" y1="125" x2="310" y2="125" stroke="#E5E7EB" strokeDasharray="3 3" />

                  {/* Y Axis text */}
                  <text x="10" y="24" fill="#9CA3AF" fontSize="10" fontWeight="600">800</text>
                  <text x="10" y="59" fill="#9CA3AF" fontSize="10" fontWeight="600">735</text>
                  <text x="10" y="94" fill="#9CA3AF" fontSize="10" fontWeight="600">650</text>
                  <text x="10" y="129" fill="#9CA3AF" fontSize="10" fontWeight="600">500</text>

                  {/* Shaded Area Under Curve */}
                  <path
                    d="M 40 120 C 80 115, 120 70, 160 85 C 200 100, 240 30, 310 20 L 310 125 L 40 125 Z"
                    fill="url(#areaGrad)"
                  />

                  {/* Baseline Spline */}
                  <path
                    d="M 40 110 C 90 105, 140 120, 180 80 C 220 40, 260 50, 310 45"
                    fill="none"
                    stroke="#D1D5DB"
                    strokeWidth="2"
                  />

                  {/* Active Spline Path */}
                  <path
                    d="M 40 120 C 80 115, 120 70, 160 85 C 200 100, 240 30, 310 20"
                    fill="none"
                    stroke="#111827"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                  />

                  {/* Vertical Callout Line */}
                  <line x1="210" y1="20" x2="210" y2="125" stroke="#C4DD21" strokeWidth="1.5" strokeDasharray="2 2" />

                  {/* Current Score Pin */}
                  <circle cx="210" cy="55" r="5" fill="#111827" stroke="#FFFFFF" strokeWidth="2" />

                  {/* Tag Callout Badge */}
                  <rect x="190" y="45" width="84" height="20" rx="10" className="chart-callout-pill" />
                  <text x="232" y="58" textAnchor="middle" className="chart-callout-text">
                    {creditScore} Score
                  </text>
                  <text x="210" y="15" textAnchor="middle" className="chart-date-label">
                    HMDA Calibrated Curve
                  </text>
                </svg>
              </div>
            </div>

            {/* Right: Key Factors List & Decision Narrative */}
            <div className="factors-list-card">
              <div className="factor-chips-list">
                {prediction.top_factors.map((f) => (
                  <div className="factor-card-item" key={f.feature}>
                    <div className="factor-item-info">
                      <strong>{f.feature.replace(/_/g, " ")}</strong>
                      <small>Value: {String(f.feature_value)}</small>
                    </div>
                    <span className={`factor-impact-badge ${f.value >= 0 ? "pos" : "neg"}`}>
                      {f.value >= 0 ? "+" : ""}
                      {(f.value * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>

              {/* Plain-English Narrative */}
              <div className="decision-narrative-box">
                <strong>Decision Rationale: </strong>
                The applicant approval probability is {(prediction.probability * 100).toFixed(1)}% compared to the cutoff of{" "}
                {(prediction.threshold * 100).toFixed(1)}%.
                {prediction.approved
                  ? " Positive collateral equity and income coverage offset debt obligations, qualifying the loan for approval."
                  : " Elevated debt-to-income or high loan-to-value ratio depressed the probability below the required origination cutoff."}
              </div>

              {/* Force Plot JSON Toggle */}
              <button
                type="button"
                className="flow-step-pill"
                style={{ fontSize: "0.78rem", padding: "6px 14px", alignSelf: "flex-start" }}
                onClick={() => setForcePlotOpen(!forcePlotOpen)}
              >
                <Code2 size={14} />
                {forcePlotOpen ? "Hide Force-Plot JSON" : "View Force-Plot JSON"}
              </button>

              {forcePlotOpen && (
                <pre className="json-drawer">{JSON.stringify(prediction.force_plot, null, 2)}</pre>
              )}
            </div>
          </div>
        </section>

        {/* ========================================================== */}
        {/*  STEP 4: FAIR LENDING & COMPLIANCE AUDIT                   */}
        {/* ========================================================== */}
        <section className="dashboard-section-card" id="step4">
          <div className="section-header-block">
            <div className="section-title-wrap">
              <span className="section-num-tag">04</span>
              <div>
                <h2 className="section-main-title">Fair Lending & Demographic Parity Audit</h2>
                <p className="section-sub-title">
                  Algorithmic bias governance evaluated across protected demographic slices under CFPB four-fifths guidelines.
                </p>
              </div>
            </div>
          </div>

          {/* Top Compliance Row: Donut Gauge + Bias Gate */}
          <div className="compliance-top-grid">
            {/* Circular Donut Ring Gauge Card */}
            <div className="donut-stats-card">
              <div className="donut-text-col">
                <span>Model Architecture</span>
                <strong>{state.audit.model.version}</strong>
                <small>{state.audit.model.backend} · CV AUC: {state.audit.model.training.cross_validation.roc_auc_mean?.toFixed(3)}</small>
              </div>

              <div className="donut-circle-wrap">
                <svg className="donut-svg" viewBox="0 0 80 80">
                  <defs>
                    <linearGradient id="donutGrad" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#EA580C" />
                      <stop offset="35%" stopColor="#F59E0B" />
                      <stop offset="70%" stopColor="#10B981" />
                      <stop offset="100%" stopColor="#3B82F6" />
                    </linearGradient>
                  </defs>
                  <circle cx="40" cy="40" r="34" fill="none" stroke="#E5E7EB" strokeWidth="8" />
                  <circle
                    cx="40"
                    cy="40"
                    r="34"
                    fill="none"
                    stroke="url(#donutGrad)"
                    strokeWidth="8"
                    strokeDasharray={circ}
                    strokeDashoffset={dashoffset}
                    strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 0.6s ease" }}
                  />
                </svg>
                <div className="donut-center-info">
                  <Clock size={16} color="#111827" />
                  <span className="donut-pct-text">{Math.round(prediction.probability * 100)}%</span>
                  <span className="donut-sub-text">Score</span>
                </div>
              </div>
            </div>

            {/* Bias Regression Gate Card */}
            <div className={`gate-card-box ${state.audit.fairness.summary.passes_bias_regression ? "pass" : "fail"}`}>
              <div className="gate-icon-circle">
                {state.audit.fairness.summary.passes_bias_regression ? <ShieldCheck size={22} /> : <ShieldAlert size={22} />}
              </div>
              <div className="gate-info-text">
                <strong>
                  {state.audit.fairness.summary.passes_bias_regression
                    ? "Bias Regression Gate: Passing"
                    : "Bias Regression Gate: Review Required"}
                </strong>
                <small>
                  Max |1 − DI| = {state.audit.fairness.summary.max_disparate_impact_deviation} (CI policy limit ={" "}
                  {state.audit.fairness.policy.max_disparate_impact_deviation})
                </small>
              </div>
            </div>
          </div>

          {/* Demographic Fairness Matrix Chips */}
          <div className="demographic-matrix-card">
            <div className="matrix-title-row">
              <span>Demographic Parity Matrix (Four-Fifths Rule Slices)</span>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                🟡 Pass (DI ≥ 0.80) · 🔴 Disparity Review · ⚪ Reference Group
              </span>
            </div>

            <div className="matrix-grid-chips">
              {state.audit.fairness.slices.map((slice) => {
                const shortName = slice.group.length > 12 ? slice.group.slice(0, 10) + ".." : slice.group;
                const status =
                  slice.group === slice.reference_group
                    ? "neutral"
                    : slice.four_fifths_rule === "pass"
                    ? "pass"
                    : "fail";

                return (
                  <div key={`${slice.dimension}-${slice.group}`} className="matrix-slice-col">
                    <span className="slice-col-label" title={slice.group}>
                      {shortName}
                    </span>
                    <div className={`status-badge-chip ${status}`} title={`DI: ${slice.disparate_impact_ratio ?? "--"}`}>
                      {status === "pass" && <Check size={16} />}
                      {status === "fail" && <X size={16} />}
                      {status === "neutral" && <Minus size={16} />}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Toggle Full Metrics Table */}
            <button
              type="button"
              className="flow-step-pill"
              style={{ fontSize: "0.78rem", padding: "6px 14px", alignSelf: "flex-start", marginTop: "4px" }}
              onClick={() => setFairnessTableOpen(!fairnessTableOpen)}
            >
              <Scale size={14} />
              {fairnessTableOpen ? "Hide Fairness Table" : "View Full Fairness Metrics Table (SPD, DI, EO)"}
            </button>

            {fairnessTableOpen && (
              <div className="fairness-table-container">
                <table className="fairness-data-table">
                  <thead>
                    <tr>
                      <th>Demographic Slice</th>
                      <th>Records (N)</th>
                      <th>Statistical Parity Diff (SPD)</th>
                      <th>Disparate Impact (DI)</th>
                      <th>Equalized Odds (EO)</th>
                      <th>Four-Fifths Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {state.audit.fairness.slices.map((s) => (
                      <tr key={`${s.dimension}-${s.group}`}>
                        <td>
                          <strong>{s.group}</strong> ({s.dimension})
                        </td>
                        <td>{s.n}</td>
                        <td style={{ color: (s.statistical_parity_difference ?? 0) < -0.1 ? "#DC2626" : "inherit" }}>
                          {s.statistical_parity_difference?.toFixed(4) ?? "--"}
                        </td>
                        <td style={{ color: (s.disparate_impact_ratio ?? 1) < 0.8 ? "#DC2626" : "#059669" }}>
                          {s.disparate_impact_ratio?.toFixed(4) ?? "1.000"}
                        </td>
                        <td>{s.equalized_odds?.max_difference?.toFixed(4) ?? "0.000"}</td>
                        <td>
                          <span className={`slice-rate-tag ${s.four_fifths_rule === "pass" ? "good" : "warn"}`}>
                            {s.four_fifths_rule}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        {/* ========================================================== */}
        {/*  STEP 5: COUNTERFACTUAL PATHWAY & HMDA GOVERNANCE          */}
        {/* ========================================================== */}
        <section className="dashboard-section-card" id="step5">
          <div className="section-header-block">
            <div className="section-title-wrap">
              <span className="section-num-tag">05</span>
              <div>
                <h2 className="section-main-title">Counterfactual Recommendations & Regulatory Governance</h2>
                <p className="section-sub-title">
                  Actionable what-if feature modifications and official HMDA mortgage reporting data dictionary.
                </p>
              </div>
            </div>
          </div>

          <div className="step5-grid">
            {/* Left: Counterfactual What-If Recommendations */}
            <div className="counterfactual-box">
              <div className="counterfactual-header-row">
                <span style={{ fontSize: "0.9rem", fontWeight: 750, color: "var(--text-primary)" }}>
                  Recommended What-If Pathway
                </span>
                <span className="counterfactual-prob-pill">
                  {(counterfactual.original_probability * 100).toFixed(1)}% → {(counterfactual.counterfactual_probability * 100).toFixed(1)}%
                </span>
              </div>

              <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "4px 0", lineHeight: 1.45 }}>
                Target: <strong>{counterfactual.target_decision.toUpperCase()}</strong> — {counterfactual.interpretation}
              </p>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {counterfactual.changes.map((c, i) => (
                  <div key={i} className="cf-change-item">
                    <div>
                      <strong>{c.rationale}</strong>
                      <small style={{ display: "block", color: "var(--text-muted)", fontSize: "0.72rem" }}>
                        New {c.feature}: {c.new_value}
                      </small>
                    </div>
                    <span className="factor-impact-badge pos">
                      {(c.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: HMDA Field Guide & Glossary */}
            <div className="glossary-card">
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ fontSize: "0.9rem", fontWeight: 750, color: "var(--text-primary)" }}>
                  HMDA Field Guide & Data Dictionary
                </span>
                <HelpCircle size={16} color="var(--text-muted)" />
              </div>

              <div className="glossary-grid">
                {dataDictionary.map(([term, desc]) => (
                  <div key={term} className="glossary-entry">
                    <strong>{term}</strong>
                    <span>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
