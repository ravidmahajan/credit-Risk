export type FairnessSlice = {
  dimension: "race" | "gender" | string;
  group: string;
  reference_group: string;
  n: number;
  approval_rate: number | null;
  observed_approval_rate: number | null;
  statistical_parity_difference: number | null;
  disparate_impact_ratio: number | null;
  equalized_odds: {
    tpr_difference: number | null;
    fpr_difference: number | null;
    max_difference: number | null;
  };
  four_fifths_rule: "pass" | "review" | "insufficient-reference" | string;
};

export type AuditResponse = {
  dataset: {
    source: string;
    scope: string;
    rows: number;
  };
  model: {
    version: string;
    backend: string;
    threshold: number;
    training: {
      rows: number;
      positive_rate: number;
      validation_auc: number | null;
      threshold: number;
      cross_validation: {
        folds: number;
        roc_auc_mean: number | null;
        roc_auc_std: number | null;
      };
      feature_policy: string;
    };
  };
  fairness: {
    policy: {
      max_disparate_impact_deviation: number;
      min_group_count: number;
      interpretation: string;
    };
    summary: {
      evaluated_slices: number;
      max_disparate_impact_deviation: number;
      passes_bias_regression: boolean;
    };
    slices: FairnessSlice[];
  };
};

export type ShapFeature = {
  feature: string;
  value: number;
  feature_value: string | number;
};

export type PredictionResponse = {
  prediction_id: string;
  decision: "approve" | "deny" | string;
  approved: boolean;
  probability: number;
  threshold: number;
  latency_ms: number;
  model_version: string;
  model_backend: string;
  protected_attributes: Record<string, string>;
  explanation_method: string;
  shap_vector: ShapFeature[];
  top_factors: ShapFeature[];
  force_plot: {
    base_value: number;
    output_value: number;
    link: string;
    features: ShapFeature[];
  };
};

export type Counterfactual = {
  target_decision: string;
  original_probability: number;
  counterfactual_probability: number;
  status?: string;
  interpretation?: string;
  changes: Array<{
    feature: string;
    new_value: number;
    probability: number;
    rationale: string;
  }>;
};

export type DemoState = {
  audit: AuditResponse;
  sampleApplicant: Record<string, string | number>;
  samplePrediction: PredictionResponse;
  counterfactual: Counterfactual;
};
