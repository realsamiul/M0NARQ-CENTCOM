export type KpiStatus = "ON_TRACK" | "EXCEEDED" | "AT_RISK" | "BEHIND" | string;

export interface KpiSummary {
  metric: string;
  value: number;
  unit: string;
  target: number;
  status: KpiStatus;
}

export interface DemoCard {
  demo_id: string;
  name: string;
  status: "PRODUCTION" | "DEVELOPMENT" | string;
  primary_metric: string;
}

export interface ModelComparisonRow {
  model: string;
  accuracy_pct?: number | null;
  params_m?: number | null;
  inference_ms?: number | null;
  demo: string;
  r2_score?: number | null;
  params_k?: number | null;
  mape_pct?: number | null;
}

export interface TrainingTrend {
  month: string; // YYYY-MM
  accuracy_pct: number;
  training_time_hrs: number;
}

export interface ApiUsageTrend {
  month: string; // YYYY-MM
  requests_k: number;
}

export interface DashboardJson {
  metadata: Record<string, unknown>;
  system_overview: {
    total_code_lines: number;
    total_datasets: number;
    satellite_data_sources: number;
    ml_models_deployed: number;
    cloud_storage_tb: number;
    api_calls_daily: number;
  };
  demo_1_flood: Record<string, unknown>;
  demo_2_crop: Record<string, unknown>;
  demo_3_urban: Record<string, unknown>;
  demo_4_freight: Record<string, unknown>;
  demo_5_lpg: Record<string, unknown>;
  demo_6_deforestation: Record<string, unknown>;
  model_comparison: Array<Record<string, unknown>>;
  infrastructure_metrics: Record<string, unknown>;
  satellite_data: Record<string, unknown>;
  business_impact: Record<string, unknown>;
  performance_trends: {
    model_training: TrainingTrend[];
    api_usage: ApiUsageTrend[];
  };
  kpi_summary: KpiSummary[];
}
