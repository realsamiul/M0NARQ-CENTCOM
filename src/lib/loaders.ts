"use server";

import { PATHS } from "./data-sources";
import type {
  ApiUsageTrend,
  DashboardJson,
  DemoCard,
  KpiSummary,
  ModelComparisonRow,
  TrainingTrend,
} from "@/types/dashboard";

// Lightweight CSV parser (no extra deps): assumes comma-separated values
// Note: This simple parser expects clean CSV data without commas in values
// For complex CSVs with quoted fields, consider using a dedicated CSV library
function parseCsv(raw: string): string[][] {
  return raw
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(","));
}

function getBaseUrl() {
  // In production, use the actual domain. In dev, use localhost with dynamic port
  if (process.env.NEXT_PUBLIC_BASE_URL) {
    return process.env.NEXT_PUBLIC_BASE_URL;
  }
  // For server-side in development
  const port = process.env.PORT || 3000;
  return `http://localhost:${port}`;
}

async function fetchText(path: string): Promise<string> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.text();
}

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${getBaseUrl()}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json() as Promise<T>;
}

export async function loadDashboardJson(): Promise<DashboardJson> {
  return fetchJson<DashboardJson>(PATHS.DASHBOARD_JSON);
}

export async function loadKpis(): Promise<KpiSummary[]> {
  const text = await fetchText(PATHS.KPI_SUMMARY);
  const [_header, ...rows] = parseCsv(text);
  return rows
    .filter((r) => r.length >= 5 && r[0])
    .map((r) => ({
      metric: r[0],
      value: Number(r[1]),
      unit: r[2],
      target: Number(r[3]),
      status: r[4],
    }));
}

export async function loadDemoCards(): Promise<DemoCard[]> {
  const text = await fetchText(PATHS.DEMO_CARDS);
  const [_header, ...rows] = parseCsv(text);
  return rows
    .filter((r) => r.length >= 4 && r[0])
    .map((r) => ({
      demo_id: r[0],
      name: r[1],
      status: r[2],
      primary_metric: r[3],
    }));
}

export async function loadModelComparison(): Promise<ModelComparisonRow[]> {
  const text = await fetchText(PATHS.MODEL_COMPARISON);
  const [_header, ...rows] = parseCsv(text);
  return rows
    .filter((r) => r.length >= 4 && r[0])
    .map((r) => ({
      model: r[0],
      accuracy_pct: r[1] ? Number(r[1]) : null,
      params_m: r[2] ? Number(r[2]) : null,
      inference_ms: r[3] ? Number(r[3]) : null,
      demo: r[4],
      r2_score: r[5] ? Number(r[5]) : null,
      params_k: r[6] ? Number(r[6]) : null,
      mape_pct: r[7] ? Number(r[7]) : null,
    }));
}

export async function loadApiUsage(): Promise<ApiUsageTrend[]> {
  const text = await fetchText(PATHS.API_USAGE_TRENDS);
  const [_header, ...rows] = parseCsv(text);
  return rows
    .filter((r) => r.length >= 2 && r[0])
    .map((r) => ({
      month: r[0],
      requests_k: Number(r[1]),
    }));
}

export async function loadTrainingTrends(): Promise<TrainingTrend[]> {
  // Try canonical first, then fallback duplicate-extension
  let text: string;
  try {
    text = await fetchText(PATHS.TRAINING_TRENDS_PRIMARY);
  } catch {
    text = await fetchText(PATHS.TRAINING_TRENDS_FALLBACK);
  }
  const [_header, ...rows] = parseCsv(text);
  return rows
    .filter((r) => r.length >= 3 && r[0])
    .map((r) => ({
      month: r[0],
      accuracy_pct: Number(r[1]),
      training_time_hrs: Number(r[2]),
    }));
}

// Markdown docs
export async function loadTechCodex(): Promise<string> {
  try {
    return await fetchText(PATHS.TECH_CODEX_PRIMARY);
  } catch {
    return await fetchText(PATHS.TECH_CODEX_FALLBACK);
  }
}

export async function loadExecutiveSummary(): Promise<string> {
  return fetchText(PATHS.EXEC_SUMMARY);
}
