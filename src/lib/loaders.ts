"use server";

import { readFile } from "fs/promises";
import { join } from "path";
import type {
  ApiUsageTrend,
  DashboardJson,
  DemoCard,
  KpiSummary,
  ModelComparisonRow,
  TrainingTrend,
} from "@/types/dashboard";

// Lightweight CSV parser (no extra deps): assumes comma-separated values
function parseCsv(raw: string): string[][] {
  return raw
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split(","));
}

// Get the public directory path
function getPublicPath(filename: string): string {
  return join(process.cwd(), "public", "data", filename);
}

// Read file directly from filesystem (works in build + runtime)
async function readFileText(filename: string): Promise<string> {
  try {
    return await readFile(getPublicPath(filename), "utf-8");
  } catch (error) {
    console.error(`Failed to read ${filename}:`, error);
    throw new Error(`Failed to load ${filename}`);
  }
}

// Parse JSON file
async function readFileJson<T>(filename: string): Promise<T> {
  const text = await readFileText(filename);
  return JSON.parse(text) as T;
}

export async function loadDashboardJson(): Promise<DashboardJson> {
  return readFileJson<DashboardJson>("DASHBOARD_DATA_COMPLETE.json");
}

export async function loadKpis(): Promise<KpiSummary[]> {
  const text = await readFileText("KPI_SUMMARY.csv");
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
  const text = await readFileText("DEMO_CARDS.csv");
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
  const text = await readFileText("MODEL_COMPARISON.csv");
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
  const text = await readFileText("API_USAGE_TRENDS.csv");
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
    text = await readFileText("TRAINING_TRENDS.csv");
  } catch {
    try {
      text = await readFileText("TRAINING_TRENDS.csv.csv");
    } catch {
      throw new Error("Could not load TRAINING_TRENDS data");
    }
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
    return await readFileText("TECH-CODEX-DASHBOARD.md");
  } catch {
    try {
      return await readFileText("TECH-CODEX-DASHBOARD.md.md");
    } catch {
      return "# Technical Documentation\n\nDocumentation not available.";
    }
  }
}

export async function loadExecutiveSummary(): Promise<string> {
  try {
    return await readFileText("executive_summary.md");
  } catch {
    return "# Executive Summary\n\nSummary not available.";
  }
}
