"use client";

import { useState } from "react";
import type { ModelComparisonRow } from "@/types/dashboard";

type SortKey = keyof ModelComparisonRow;
type SortOrder = "asc" | "desc";

interface ModelTableProps {
  models: ModelComparisonRow[];
}

export default function ModelTable({ models }: ModelTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("model");
  const [sortOrder, setSortOrder] = useState<SortOrder>("asc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const sortedModels = [...models].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];

    if (aVal === null || aVal === undefined) return 1;
    if (bVal === null || bVal === undefined) return -1;

    if (typeof aVal === "string" && typeof bVal === "string") {
      return sortOrder === "asc"
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }

    if (typeof aVal === "number" && typeof bVal === "number") {
      return sortOrder === "asc" ? aVal - bVal : bVal - aVal;
    }

    return 0;
  });

  const formatValue = (val: number | null | undefined, suffix = "") => {
    if (val === null || val === undefined) return "-";
    return `${val}${suffix}`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-border">
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("model")}
            >
              Model {sortKey === "model" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("accuracy_pct")}
            >
              Accuracy{" "}
              {sortKey === "accuracy_pct" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("params_m")}
            >
              Params (M){" "}
              {sortKey === "params_m" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("inference_ms")}
            >
              Inference{" "}
              {sortKey === "inference_ms" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("demo")}
            >
              Demo {sortKey === "demo" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("r2_score")}
            >
              R² Score{" "}
              {sortKey === "r2_score" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
            <th
              className="cursor-pointer p-3 text-left font-medium hover:bg-muted"
              onClick={() => handleSort("mape_pct")}
            >
              MAPE{" "}
              {sortKey === "mape_pct" && (sortOrder === "asc" ? "↑" : "↓")}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedModels.map((model, idx) => (
            <tr
              key={idx}
              className="border-b border-border hover:bg-muted/50"
            >
              <td className="p-3 font-medium">{model.model}</td>
              <td className="p-3">{formatValue(model.accuracy_pct, "%")}</td>
              <td className="p-3">{formatValue(model.params_m)}</td>
              <td className="p-3">{formatValue(model.inference_ms, "ms")}</td>
              <td className="p-3">{model.demo}</td>
              <td className="p-3">{formatValue(model.r2_score)}</td>
              <td className="p-3">{formatValue(model.mape_pct, "%")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
