import { loadDashboardJson } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Server } from "lucide-react";

export default async function SystemOverview() {
  const dashboard = await loadDashboardJson();
  const sys = dashboard.system_overview;

  const metrics = [
    {
      label: "Total Code Lines",
      value: sys.total_code_lines.toLocaleString(),
    },
    {
      label: "Total Datasets",
      value: sys.total_datasets,
    },
    {
      label: "Satellite Data Sources",
      value: sys.satellite_data_sources,
    },
    {
      label: "ML Models Deployed",
      value: sys.ml_models_deployed,
    },
    {
      label: "Cloud Storage",
      value: `${sys.cloud_storage_tb} TB`,
    },
    {
      label: "API Calls (Daily)",
      value: sys.api_calls_daily.toLocaleString(),
    },
  ];

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="System Overview" icon={Server} />
      </div>
      <div className="grid grid-cols-2 gap-4 phone:grid-cols-3 laptop:grid-cols-6">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="rounded-lg border border-border bg-card p-4"
          >
            <div className="mb-1 text-xs text-muted-foreground">
              {metric.label}
            </div>
            <div className="text-xl font-bold">{metric.value}</div>
          </div>
        ))}
      </div>
    </Container>
  );
}
