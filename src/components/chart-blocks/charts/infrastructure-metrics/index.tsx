import { loadDashboardJson } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Cloud } from "lucide-react";

export default async function InfrastructureMetrics() {
  const dashboard = await loadDashboardJson();
  const infra = dashboard.infrastructure_metrics;

  const metrics = [
    {
      label: "Cloud Provider",
      value: infra.cloud_provider,
    },
    {
      label: "GPU Hours/Month",
      value: infra.gpu_hours_monthly.toLocaleString(),
    },
    {
      label: "Storage",
      value: `${infra.storage_gb} GB`,
    },
    {
      label: "API Requests/Month",
      value: infra.api_requests_monthly.toLocaleString(),
    },
    {
      label: "Uptime",
      value: `${infra.uptime_pct}%`,
    },
    {
      label: "Avg Response Time",
      value: `${infra.avg_response_time_ms}ms`,
    },
    {
      label: "Peak Concurrent Users",
      value: infra.peak_concurrent_users,
    },
    {
      label: "Data Processed/Month",
      value: `${infra.data_processed_tb_monthly} TB`,
    },
  ];

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="Infrastructure Metrics" icon={Cloud} />
      </div>
      <div className="grid grid-cols-2 gap-4 phone:grid-cols-3 laptop:grid-cols-4">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="rounded-lg border border-border bg-card p-4"
          >
            <div className="mb-2 text-xs text-muted-foreground">
              {metric.label}
            </div>
            <div className="text-xl font-bold">{metric.value}</div>
          </div>
        ))}
      </div>
    </Container>
  );
}
