import { loadKpis } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Activity } from "lucide-react";

function getStatusColor(status: string): string {
  switch (status) {
    case "EXCEEDED":
      return "bg-green-500";
    case "ON_TRACK":
      return "bg-blue-500";
    case "AT_RISK":
      return "bg-yellow-500";
    case "BEHIND":
      return "bg-red-500";
    default:
      return "bg-gray-500";
  }
}

function getStatusTextColor(status: string): string {
  switch (status) {
    case "EXCEEDED":
      return "text-green-600";
    case "ON_TRACK":
      return "text-blue-600";
    case "AT_RISK":
      return "text-yellow-600";
    case "BEHIND":
      return "text-red-600";
    default:
      return "text-gray-600";
  }
}

export default async function KpiGrid() {
  const kpis = await loadKpis();

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="KPI Summary" icon={Activity} />
      </div>
      <div className="grid grid-cols-1 gap-4 phone:grid-cols-2 laptop:grid-cols-4">
        {kpis.map((kpi) => {
          const progress = Math.min(100, (kpi.value / kpi.target) * 100);
          return (
            <div
              key={kpi.metric}
              className="rounded-lg border border-border bg-card p-4"
            >
              <div className="mb-2 text-sm text-muted-foreground">
                {kpi.metric}
              </div>
              <div className="mb-1 text-2xl font-bold">
                {kpi.value}
                <span className="ml-1 text-sm font-normal">{kpi.unit}</span>
              </div>
              <div className="mb-2 text-xs text-muted-foreground">
                Target: {kpi.target} {kpi.unit}
              </div>
              <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full ${getStatusColor(kpi.status)}`}
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div
                className={`text-xs font-medium ${getStatusTextColor(kpi.status)}`}
              >
                {kpi.status}
              </div>
            </div>
          );
        })}
      </div>
    </Container>
  );
}
