import { loadDashboardJson } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { TrendingUp } from "lucide-react";

export default async function BusinessImpact() {
  const dashboard = await loadDashboardJson();
  const impact = dashboard.business_impact;
  
  // Get lives saved from flood demo
  const livesSaved = dashboard.demo_1_flood?.lives_potentially_saved || 0;

  const metrics = [
    {
      label: "Total Cost Savings",
      value: `$${(impact.total_cost_savings_usd / 1000000).toFixed(2)}M`,
      highlight: true,
    },
    {
      label: "Lives Potentially Saved",
      value: livesSaved.toLocaleString(),
      highlight: true,
    },
    {
      label: "Clients Served",
      value: impact.clients_served,
    },
    {
      label: "Countries Deployed",
      value: impact.countries_deployed,
    },
    {
      label: "Government Partnerships",
      value: impact.government_partnerships,
    },
    {
      label: "Patent Applications",
      value: impact.patent_applications,
    },
    {
      label: "Research Publications",
      value: impact.research_publications,
    },
    {
      label: "Team Size",
      value: impact.team_size,
    },
  ];

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="Business Impact" icon={TrendingUp} />
      </div>
      <div className="grid grid-cols-2 gap-4 phone:grid-cols-3 laptop:grid-cols-4">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className={`rounded-lg border border-border bg-card p-4 ${
              metric.highlight
                ? "bg-gradient-to-br from-blue-50 to-card dark:from-blue-950 dark:to-card"
                : ""
            }`}
          >
            <div className="mb-2 text-xs text-muted-foreground">
              {metric.label}
            </div>
            <div
              className={`text-xl font-bold ${
                metric.highlight ? "text-blue-600 dark:text-blue-400" : ""
              }`}
            >
              {metric.value}
            </div>
          </div>
        ))}
      </div>
    </Container>
  );
}
