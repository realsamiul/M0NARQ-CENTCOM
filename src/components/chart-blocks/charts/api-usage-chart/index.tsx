import { loadApiUsage } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { TrendingUp } from "lucide-react";
import ApiUsageChart from "./chart";

export default async function ApiUsage() {
  const data = await loadApiUsage();

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="API Usage Trends" icon={TrendingUp} />
      </div>
      <ApiUsageChart data={data} />
    </Container>
  );
}
