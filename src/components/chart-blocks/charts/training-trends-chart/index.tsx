import { loadTrainingTrends } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Activity } from "lucide-react";
import TrainingTrendsChart from "./chart";

export default async function TrainingTrends() {
  const data = await loadTrainingTrends();

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="Model Training Trends" icon={Activity} />
      </div>
      <TrainingTrendsChart data={data} />
    </Container>
  );
}
