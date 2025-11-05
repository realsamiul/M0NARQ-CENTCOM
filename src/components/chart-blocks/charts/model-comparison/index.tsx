import { loadModelComparison } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Database } from "lucide-react";
import ModelTable from "./table";

export default async function ModelComparison() {
  const models = await loadModelComparison();

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="Model Comparison" icon={Database} />
      </div>
      <ModelTable models={models} />
    </Container>
  );
}
