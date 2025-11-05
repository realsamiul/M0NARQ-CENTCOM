import { loadDemoCards } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { Layers } from "lucide-react";

export default async function DemoCards() {
  const cards = await loadDemoCards();

  return (
    <Container className="py-4">
      <div className="mb-4">
        <ChartTitle title="Demo Applications" icon={Layers} />
      </div>
      <div className="grid grid-cols-1 gap-4 phone:grid-cols-2 laptop:grid-cols-3">
        {cards.map((card) => (
          <div
            key={card.demo_id}
            className="rounded-lg border border-border bg-card p-4 transition-all hover:shadow-lg"
          >
            <div className="mb-3 flex items-start justify-between">
              <h3 className="text-base font-semibold">{card.name}</h3>
              <span
                className={`rounded-full px-2 py-1 text-xs font-medium ${
                  card.status === "PRODUCTION"
                    ? "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300"
                    : "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300"
                }`}
              >
                {card.status}
              </span>
            </div>
            <div className="text-sm text-muted-foreground">
              <span className="font-medium">Primary Metric:</span>{" "}
              <span className="text-foreground">{card.primary_metric}</span>
            </div>
          </div>
        ))}
      </div>
    </Container>
  );
}
