import {
  AverageTicketsCreated,
  Conversions,
  CustomerSatisfication,
  Metrics,
  TicketByChannels,
  KpiGrid,
  DemoCards,
  SystemOverview,
  ModelComparison,
  ApiUsage,
  TrainingTrends,
  DocsSections,
} from "@/components/chart-blocks";
import Container from "@/components/container";

export default function Home() {
  return (
    <div>
      {/* System Overview from OURDATA */}
      <SystemOverview />
      
      {/* KPI Grid from OURDATA */}
      <KpiGrid />
      
      {/* Demo Cards from OURDATA */}
      <DemoCards />

      {/* Charts section */}
      <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-2 laptop:divide-x laptop:divide-y-0 laptop:divide-border">
        <Container className="py-4 laptop:col-span-1">
          <ApiUsage />
        </Container>
        <Container className="py-4 laptop:col-span-1">
          <TrainingTrends />
        </Container>
      </div>

      {/* Model Comparison Table */}
      <ModelComparison />

      {/* Documentation Sections */}
      <DocsSections />

      {/* Original demo components (keep for reference) */}
      <div className="border-t-4 border-border pt-4">
        <Metrics />
        <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-3 laptop:divide-x laptop:divide-y-0 laptop:divide-border">
          <Container className="py-4 laptop:col-span-2">
            <AverageTicketsCreated />
          </Container>
          <Container className="py-4 laptop:col-span-1">
            <Conversions />
          </Container>
        </div>
        <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-2 laptop:divide-x laptop:divide-y-0 laptop:divide-border">
          <Container className="py-4 laptop:col-span-1">
            <TicketByChannels />
          </Container>
          <Container className="py-4 laptop:col-span-1">
            <CustomerSatisfication />
          </Container>
        </div>
      </div>
    </div>
  );
}
