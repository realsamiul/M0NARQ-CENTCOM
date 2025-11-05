import { loadTechCodex, loadExecutiveSummary } from "@/lib/loaders";
import Container from "@/components/container";
import ChartTitle from "../../components/chart-title";
import { BookOpen, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";

export default async function DocsSections() {
  const [codex, exec] = await Promise.all([
    loadTechCodex(),
    loadExecutiveSummary(),
  ]);

  return (
    <>
      <Container className="py-4">
        <div className="mb-4">
          <ChartTitle title="Executive Summary" icon={FileText} />
        </div>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown>{exec}</ReactMarkdown>
        </div>
      </Container>
      <Container className="py-4">
        <div className="mb-4">
          <ChartTitle title="Technical Codex" icon={BookOpen} />
        </div>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown>{codex}</ReactMarkdown>
        </div>
      </Container>
    </>
  );
}
