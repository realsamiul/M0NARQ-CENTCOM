"use client";

import { VChart } from "@visactor/react-vchart";
import type { IAreaChartSpec } from "@visactor/vchart";
import type { ApiUsageTrend } from "@/types/dashboard";

interface ApiUsageChartProps {
  data: ApiUsageTrend[];
}

export default function ApiUsageChart({ data }: ApiUsageChartProps) {
  const spec: IAreaChartSpec = {
    type: "area",
    data: [
      {
        id: "apiUsage",
        values: data.map((d) => ({
          month: d.month,
          requests: d.requests_k,
        })),
      },
    ],
    xField: "month",
    yField: "requests",
    point: {
      visible: true,
      style: {
        fill: "#3161F8",
        size: 6,
      },
    },
    line: {
      style: {
        stroke: "#3161F8",
        lineWidth: 2,
      },
    },
    area: {
      style: {
        fill: "#3161F8",
        fillOpacity: 0.1,
      },
    },
    axes: [
      {
        orient: "left",
        title: {
          visible: true,
          text: "Requests (thousands)",
        },
      },
      {
        orient: "bottom",
        title: {
          visible: true,
          text: "Month",
        },
      },
    ],
    tooltip: {
      mark: {
        visible: true,
      },
      dimension: {
        visible: true,
      },
    },
  };

  return (
    <div className="h-96 w-full">
      <VChart spec={spec} />
    </div>
  );
}
