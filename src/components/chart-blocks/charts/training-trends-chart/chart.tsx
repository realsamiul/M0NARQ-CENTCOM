"use client";

import { VChart } from "@visactor/react-vchart";
import type { ICommonChartSpec } from "@visactor/vchart";
import type { TrainingTrend } from "@/types/dashboard";

interface TrainingTrendsChartProps {
  data: TrainingTrend[];
}

export default function TrainingTrendsChart({
  data,
}: TrainingTrendsChartProps) {
  const spec: ICommonChartSpec = {
    type: "common",
    data: [
      {
        id: "accuracy",
        values: data.map((d) => ({
          month: d.month,
          value: d.accuracy_pct,
          type: "Accuracy %",
        })),
      },
      {
        id: "time",
        values: data.map((d) => ({
          month: d.month,
          value: d.training_time_hrs,
          type: "Training Time (hrs)",
        })),
      },
    ],
    series: [
      {
        type: "line",
        dataId: "accuracy",
        xField: "month",
        yField: "value",
        line: {
          style: {
            stroke: "#3161F8",
            lineWidth: 2,
          },
        },
        point: {
          visible: true,
          style: {
            fill: "#3161F8",
            size: 6,
          },
        },
      },
      {
        type: "line",
        dataId: "time",
        xField: "month",
        yField: "value",
        line: {
          style: {
            stroke: "#60C2FB",
            lineWidth: 2,
          },
        },
        point: {
          visible: true,
          style: {
            fill: "#60C2FB",
            size: 6,
          },
        },
      },
    ],
    axes: [
      {
        orient: "left",
        seriesIndex: [0],
        title: {
          visible: true,
          text: "Accuracy (%)",
        },
        min: 0,
        max: 100,
      },
      {
        orient: "right",
        seriesIndex: [1],
        title: {
          visible: true,
          text: "Training Time (hours)",
        },
        min: 0,
      },
      {
        orient: "bottom",
        title: {
          visible: true,
          text: "Month",
        },
      },
    ],
    legends: {
      visible: true,
    },
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
