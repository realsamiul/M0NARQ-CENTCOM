# VisActor Next.js Dashboard Template

A modern dashboard template built with [VisActor](https://visactor.io/) and Next.js, featuring a beautiful UI and rich data visualization components.

[Live Demo](https://visactor-next-template.vercel.app/)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?demo-description=A%20modern%20dashboard%20with%20VisActor%20charts%2C%20dark%20mode%2C%20and%20data%20visualization%20for%20seamless%20analytics.&demo-image=%2F%2Fimages.ctfassets.net%2Fe5382hct74si%2F646TLqKGSTOnp1CD1IUqoM%2Fa119adac1f5a844f9d42f807ddc075f5%2Fthumbnail.png&demo-title=VisActor%20Next.js%20Template&demo-url=https%3A%2F%2Fvisactor-next-template.vercel.app%2F&from=templates&project-name=VisActor%20Next.js%20Template&repository-name=visactor-nextjs-template&repository-url=https%3A%2F%2Fgithub.com%2Fmengxi-ream%2Fvisactor-next-template&skippable-integrations=1)

## Features

- 📊 **Rich Visualizations** - Powered by VisActor, including bar charts, gauge charts, circle packing charts, and more
- 🌗 **Dark Mode** - Seamless dark/light mode switching with system preference support
- 📱 **Responsive Design** - Fully responsive layout that works on all devices
- 🎨 **Beautiful UI** - Modern and clean interface built with Tailwind CSS
- ⚡️ **Next.js 15** - Built on the latest Next.js features and best practices
- 🔄 **State Management** - Efficient state management with Jotai
- 📦 **Component Library** - Includes Shadcn components styled with Tailwind

## Tech Stack

- [Next.js](https://nextjs.org/) - React framework
- [VisActor](https://visactor.io/) - Visualization library
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Shadcn](https://ui.shadcn.com/) - UI components
- [Jotai](https://jotai.org/) - State management
- [TypeScript](https://www.typescriptlang.org/) - Type safety

## Quick Start

You can deploy this template to Vercel by clicking the button above, or clone this repository and run it locally.

[Github Repo](https://github.com/mengxi-ream/visactor-next-template)

1. Clone this repository

```bash
git clone https://github.com/mengxi-ream/visactor-next-template
```

2. Install dependencies

```bash
pnpm install
```

3. Run the development server

```bash
pnpm dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```bash
src/
├── app/ # App router pages
├── components/ # React components
│ ├── chart-blocks/ # Chart components
│ ├── nav/ # Navigation components
│ └── ui/ # UI components
├── config/ # Configuration files
├── data/ # Sample data
├── hooks/ # Custom hooks
├── lib/ # Utility functions
├── style/ # Global style
└── types/ # TypeScript types
```

## Charts

This template includes several chart examples:

- Average Tickets Created (Bar Chart)
- Ticket by Channels (Gauge Chart)
- Conversions (Circle Packing Chart)
- Customer Satisfaction (Linear Progress)
- Metrics Overview

## Using YOUR Data

This dashboard is designed to work with your own data. The current implementation uses real production data from the M0NARQ Decision OS platform, demonstrating how to integrate custom datasets.

### Data Location

All data files should be placed in the `public/data/` directory. The dashboard uses server-side loaders to fetch and parse these files at runtime.

### Supported Data Formats

The dashboard currently supports:
- **JSON files** - For complex structured data (e.g., `DASHBOARD_DATA_COMPLETE.json`)
- **CSV files** - For tabular data (e.g., `KPI_SUMMARY.csv`, `MODEL_COMPARISON.csv`)
- **Markdown files** - For documentation and narrative content (e.g., `executive_summary.md`)

### Expected Data Schemas

#### KPI Summary (`KPI_SUMMARY.csv`)
```csv
metric,value,unit,target,status
Overall System Accuracy,89.2,%,90.0,ON_TRACK
Processing Speed,99.5,% faster,95.0,EXCEEDED
```

#### Demo Cards (`DEMO_CARDS.csv`)
```csv
demo_id,name,status,primary_metric
demo_1_flood,HAWKEYE Flood Intelligence,PRODUCTION,91.0% Accuracy
demo_2_crop,HAWKEYE Crop Intelligence,PRODUCTION,In Development
```

#### Model Comparison (`MODEL_COMPARISON.csv`)
```csv
model,accuracy_pct,params_m,inference_ms,demo,r2_score,params_k,mape_pct
SegFormer-B2,91.0,27.4,45,Flood,,,
XGBoost,,,3,Freight,0.7,1200.0,
```

#### API Usage Trends (`API_USAGE_TRENDS.csv`)
```csv
month,requests_k
2024-07,280
2024-08,310
```

#### Training Trends (`TRAINING_TRENDS.csv`)
```csv
month,accuracy_pct,training_time_hrs
2024-07,85,18
2024-08,87,16
```

#### Dashboard JSON (`DASHBOARD_DATA_COMPLETE.json`)
```json
{
  "metadata": {...},
  "system_overview": {
    "total_code_lines": 87326,
    "total_datasets": 52,
    "satellite_data_sources": 4,
    "ml_models_deployed": 11,
    "cloud_storage_tb": 1.2,
    "api_calls_daily": 15000
  },
  "performance_trends": {
    "model_training": [...],
    "api_usage": [...]
  }
}
```

### Adding a New Dataset and Chart

To integrate a new dataset:

1. **Add your data file** to `public/data/`

2. **Define TypeScript types** in `src/types/dashboard.ts`:
```typescript
export interface MyDataType {
  field1: string;
  field2: number;
}
```

3. **Add path constant** in `src/lib/data-sources.ts`:
```typescript
export const PATHS = {
  // ... existing paths
  MY_DATA: `${DATA_BASE}/my-data.csv`,
} as const;
```

4. **Create a loader** in `src/lib/loaders.ts`:
```typescript
export async function loadMyData(): Promise<MyDataType[]> {
  const text = await fetchText(PATHS.MY_DATA);
  const [_header, ...rows] = parseCsv(text);
  return rows.map((r) => ({
    field1: r[0],
    field2: Number(r[1]),
  }));
}
```

5. **Create a component** in `src/components/chart-blocks/charts/my-chart/`:
```typescript
import { loadMyData } from "@/lib/loaders";

export default async function MyChart() {
  const data = await loadMyData();
  
  return (
    <Container className="py-4">
      {/* Your chart/visualization here */}
    </Container>
  );
}
```

6. **Export and use** in your page:
```typescript
// In src/components/chart-blocks/index.tsx
export { default as MyChart } from "./charts/my-chart";

// In src/app/(dashboard)/page.tsx
import { MyChart } from "@/components/chart-blocks";
```

### Notes

- All loaders use `"use server"` directive to ensure server-side execution
- CSV files use a simple comma-based parser - ensure your data doesn't have commas in values or use proper escaping
- Files with duplicate extensions (e.g., `.csv.csv`, `.md.md`) are handled automatically with fallback paths
- Data is fetched with `cache: "no-store"` to ensure fresh data on each request

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [VisActor](https://visactor.io/) - For the amazing visualization library
- [Vercel](https://vercel.com) - For the incredible deployment platform
- [Next.js](https://nextjs.org/) - For the awesome React framework
