import { useState } from "react";
import AnalyzeTab from "./components/AnalyzeTab";
import BatchTab from "./components/BatchTab";
import BenchmarkTab from "./components/BenchmarkTab";
import FeedbackTab from "./components/FeedbackTab";
import PlaygroundTab from "./components/PlaygroundTab";

const TABS = [
  { key: "analyze", label: "Analyze" },
  { key: "batch", label: "Batch" },
  { key: "benchmark", label: "Benchmark" },
  { key: "playground", label: "Robustness" },
  { key: "feedback", label: "Feedback" },
];

function savedTab() {
  try {
    return localStorage.getItem("tab") || "analyze";
  } catch {
    return "analyze";
  }
}

function App() {
  const [tab, setTab] = useState(savedTab);
  // Tabs are mounted on first visit and then kept, so results survive switching
  const [visited, setVisited] = useState(() => new Set([savedTab()]));

  const open = (key) => {
    setTab(key);
    setVisited((v) => new Set(v).add(key));
    try {
      localStorage.setItem("tab", key);
    } catch {
      /* storage unavailable: fine, the tab just isn't remembered */
    }
  };

  const content = {
    analyze: <AnalyzeTab />,
    batch: <BatchTab />,
    benchmark: <BenchmarkTab />,
    playground: <PlaygroundTab />,
    feedback: <FeedbackTab />,
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-8">
      <div className="max-w-screen-xl mx-auto">
        <h1 className="text-3xl sm:text-4xl font-bold text-left mb-2">AI Email Spam Classifier</h1>
        <p className="text-left text-gray-600">
          Regex + Statistical ML + fine-tuned DistilBERT + Zero-shot NLI, combined by a learned stacking ensemble
        </p>
        <p className="text-left text-sm text-gray-500 mb-4">
          Trained on short SMS-style messages (UCI SMS Spam Collection); long emails, HTML and phishing pages may be less accurate.
        </p>

        <nav role="tablist" aria-label="Sections" className="flex gap-1 overflow-x-auto overflow-y-hidden border-b mb-6">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => open(t.key)}
              className={`px-4 py-2 -mb-px border-b-2 whitespace-nowrap ${tab === t.key ? "border-blue-600 text-blue-700 font-semibold" : "border-transparent text-gray-600 hover:text-gray-900"}`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {TABS.map((t) =>
          visited.has(t.key) ? (
            <div key={t.key} role="tabpanel" hidden={tab !== t.key}>
              {content[t.key]}
            </div>
          ) : null
        )}
      </div>
    </div>
  );
}

export default App;
