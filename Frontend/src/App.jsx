import { useState } from "react";
import axios from "axios";

// The three classifiers, in display order. `key` matches the API response.
const MODELS = [
  { key: "regex", name: "Regex Rules", bar: "bg-blue-600" },
  { key: "ml", name: "Statistical ML", detail: "Logistic Regression", bar: "bg-green-600" },
  { key: "llm", name: "Zero-shot NLI", detail: "BART-large-MNLI", bar: "bg-purple-600" },
];

const pct = (value) => `${value}%`;

function ModelCard({ model, data, weight }) {
  return (
    <div className="bg-white p-3 rounded-xl shadow-md">
      <h2 className="font-bold text-lg">{model.name}</h2>
      {model.detail && <p className="text-xs text-gray-500 mb-1">{model.detail}</p>}

      {data.status === "error" ? (
        <p className="text-red-600 mt-2">Unavailable: {data.error}</p>
      ) : (
        <>
          <p>Prediction: <b>{data.prediction}</b></p>
          <p className="mt-1">Confidence: {pct(data.confidence)}</p>
          <p className="mt-1">Spam probability: {pct(data.spam_probability)}</p>
          <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
            <div className={`${model.bar} h-3 rounded-full`} style={{ width: pct(data.spam_probability) }}></div>
          </div>
          {weight !== undefined && (
            <p className="text-xs text-gray-500 mt-2">Vote weight: {pct(Math.round(weight * 100))}</p>
          )}
          {model.key === "regex" && (
            <div className="mt-1">
              <p>Score: {data.score}</p>
              <h3 className="font-semibold">Matched Rules</h3>
              {data.matched_rules.length === 0 && <p className="text-gray-500">None</p>}
              <ul>
                {data.matched_rules.map((rule) => (
                  <li key={rule}>✓ {rule}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function App() {
  const [emailText, setEmailText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeEmail = async () => {
    if (!emailText.trim()) {
      alert("Please enter an email.");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/analyze", {
        email_text: emailText,
      });

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Backend connection failed.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-screen-xl mx-auto">

        <h1 className="text-4xl font-bold text-left mb-2">
          AI Email Spam Classifier
        </h1>

        <p className="text-left text-gray-600">
          Regex + Statistical ML + Zero-shot NLI, combined by a weighted vote
        </p>
        <p className="text-left text-sm text-gray-500 mb-6">
          Trained on short SMS-style messages (UCI SMS Spam Collection); long emails,
          HTML and phishing pages may be less accurate.
        </p>

        {/* Input Section */}
        <div className="grid md:grid-cols-2 gap-6 items-start">

          {/* Left: Input */}
          <div>
            <textarea
              className="w-full border rounded-lg p-4 h-40 bg-white"
              placeholder="Paste email text here..."
              value={emailText}
              onChange={(e) => setEmailText(e.target.value)}
            />
            <button
              onClick={analyzeEmail}
              className="mt-2 bg-blue-600 text-white px-3 py-2 rounded-lg"
            >
              Analyze Email
            </button>
          </div>

          {/* Right: 3 Model Cards (only shown after result) */}
          {result && (
            <div className="grid grid-cols-3 gap-3">
              {MODELS.map((model) => (
                <ModelCard
                  key={model.key}
                  model={model}
                  data={result[model.key]}
                  weight={result.weights?.[model.key]}
                />
              ))}
            </div>
          )}

        </div>

        {/* Loading Indicator — outside result block so it shows during fetch */}
        {loading && (
          <div className="mt-6">
            <p>Analyzing...</p>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="grid md:grid-cols-2 gap-6 mt-8">

            {/* Left Column */}
            <div className="space-y-6">

              {/* Model Comparison Table */}
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h2 className="text-2xl font-bold mb-4">Model Comparison</h2>
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Model</th>
                      <th className="text-left p-2">Prediction</th>
                      <th className="text-left p-2">Confidence</th>
                      <th className="text-left p-2">Spam prob.</th>
                      <th className="text-left p-2">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {MODELS.map((model) => {
                      const data = result[model.key];
                      const weight = result.weights?.[model.key];
                      return data.status === "error" ? (
                        <tr key={model.key} className="border-b text-gray-400">
                          <td className="p-2">{model.name}</td>
                          <td className="p-2" colSpan={4}>unavailable</td>
                        </tr>
                      ) : (
                        <tr key={model.key} className="border-b">
                          <td className="p-2">{model.name}</td>
                          <td className="p-2">{data.prediction}</td>
                          <td className="p-2">{pct(data.confidence)}</td>
                          <td className="p-2">{pct(data.spam_probability)}</td>
                          <td className="p-2">{weight !== undefined ? pct(Math.round(weight * 100)) : "-"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Final Verdict — bg color reflects spam/ham */}
              <div
                className={`p-6 rounded-xl ${
                  result.final_verdict === "spam"
                    ? "bg-red-100"
                    : "bg-green-100"
                }`}
              >
                <h2 className="text-2xl font-bold">Final Verdict</h2>
                <div className="flex justify-between items-center mt-3">
                  <p className="font-semibold">Verdict</p>
                  <span
                    className={`px-4 py-2 rounded-full font-bold text-white ${
                      result.final_verdict === "spam"
                        ? "bg-red-600"
                        : "bg-green-600"
                    }`}
                  >
                    {result.final_verdict.toUpperCase()}
                  </span>
                </div>
                <p className="mt-2">Weighted spam probability: {pct(result.spam_probability)}</p>
                <p>Models agreeing with verdict: {result.agreement}</p>
                <p className="mt-1">Processing Time: {result.processing_time} sec</p>
              </div>

            </div>

            {/* Right Column — Classification Pipeline (the 3 models run in parallel) */}
            <div className="bg-white p-6 rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold mb-4">Classification Pipeline</h2>
              <div className="text-center space-y-2">
                <p className="font-semibold">Email Input</p>
                <p>↓ ↓ ↓</p>
                <div className="grid grid-cols-3 gap-2">
                  {MODELS.map((model) => (
                    <div key={model.key} className="border rounded-lg p-2 text-sm">
                      {model.name}
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-500">run in parallel</p>
                <p>↓ ↓ ↓</p>
                <p>
                  Weighted average of spam probabilities
                  <span className="block text-xs text-gray-500">
                    weights = each model&apos;s measured F1 on validation data
                  </span>
                </p>
                <p>↓</p>
                <p className="font-bold">Final Verdict (spam if ≥ 50%)</p>
              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}

export default App;
