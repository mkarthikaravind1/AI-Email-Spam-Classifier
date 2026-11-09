import { useState } from "react";
import axios from "axios";

// The three classifiers, in display order. `key` matches the API response.
const MODELS = [
  { key: "regex", name: "Regex Rules", bar: "bg-blue-600" },
  { key: "ml", name: "Statistical ML", detail: "Logistic Regression", bar: "bg-green-600" },
  { key: "llm", name: "Zero-shot NLI", detail: "BART-large-MNLI", bar: "bg-purple-600" },
];

const pct = (value) => `${value}%`;

// Backend URL: set VITE_API_URL in Frontend/.env (see .env.example) to change it
const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
const MAX_CHARS = 5000; // must match the backend's limit

// Turns an axios error into a message a user can act on
function errorMessage(error) {
  if (!error.response) {
    return `Can't reach the backend at ${API_URL}. Is it running?`;
  }
  const detail = error.response.data?.detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors (422): [{ msg: "Value error, ..." }, ...]
    return detail.map((d) => d.msg.replace(/^Value error, /, "")).join("; ");
  }
  return detail || `The backend returned an error (HTTP ${error.response.status}).`;
}

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
  const [error, setError] = useState("");

  const analyzeEmail = async () => {
    if (loading) return;
    if (!emailText.trim()) {
      setError("Please enter an email.");
      return;
    }

    // Clear the previous result so it is never shown next to a new input
    setResult(null);
    setError("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        email_text: emailText,
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 sm:p-8">
      <div className="max-w-screen-xl mx-auto">

        <h1 className="text-3xl sm:text-4xl font-bold text-left mb-2">
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
              maxLength={MAX_CHARS}
              onChange={(e) => setEmailText(e.target.value)}
            />
            <div className="flex items-center justify-between mt-2">
              <button
                onClick={analyzeEmail}
                disabled={loading}
                className="bg-blue-600 text-white px-3 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Analyzing..." : "Analyze Email"}
              </button>
              <span className={`text-sm ${emailText.length >= MAX_CHARS ? "text-red-600" : "text-gray-500"}`}>
                {emailText.length} / {MAX_CHARS}
              </span>
            </div>
            {error && (
              <p role="alert" className="mt-3 p-3 rounded-lg bg-red-100 text-red-700">
                {error}
              </p>
            )}
          </div>

          {/* Right: 3 Model Cards (only shown after result) */}
          {result && (
            <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-1 xl:grid-cols-3 gap-3">
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
          <div className="mt-6" aria-live="polite">
            <p>Analyzing... (the first request can take a while while the models warm up)</p>
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
                <div className="overflow-x-auto">
                <table className="w-full border-collapse min-w-[28rem]">
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
