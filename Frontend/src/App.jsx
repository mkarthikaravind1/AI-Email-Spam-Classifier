import { useState } from "react";
import axios from "axios";

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

        <p className="text-left text-gray-600 mb-6">
          Regex + Statistical ML + LLM
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

              <div className="bg-white p-3 rounded-xl shadow-md">
                <h2 className="font-bold text-lg mb-1">Regex Classifier</h2>
                <p>Prediction: <b>{result.regex.prediction}</b></p>
                <p>Score: {result.regex.score}</p>
                <p className="mt-1">Confidence: {result.regex.confidence}%</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                  <div className="bg-blue-600 h-3 rounded-full" style={{ width: `${result.regex.confidence}%` }}></div>
                </div>
                <div className="mt-1">
                  <h3 className="font-semibold">Matched Rules</h3>
                  <ul>
                    {result.regex.matched_rules.map((rule) => (
                      <li key={rule}>✓ {rule}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="bg-white p-3 rounded-xl shadow-md">
                <h2 className="font-bold text-lg mb-1">Statistical ML</h2>
                <p>Prediction: <b>{result.ml.prediction}</b></p>
                <p className="mt-1">Confidence: {result.ml.confidence}%</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                  <div className="bg-green-600 h-3 rounded-full" style={{ width: `${result.ml.confidence}%` }}></div>
                </div>
              </div>

              <div className="bg-white p-3 rounded-xl shadow-md">
                <h2 className="font-bold text-lg mb-1">LLM</h2>
                <p>Prediction: <b>{result.llm.prediction}</b></p>
                <p className="mt-1">Confidence: {result.llm.confidence}%</p>
                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                  <div className="bg-purple-600 h-3 rounded-full" style={{ width: `${result.llm.confidence}%` }}></div>
                </div>
              </div>

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
          <>
            

            {/* Bottom 2-Column Section */}
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
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b">
                        <td className="p-2">Regex</td>
                        <td className="p-2">{result.regex.prediction}</td>
                        <td className="p-2">{result.regex.confidence}%</td>
                      </tr>
                      <tr className="border-b">
                        <td className="p-2">Statistical ML</td>
                        <td className="p-2">{result.ml.prediction}</td>
                        <td className="p-2">{result.ml.confidence}%</td>
                      </tr>
                      <tr>
                        <td className="p-2">BART LLM</td>
                        <td className="p-2">{result.llm.prediction}</td>
                        <td className="p-2">{result.llm.confidence}%</td>
                      </tr>
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
                  <p>Agreement: {result.agreement}</p>
                  <p className="mt-1">Processing Time: {result.processing_time} sec</p>
                </div>

              </div>

              {/* Right Column — Classification Pipeline */}
              <div className="bg-white p-6 rounded-xl shadow-lg">
                <h2 className="text-2xl font-bold mb-4">Classification Pipeline</h2>
                <div className="text-center space-y-2">
                  <p>Email Input</p>
                  <p>↓</p>
                  <p>Regex Classifier</p>
                  <p>↓</p>
                  <p>Statistical ML (Naive Bayes)</p>
                  <p>↓</p>
                  <p>BART LLM</p>
                  <p>↓</p>
                  <p>Majority Voting</p>
                  <p>↓</p>
                  <p className="font-bold">Final Verdict</p>
                </div>
              </div>

            </div>
          </>
        )}

      </div>
    </div>
  );
}

export default App;