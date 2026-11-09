import { useState } from "react";
import { api, errorMessage, ENSEMBLE_COLOR, MAX_CHARS, MODEL_BY_KEY, pct } from "../api";

function Prob({ value }) {
  const spam = value >= 50;
  return (
    <span className={`tabular-nums ${spam ? "text-red-700 font-semibold" : "text-gray-700"}`}>
      {pct(value, 0)}
    </span>
  );
}

export default function PlaygroundTab() {
  const [text, setText] = useState("Congratulations! You won a free cash prize. Claim it now by replying WIN");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (loading) return;
    if (!text.trim()) return setError("Please enter a spam message to attack.");
    setLoading(true);
    setError("");
    try {
      const response = await api.post("/robustness", { email_text: text });
      setData(response.data);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const models = data ? Object.keys(data.variants[0].raw.models) : [];

  return (
    <div className="space-y-6">
      <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-bold">Robustness playground</h2>
        <p className="text-sm text-gray-600 mb-3">
          Spammers disguise words so filters miss them. Paste a spam message; it is rewritten with six common tricks and
          every fast model scores each version, first as-is and then after de-obfuscation. A red number = still caught.
        </p>
        <textarea className="w-full border rounded-lg p-3 h-24" value={text} maxLength={MAX_CHARS}
          aria-label="Spam message to attack" onChange={(e) => setText(e.target.value)} />
        <button onClick={run} disabled={loading}
          className="mt-2 bg-blue-600 text-white px-4 py-2 rounded-lg disabled:opacity-50">
          {loading ? "Attacking..." : "Run attacks"}
        </button>
        {error && <p role="alert" className="mt-3 p-3 rounded-lg bg-red-100 text-red-700">{error}</p>}
      </section>

      {data && (
        <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[48rem]">
              <thead>
                <tr className="border-b text-left">
                  <th className="p-2">Attack and disguised text</th>
                  {models.map((m) => (
                    <th key={m} className="p-2 whitespace-nowrap">
                      <span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: MODEL_BY_KEY[m].color }} />
                      {MODEL_BY_KEY[m].name}
                    </th>
                  ))}
                  <th className="p-2 whitespace-nowrap">
                    <span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: ENSEMBLE_COLOR }} />
                    Stacking
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.variants.map((v) => (
                  <tr key={v.name} className="border-b align-top">
                    <td className="p-2 max-w-[22rem]">
                      <p className="font-semibold">{v.description}</p>
                      <code className="block text-xs text-gray-600 break-all mt-1">{v.text}</code>
                    </td>
                    {models.map((m) => (
                      <td key={m} className="p-2 whitespace-nowrap">
                        <Prob value={v.raw.models[m]} /> → <Prob value={v.normalized.models[m]} />
                      </td>
                    ))}
                    <td className="p-2 whitespace-nowrap">
                      <Prob value={v.raw.ensemble} /> → <Prob value={v.normalized.ensemble} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-500 mt-3">
            Each cell: spam probability without → with de-obfuscation. The slow zero-shot model is left out to keep this fast.
          </p>
        </section>
      )}
    </div>
  );
}
