import { useMemo, useState } from "react";
import { api, errorMessage, MODELS, pct } from "../api";

const PAGE = 100;

function Stat({ label, value, sub }) {
  return (
    <div className="border rounded-lg p-3 text-center bg-white">
      <p className="text-2xl font-bold tabular-nums">{value}</p>
      <p className="text-xs text-gray-600">{label}</p>
      {sub && <p className="text-xs text-gray-500">{sub}</p>}
    </div>
  );
}

function csvCell(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadCsv(data, sourceName) {
  const modelKeys = MODELS.map((m) => m.key);
  const header = ["row", "text", "verdict", "spam_probability", ...modelKeys.map((k) => `${k}_spam_probability`),
    "used_zero_shot", "link_risk", ...(data.label_column ? ["label", "correct"] : [])];
  const lines = data.rows.map((r) => [
    r.row, r.text, r.verdict, r.spam_probability, ...modelKeys.map((k) => r.models[k] ?? ""),
    r.escalated, r.link_risk, ...(data.label_column ? [r.label, r.correct] : []),
  ].map(csvCell).join(","));
  const blob = new Blob(["﻿" + [header.join(","), ...lines].join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = sourceName.replace(/\.csv$/i, "") + "_results.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export default function BatchTab() {
  const [file, setFile] = useState(null);
  const [column, setColumn] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [shown, setShown] = useState(PAGE);

  const run = async (chosenFile, chosenColumn) => {
    if (!chosenFile || loading) return;
    setLoading(true);
    setError("");
    setData(null);
    const form = new FormData();
    form.append("file", chosenFile);
    if (chosenColumn) form.append("column", chosenColumn);
    try {
      const response = await api.post("/analyze-batch", form);
      setData(response.data);
      setColumn(response.data.text_column);
      setFilter("all");
      setShown(PAGE);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const onFile = (event) => {
    const chosen = event.target.files?.[0];
    event.target.value = "";
    if (!chosen) return;
    setFile(chosen);
    setColumn("");
    run(chosen, "");
  };

  const rows = useMemo(() => {
    if (!data) return [];
    const list = data.rows.filter((r) =>
      filter === "spam" ? r.verdict === "spam"
        : filter === "ham" ? r.verdict === "ham"
          : filter === "mistakes" ? r.correct === false
            : filter === "links" ? r.link_risk === "high" || r.link_risk === "medium"
              : true);
    return [...list].sort((a, b) => b.spam_probability - a.spam_probability);
  }, [data, filter]);

  const s = data?.summary;
  const ev = data?.evaluation;
  const filters = [
    ["all", "All"], ["spam", "Spam"], ["ham", "Ham"], ["links", "Risky links"],
    ...(data?.label_column ? [["mistakes", "Mistakes"]] : []),
  ];

  return (
    <div className="space-y-6">
      <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-bold">Batch analysis</h2>
        <p className="text-sm text-gray-600 mb-3">
          Upload a CSV with one message per row (up to 2,000 rows). The text column is detected automatically. If the
          file also has a <code>label</code> column with spam/ham values, you also get the accuracy on your data.
          The cascade is used: fast models for every row, the slow zero-shot model only for uncertain rows.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <label className={`bg-blue-600 text-white px-4 py-2 rounded-lg ${loading ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}>
            {loading ? "Analyzing..." : "Upload CSV"}
            <input type="file" accept=".csv,text/csv" className="sr-only" disabled={loading} onChange={onFile} />
          </label>
          {file && <span className="text-sm text-gray-600">{file.name}</span>}
          {data && data.columns.length > 1 && (
            <label className="text-sm flex items-center gap-2">
              Text column
              <select className="border rounded-md px-2 py-1 bg-white" value={column} disabled={loading}
                onChange={(e) => { setColumn(e.target.value); run(file, e.target.value); }}>
                {data.columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          )}
        </div>
        <p className="text-xs text-gray-500 mt-2">Try <code>Samples/batch_messages.csv</code> from the project folder.</p>
        {error && <p role="alert" className="mt-3 p-3 rounded-lg bg-red-100 text-red-700">{error}</p>}
      </section>

      {data && (
        <>
          {data.truncated && (
            <p className="p-3 rounded-lg bg-amber-100 text-amber-900 text-sm">
              The file has {data.total_rows} rows; only the first 2,000 were analysed.
            </p>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <Stat label="messages analysed" value={data.analyzed_rows}
              sub={data.skipped_empty ? `${data.skipped_empty} empty skipped` : null} />
            <Stat label="spam" value={s.spam} sub={pct(s.spam_share * 100)} />
            <Stat label="ham" value={s.ham} />
            <Stat label="high-risk links" value={s.phishing_links} />
            <Stat label="needed the slow model" value={s.escalated} />
            <Stat label="seconds" value={s.seconds} />
          </div>

          {ev && (
            <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
              <h2 className="text-xl font-bold mb-1">Accuracy on your labels</h2>
              <p className="text-sm text-gray-600 mb-3">Compared with the “{data.label_column}” column.</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Stat label="accuracy" value={pct(ev.accuracy * 100)} />
                <Stat label="spam precision" value={pct(ev.spam_precision * 100)} />
                <Stat label="spam recall" value={pct(ev.spam_recall * 100)} />
                <Stat label="spam F1" value={ev.spam_f1.toFixed(3)} />
              </div>
              <p className="text-sm mt-3">
                {ev.confusion[0][1]} false alarm{ev.confusion[0][1] === 1 ? "" : "s"} · {ev.confusion[1][0]} spam missed
                · {ev.confusion[1][1]} spam caught · {ev.confusion[0][0]} ham kept
              </p>
            </section>
          )}

          <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
              <div role="radiogroup" aria-label="Filter rows" className="inline-flex flex-wrap rounded-lg border p-0.5">
                {filters.map(([key, label]) => (
                  <button key={key} role="radio" aria-checked={filter === key}
                    onClick={() => { setFilter(key); setShown(PAGE); }}
                    className={`px-3 py-1 text-sm rounded-md ${filter === key ? "bg-gray-900 text-white" : "text-gray-700 hover:bg-gray-100"}`}>
                    {label}
                  </button>
                ))}
              </div>
              <button onClick={() => downloadCsv(data, file?.name ?? "batch.csv")}
                className="px-4 py-2 rounded-lg border hover:bg-gray-50">⬇ Download results CSV</button>
            </div>
            <p className="text-xs text-gray-500 mb-2">{rows.length} rows, most spam-like first.</p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm min-w-[40rem]">
                <thead>
                  <tr className="border-b text-left">
                    <th className="p-2">Row</th><th className="p-2">Message</th><th className="p-2">Verdict</th>
                    <th className="p-2">Spam prob.</th><th className="p-2">Links</th>
                    {data.label_column && <th className="p-2">Label</th>}
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, shown).map((r) => (
                    <tr key={r.row} className="border-b align-top">
                      <td className="p-2 tabular-nums text-gray-500">{r.row}</td>
                      <td className="p-2 max-w-[28rem]">
                        <p className="line-clamp-2 break-words" title={r.text}>{r.text}</p>
                        {r.escalated && <p className="text-xs text-gray-500">uncertain: zero-shot model also used</p>}
                      </td>
                      <td className="p-2">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold text-white ${r.verdict === "spam" ? "bg-red-600" : "bg-green-700"}`}>
                          {r.verdict.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-2 tabular-nums">{pct(r.spam_probability)}</td>
                      <td className="p-2">{r.link_risk === "high" ? "⛔ high" : r.link_risk === "medium" ? "⚠️ medium" : r.link_risk === "low" ? "✓" : "-"}</td>
                      {data.label_column && (
                        <td className="p-2 whitespace-nowrap">{r.label} {r.correct ? "✓" : <span className="text-red-700 font-bold">✗</span>}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {rows.length > shown && (
              <button onClick={() => setShown(shown + PAGE)} className="mt-3 px-4 py-2 rounded-lg border hover:bg-gray-50">
                Show {Math.min(PAGE, rows.length - shown)} more
              </button>
            )}
          </section>
        </>
      )}
    </div>
  );
}
