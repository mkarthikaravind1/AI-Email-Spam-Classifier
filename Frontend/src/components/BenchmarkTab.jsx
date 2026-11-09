import { useEffect, useState } from "react";
import { api, errorMessage, ENSEMBLE_COLOR, MODEL_BY_KEY, pct } from "../api";
import LineChart from "./LineChart";

const f3 = (v) => (v === null || v === undefined ? "-" : v.toFixed(3));

function Card({ title, subtitle, children }) {
  return (
    <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
      <h2 className="text-xl font-bold">{title}</h2>
      {subtitle && <p className="text-sm text-gray-600 mb-3">{subtitle}</p>}
      <div className={subtitle ? "" : "mt-3"}>{children}</div>
    </section>
  );
}

function Confusion({ name, color, m }) {
  const [[tn, fp], [fn, tp]] = m.confusion;
  const cell = (v, label, bad) => (
    <div className={`p-2 rounded text-center ${bad ? "bg-red-50" : "bg-green-50"}`}>
      <p className="text-lg font-bold tabular-nums">{v}</p>
      <p className="text-[11px] text-gray-600">{label}</p>
    </div>
  );
  return (
    <div>
      <p className="font-semibold text-sm mb-1">
        <span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: color }} />
        {name} <span className="font-normal text-gray-500">(n={m.n})</span>
      </p>
      <div className="grid grid-cols-2 gap-1">
        {cell(tn, "ham kept", false)}
        {cell(fp, "false alarms", fp > 0)}
        {cell(fn, "spam missed", fn > 0)}
        {cell(tp, "spam caught", false)}
      </div>
    </div>
  );
}

export default function BenchmarkTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    api.get("/benchmark")
      .then((r) => { setData(r.data); setError(""); })
      .catch((err) => setError(errorMessage(err)));
  }, [attempt]);

  if (error) {
    return (
      <div role="alert" className="p-3 rounded-lg bg-red-100 text-red-700 flex flex-wrap items-center gap-3">
        <span>{error}</span>
        <button onClick={() => { setError(""); setAttempt((a) => a + 1); }}
          className="px-3 py-1 rounded-lg bg-white border border-red-300 text-red-800">Retry</button>
      </div>
    );
  }
  if (!data) return <p className="text-gray-600">Loading benchmark...</p>;

  const models = Object.entries(data.models).map(([key, m]) => ({ key, ...MODEL_BY_KEY[key], m }));
  const stack = data.ensembles[0];
  const rows = [
    { name: data.baseline.label, m: data.baseline, eval: "full test split", muted: true },
    ...models.map((x) => ({ name: x.m.label, color: x.color, m: x.m, eval: x.m.eval_set })),
    ...data.ensembles.map((e, i) => ({ name: e.name, color: i === 0 ? ENSEMBLE_COLOR : undefined, m: e,
      eval: e.n === data.dataset.n_test ? "full test split" : `test sample (${e.n})` })),
  ];
  const bestF1 = Math.max(...rows.filter((r) => r.eval === "full test split").map((r) => r.m.spam_f1));
  const rocSeries = [
    ...models.map((x) => ({ key: x.key, name: `${x.name}${x.m.eval_set.startsWith("test sample") ? " (sample)" : ""}`, color: x.color, points: x.m.roc })),
    { key: "stacking", name: "Stacking", color: ENSEMBLE_COLOR, points: stack.roc },
  ];
  const calSeries = [
    ...models.map((x) => ({ key: x.key, name: x.name, color: x.color,
      points: x.m.calibration.map((b) => [b.predicted, b.actual]) })),
    { key: "stacking", name: "Stacking", color: ENSEMBLE_COLOR, points: stack.calibration.map((b) => [b.predicted, b.actual]) },
  ];
  const robustModels = Object.keys(data.robustness.clean);

  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-600">
        All numbers are on a held-out test split of {data.dataset.n_test} messages that no model, weight or stacker was
        trained on ({pct(data.dataset.spam_share * 100)} spam). Generated {new Date(data.generated_at).toLocaleString()}
        {data.dataset.feedback_labels_used > 0 && ` with ${data.dataset.feedback_labels_used} of your feedback labels`}.
        The zero-shot model is slow, so it is evaluated on a stratified sample.
      </p>

      <Card title="Leaderboard" subtitle="Spam F1 balances catching spam (recall) against false alarms (precision). Best full-test F1 in bold.">
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[44rem]">
            <thead>
              <tr className="border-b text-left">
                <th className="p-2">Model / ensemble</th><th className="p-2">Spam F1</th><th className="p-2">Precision</th>
                <th className="p-2">Recall</th><th className="p-2">AUC</th><th className="p-2">False alarms</th>
                <th className="p-2">Missed spam</th><th className="p-2">Evaluated on</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className={`border-b ${r.muted ? "text-gray-500" : ""}`}>
                  <td className="p-2">
                    {r.color && <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ background: r.color }} />}
                    {r.name}
                  </td>
                  <td className={`p-2 tabular-nums ${r.m.spam_f1 === bestF1 && r.eval === "full test split" ? "font-bold" : ""}`}>{f3(r.m.spam_f1)}</td>
                  <td className="p-2 tabular-nums">{f3(r.m.spam_precision)}</td>
                  <td className="p-2 tabular-nums">{f3(r.m.spam_recall)}</td>
                  <td className="p-2 tabular-nums">{f3(r.m.auc)}</td>
                  <td className="p-2 tabular-nums">{r.m.false_alarms}</td>
                  <td className="p-2 tabular-nums">{r.m.missed_spam}</td>
                  <td className="p-2 text-gray-500">{r.eval}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card title="ROC curves" subtitle="Spam caught vs. ham wrongly flagged, at every threshold. Closer to the top-left is better.">
          <LineChart series={rocSeries} xLabel="False positive rate" yLabel="True positive rate (recall)"
            referenceLabel="random guessing" ariaLabel="ROC curves for each model" />
        </Card>
        <Card title="Calibration" subtitle="When a model says 80% spam, is it spam 80% of the time? On the dashed line = honest probabilities.">
          <LineChart series={calSeries} max={100} format={(v) => `${Math.round(v)}%`} markers
            xLabel="Predicted spam probability" yLabel="Actual share of spam" referenceLabel="perfect calibration"
            ariaLabel="Calibration curves for each model" />
        </Card>
      </div>

      <Card title="Confusion matrices">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {models.map((x) => <Confusion key={x.key} name={x.name} color={x.color} m={x.m} />)}
          <Confusion name="Stacking" color={ENSEMBLE_COLOR} m={stack} />
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        {data.cascade && (
          <Card title="Cascade: speed vs. accuracy" subtitle={`Fast models first; the zero-shot model only when the fast verdict is between ${data.cascade.low}% and ${data.cascade.high}%. Measured on ${data.cascade.n} test messages.`}>
            <div className="grid grid-cols-3 gap-3 text-center">
              {[["Fast models only", data.cascade.f1_fast_only, data.cascade.latency_ms_fast],
                ["Cascade", data.cascade.f1_cascade, data.cascade.latency_ms_cascade],
                ["All models", data.cascade.f1_all_models, data.cascade.latency_ms_all_models]].map(([name, f1, ms]) => (
                <div key={name} className="border rounded-lg p-3">
                  <p className="text-sm text-gray-600">{name}</p>
                  <p className="text-2xl font-bold tabular-nums">{f3(f1)}</p>
                  <p className="text-xs text-gray-500">spam F1</p>
                  <p className="text-lg font-semibold tabular-nums mt-2">{ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`}</p>
                  <p className="text-xs text-gray-500">avg. latency</p>
                </div>
              ))}
            </div>
            <p className="text-sm mt-3">Only <b>{pct(data.cascade.escalation_rate * 100)}</b> of messages needed the slow model.</p>
          </Card>
        )}
        <Card title="What the stacker learned" subtitle="Logistic regression on each model's log-odds, trained on out-of-fold predictions. Bigger coefficient = trusted more.">
          {Object.entries(data.stacking).map(([kind, s]) => (
            <div key={kind} className="mb-3">
              <p className="font-semibold text-sm">{kind === "fast" ? "Fast stacker (used when the zero-shot model is skipped)" : "Full stacker (all models)"} · trained on {s.n_train} messages</p>
              <ul className="text-sm mt-1">
                {s.features.map((f, i) => (
                  <li key={f}>
                    <span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ background: MODEL_BY_KEY[f]?.color }} />
                    {MODEL_BY_KEY[f]?.name}: <span className="tabular-nums">{s.coef[i].toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </Card>
      </div>

      <Card title="Robustness to obfuscation" subtitle={`Share of the ${data.robustness.n_spam} test spam messages still caught after each attack, without → with de-obfuscation.`}>
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[40rem]">
            <thead>
              <tr className="border-b text-left">
                <th className="p-2">Attack</th>
                {robustModels.map((m) => <th key={m} className="p-2">{MODEL_BY_KEY[m]?.name ?? "Stacking"}</th>)}
              </tr>
            </thead>
            <tbody>
              <tr className="border-b bg-gray-50">
                <td className="p-2">No attack</td>
                {robustModels.map((m) => <td key={m} className="p-2 tabular-nums">{pct(data.robustness.clean[m] * 100, 0)}</td>)}
              </tr>
              {data.robustness.attacks.map((a) => (
                <tr key={a.name} className="border-b">
                  <td className="p-2">
                    <p>{a.description}</p>
                    <code className="text-xs text-gray-500">{a.example}</code>
                  </td>
                  {robustModels.map((m) => {
                    const drop = a.raw[m] < data.robustness.clean[m] - 0.05;
                    return (
                      <td key={m} className="p-2 tabular-nums whitespace-nowrap">
                        <span className={drop ? "text-red-700 font-semibold" : ""}>{pct(a.raw[m] * 100, 0)}</span>
                        {" → "}
                        <span>{pct(a.normalized[m] * 100, 0)}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Where the fast models disagree" subtitle="Test messages the models split on: the most instructive cases. Spam probability per model.">
        <div className="space-y-3">
          {data.disagreements.map((d) => (
            <div key={d.text} className="border rounded-lg p-3 text-sm">
              <p className="break-words">“{d.text}”</p>
              <p className="mt-1 text-xs text-gray-600">
                True label: <b className={d.label === "spam" ? "text-red-700" : "text-green-800"}>{d.label}</b>
                {" · "}Stacking: {pct(d.stacking, 0)}
                {Object.entries(d.models).map(([m, p]) => ` · ${MODEL_BY_KEY[m]?.name}: ${pct(p, 0)}`).join("")}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
