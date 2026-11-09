import { useState } from "react";
import { api, errorMessage, ENSEMBLE_COLOR, MAX_CHARS, MODELS, pct } from "../api";
import HighlightedText from "./HighlightedText";
import LinkReport from "./LinkReport";

const EXAMPLES = [
  "Hey, are we still meeting for lunch tomorrow at 1?",
  "WINNER!! You have been selected for a FR33 c@sh prize. Claim now: http://paypa1.com/login",
  "Your parcel is on hold. Pay the £1.99 fee at https://dhl-delivery-update.xyz/track",
  "I won the match yesterday!",
];

function Segmented({ value, onChange, options, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border bg-white p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={`px-3 py-1 text-sm rounded-md ${value === o.value ? "bg-gray-900 text-white" : "text-gray-700 hover:bg-gray-100"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function Card({ title, children, className = "" }) {
  return (
    <section className={`bg-white p-4 sm:p-6 rounded-xl shadow-md ${className}`}>
      {title && <h2 className="text-xl font-bold mb-3">{title}</h2>}
      {children}
    </section>
  );
}

function ModelCard({ model, data, weight, coefficient }) {
  return (
    <div className="bg-white p-3 rounded-xl shadow-md border-t-4" style={{ borderTopColor: model.color }}>
      <h3 className="font-bold">{model.name}</h3>
      <p className="text-xs text-gray-500 mb-2">{model.detail}</p>
      {!data ? (
        <p className="text-gray-400 text-sm">Not trained</p>
      ) : data.status === "skipped" ? (
        <p className="text-gray-500 text-sm">Skipped: {data.reason}. Switch to “Run all models” to see it.</p>
      ) : data.status === "error" ? (
        <p className="text-red-600 text-sm">Unavailable: {data.error}</p>
      ) : (
        <>
          <p>Prediction: <b>{data.prediction}</b></p>
          <p className="text-sm">Confidence: {pct(data.confidence)}</p>
          <p className="text-sm">Spam probability: {pct(data.spam_probability)}</p>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-2" aria-hidden="true">
            <div className="h-2 rounded-full" style={{ width: pct(data.spam_probability), background: model.color }} />
          </div>
          {coefficient !== undefined && (
            <p className="text-xs text-gray-500 mt-2">Stacking coefficient: {coefficient.toFixed(2)}</p>
          )}
          {weight !== undefined && (
            <p className="text-xs text-gray-500 mt-2">Vote weight: {pct(weight * 100, 0)}</p>
          )}
          {model.key === "regex" && data.matched_rules.length > 0 && (
            <ul className="text-xs mt-2 text-gray-700">
              {data.matched_rules.map((rule) => <li key={rule}>✓ {rule}</li>)}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

// How each model pushed the stacking verdict (log-odds), as a diverging bar
function Contributions({ result }) {
  if (result.contributions) {
    const entries = Object.entries(result.contributions);
    const max = Math.max(1, ...entries.map(([, v]) => Math.abs(v)));
    return (
      <div className="space-y-2">
        <p className="text-sm text-gray-600">
          A logistic regression learned how much to trust each model. Each bar is how far that model pushed
          this verdict (log-odds): right = towards spam, left = towards ham.
        </p>
        {entries.map(([key, v]) => {
          const model = MODELS.find((m) => m.key === key);
          const width = (Math.abs(v) / max) * 50;
          return (
            <div key={key} className="flex items-center gap-2 text-sm">
              <span className="w-28 shrink-0">{model?.name ?? key}</span>
              <div className="relative flex-1 h-4 bg-gray-100 rounded">
                <div className="absolute top-0 bottom-0 w-px bg-gray-400" style={{ left: "50%" }} />
                <div
                  className="absolute top-0.5 bottom-0.5 rounded"
                  style={{ background: model?.color, width: `${width}%`, left: v >= 0 ? "50%" : `${50 - width}%` }}
                />
              </div>
              <span className="w-14 text-right tabular-nums">{v > 0 ? "+" : ""}{v.toFixed(2)}</span>
            </div>
          );
        })}
        {result.ignored?.length > 0 && (
          <p className="text-xs text-gray-500">Not used by this stacker: {result.ignored.join(", ")}</p>
        )}
      </div>
    );
  }
  return (
    <div className="space-y-1 text-sm">
      <p className="text-gray-600">Weighted average of spam probabilities; weight = each model&apos;s validation F1.</p>
      {Object.entries(result.weights || {}).map(([key, w]) => (
        <p key={key}>{MODELS.find((m) => m.key === key)?.name ?? key}: {pct(w * 100, 0)}</p>
      ))}
    </div>
  );
}

function FeedbackButtons({ text, predicted }) {
  const [state, setState] = useState({ status: "idle" });
  const send = async (label) => {
    setState({ status: "sending" });
    try {
      const { data } = await api.post("/feedback", { email_text: text, label, predicted });
      setState({ status: "sent", label, total: data.total });
    } catch (err) {
      setState({ status: "error", message: errorMessage(err) });
    }
  };
  if (state.status === "sent") {
    return (
      <p className="text-sm mt-3">
        Saved as <b>{state.label}</b>. You now have {state.total} label{state.total === 1 ? "" : "s"}; retrain from the Feedback tab.
      </p>
    );
  }
  const other = predicted === "spam" ? "ham" : "spam";
  return (
    <div className="flex flex-wrap items-center gap-2 mt-3 text-sm">
      <span>Was this right?</span>
      <button disabled={state.status === "sending"} onClick={() => send(predicted)}
        className="px-3 py-1 rounded-lg bg-white border hover:bg-gray-50 disabled:opacity-50">👍 Yes</button>
      <button disabled={state.status === "sending"} onClick={() => send(other)}
        className="px-3 py-1 rounded-lg bg-white border hover:bg-gray-50 disabled:opacity-50">👎 No, it&apos;s {other}</button>
      {state.status === "error" && <span className="text-red-700">{state.message}</span>}
    </div>
  );
}

export default function AnalyzeTab() {
  const [emailText, setEmailText] = useState("");
  const [options, setOptions] = useState({ mode: "cascade", ensemble: "stacking", normalize: true });
  const [result, setResult] = useState(null);
  const [submitted, setSubmitted] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeEmail = async () => {
    if (loading) return;
    if (!emailText.trim()) {
      setError("Please enter an email.");
      return;
    }
    setResult(null);
    setError("");
    setLoading(true);
    try {
      const response = await api.post("/analyze", { email_text: emailText, ...options });
      setResult(response.data);
      setSubmitted(emailText.trim());
    } catch (err) {
      console.error(err);
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const spam = result?.final_verdict === "spam";
  const phishing = result?.links?.level === "high";

  return (
    <div className="space-y-6">
      <div className="grid lg:grid-cols-2 gap-6 items-start">
        {/* Input */}
        <div>
          <textarea
            className="w-full border rounded-lg p-4 h-40 bg-white"
            placeholder="Paste email text here..."
            aria-label="Message to analyse"
            value={emailText}
            maxLength={MAX_CHARS}
            onChange={(e) => setEmailText(e.target.value)}
          />
          <div className="flex flex-wrap items-center justify-between gap-2 mt-2">
            <button
              onClick={analyzeEmail}
              disabled={loading}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Analyze Email"}
            </button>
            <span className={`text-sm ${emailText.length >= MAX_CHARS ? "text-red-600" : "text-gray-500"}`}>
              {emailText.length} / {MAX_CHARS}
            </span>
          </div>
          <div className="flex flex-wrap gap-2 mt-3 text-xs">
            <span className="text-gray-500 self-center">Try:</span>
            {EXAMPLES.map((ex) => (
              <button key={ex} onClick={() => setEmailText(ex)}
                className="px-2 py-1 rounded-full bg-white border hover:bg-gray-50 max-w-[16rem] truncate">
                {ex}
              </button>
            ))}
          </div>
          {error && <p role="alert" className="mt-3 p-3 rounded-lg bg-red-100 text-red-700">{error}</p>}
          {loading && <p className="mt-3 text-gray-600" aria-live="polite">Analyzing... (the first request can take a while while the models warm up)</p>}
        </div>

        {/* Options */}
        <Card title="Options">
          <div className="space-y-4 text-sm">
            <div>
              <p className="font-semibold mb-1">Speed</p>
              <Segmented label="Speed" value={options.mode} onChange={(mode) => setOptions({ ...options, mode })}
                options={[{ value: "cascade", label: "Cascade" }, { value: "full", label: "Run all models" }]} />
              <p className="text-gray-500 mt-1">
                Cascade runs the fast models first and only calls the slow zero-shot model when they are unsure.
              </p>
            </div>
            <div>
              <p className="font-semibold mb-1">Combine models with</p>
              <Segmented label="Ensemble method" value={options.ensemble} onChange={(ensemble) => setOptions({ ...options, ensemble })}
                options={[{ value: "stacking", label: "Learned stacking" }, { value: "weighted", label: "F1-weighted vote" }]} />
            </div>
            <label className="flex items-start gap-2">
              <input type="checkbox" className="mt-1" checked={options.normalize}
                onChange={(e) => setOptions({ ...options, normalize: e.target.checked })} />
              <span>
                <span className="font-semibold">Undo obfuscation</span>
                <span className="block text-gray-500">Fixes fr33 c@sh, f r e e, Cyrillic look-alikes and invisible characters before classifying.</span>
              </span>
            </label>
          </div>
        </Card>
      </div>

      {result && (
        <>
          {/* Verdict */}
          <section className={`p-4 sm:p-6 rounded-xl ${spam ? "bg-red-100" : "bg-green-100"}`} aria-live="polite">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-2xl font-bold">Final Verdict</h2>
                <p className="text-sm text-gray-700">
                  {result.method} · spam probability {pct(result.spam_probability)} · {result.agreement} models agree · {result.processing_time} s
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                {phishing && <span className="px-3 py-2 rounded-full font-bold bg-gray-900 text-white">⛔ PHISHING LINK</span>}
                <span className={`px-4 py-2 rounded-full font-bold text-white ${spam ? "bg-red-600" : "bg-green-700"}`}>
                  {result.final_verdict.toUpperCase()}
                </span>
              </div>
            </div>
            <ul className="mt-3 text-sm space-y-1 text-gray-800">
              {result.alternative && (
                <li>
                  {result.alternative.method} would say <b>{result.alternative.final_verdict}</b> ({pct(result.alternative.spam_probability)}).
                </li>
              )}
              {result.cascade?.mode === "cascade" && (
                <li>
                  {result.cascade.escalated
                    ? `Cascade: ${result.cascade.reason}, so the zero-shot model was also run.`
                    : `Cascade: the fast models were confident (outside ${result.cascade.low}–${result.cascade.high}%), so the zero-shot model was skipped, saving about ${Math.round(result.cascade.time_saved_ms)} ms.`}
                </li>
              )}
              {result.explanation.normalization.length > 0 && (
                <li>De-obfuscated before classifying: {result.explanation.normalization.join(", ")}.</li>
              )}
              {result.queued_for_review && <li>Added to the labelling queue ({result.queued_for_review}).</li>}
            </ul>
            <FeedbackButtons text={submitted} predicted={result.final_verdict} />
          </section>

          {/* Model cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
            {MODELS.map((model) => (
              <ModelCard key={model.key} model={model} data={result[model.key]}
                weight={result.weights?.[model.key]} coefficient={result.coefficients?.[model.key]} />
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-6 items-start">
            <Card title="Why? Word-level explanation">
              <HighlightedText
                text={result.explanation.analyzed_text}
                mlWords={result.explanation.ml_words}
                regexSpans={result.explanation.regex_spans}
                links={result.explanation.normalization.length ? [] : result.links.links}
              />
            </Card>
            <div className="space-y-6">
              <Card title="Link safety">
                <LinkReport links={result.links} />
              </Card>
              <Card title="How the verdict was made">
                <Contributions result={result} />
              </Card>
            </div>
          </div>

          <Card title="Classification Pipeline">
            <div className="text-center space-y-2 text-sm">
              <p className="font-semibold">Message → de-obfuscation → link check</p>
              <p aria-hidden="true">↓</p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {MODELS.filter((m) => !m.slow).map((m) => (
                  <div key={m.key} className="border rounded-lg p-2 border-l-4" style={{ borderLeftColor: m.color }}>{m.name}</div>
                ))}
              </div>
              <p className="text-xs text-gray-500">fast models run in parallel</p>
              <p aria-hidden="true">↓</p>
              <p>Uncertain (between 15% and 85%)? → also run <b>Zero-shot NLI</b> (cascade)</p>
              <p aria-hidden="true">↓</p>
              <p>
                <span className="inline-block w-3 h-3 rounded-sm align-middle mr-1" style={{ background: ENSEMBLE_COLOR }} />
                Learned stacking (or F1-weighted vote) → <b>Final verdict</b> (spam if ≥ 50%)
              </p>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
