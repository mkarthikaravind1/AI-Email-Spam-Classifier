import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage, pct } from "../api";

// Fetches everything this tab shows (no state changes here)
async function load() {
  try {
    const [s, q, r] = await Promise.all([
      api.get("/feedback/stats"), api.get("/feedback/queue"), api.get("/retrain/status"),
    ]);
    return { stats: s.data, queue: q.data, retrain: r.data };
  } catch (err) {
    return { error: errorMessage(err) };
  }
}

function Stat({ label, value }) {
  return (
    <div className="border rounded-lg p-3 text-center">
      <p className="text-2xl font-bold tabular-nums">{value}</p>
      <p className="text-xs text-gray-600">{label}</p>
    </div>
  );
}

export default function FeedbackTab() {
  const [stats, setStats] = useState(null);
  const [queue, setQueue] = useState([]);
  const [retrain, setRetrain] = useState(null);
  const [error, setError] = useState("");
  const poll = useRef(null);

  // Applies a load() result to state; returns the retrain status (or null on error)
  const apply = useCallback((d) => {
    if (d.error) {
      setError(d.error);
      return null;
    }
    setStats(d.stats);
    setQueue(d.queue);
    setRetrain(d.retrain);
    setError("");
    return d.retrain;
  }, []);

  const refresh = useCallback(async () => apply(await load()), [apply]);

  const watchRetrain = useCallback(() => {
    clearInterval(poll.current);
    poll.current = setInterval(async () => {
      const r = await refresh();
      if (!r || r.state !== "running") clearInterval(poll.current);
    }, 3000);
  }, [refresh]);

  useEffect(() => {
    // Load on open, and keep watching if a retrain is already in progress
    let cancelled = false;
    load().then((d) => {
      if (cancelled) return;
      const r = apply(d);
      if (r?.state === "running") watchRetrain();
    });
    return () => {
      cancelled = true;
      clearInterval(poll.current);
    };
  }, [apply, watchRetrain]);

  const label = async (item, value) => {
    try {
      await api.post("/feedback", { email_text: item.text, label: value });
      refresh();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const dismiss = async (item) => {
    try {
      await api.delete(`/feedback/queue/${item.id}`);
      refresh();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const startRetrain = async () => {
    try {
      const { data } = await api.post("/retrain");
      setRetrain(data);
      watchRetrain();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const running = retrain?.state === "running";
  const newLabels = stats ? stats.total - stats.used_in_model : 0;

  return (
    <div className="space-y-6">
      {error && <p role="alert" className="p-3 rounded-lg bg-red-100 text-red-700">{error}</p>}

      <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-bold">Feedback loop</h2>
        <p className="text-sm text-gray-600 mb-4">
          Your labels (from 👍/👎 on the Analyze tab, or from the queue below) are stored locally and added to the
          training data when you retrain. The test set is never changed, so the Benchmark stays comparable.
        </p>
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <Stat label="labels saved" value={stats.total} />
            <Stat label="spam" value={stats.spam} />
            <Stat label="ham" value={stats.ham} />
            <Stat label="corrected the model" value={stats.corrections} />
            <Stat label="used in current model" value={stats.used_in_model} />
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3 mt-4">
          <button onClick={startRetrain} disabled={running || !stats || stats.total === 0}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed">
            {running ? "Retraining..." : `Retrain with ${stats?.total ?? 0} label${stats?.total === 1 ? "" : "s"}`}
          </button>
          <span className="text-sm text-gray-600" aria-live="polite">
            {running && "Retraining the ML model, regex calibration and stacker (about a minute)..."}
            {retrain?.state === "done" && `Done at ${new Date(retrain.finished_at).toLocaleTimeString()}. The new model is live and the Benchmark is updated.`}
            {retrain?.state === "failed" && <span className="text-red-700">Retraining failed: {retrain.error}</span>}
            {!running && retrain?.state !== "done" && newLabels > 0 && `${newLabels} new label${newLabels === 1 ? "" : "s"} not yet in the model.`}
          </span>
        </div>
        <p className="text-xs text-gray-500 mt-2">DistilBERT and the zero-shot model are not retrained here (they take much longer).</p>
      </section>

      <section className="bg-white p-4 sm:p-6 rounded-xl shadow-md">
        <h2 className="text-xl font-bold">Needs a label</h2>
        <p className="text-sm text-gray-600 mb-3">
          Messages you analysed where the models were unsure or disagreed, closest to 50% first: labelling these
          teaches the model the most (active learning).
        </p>
        {queue.length === 0 ? (
          <p className="text-gray-500 text-sm">Nothing waiting. Uncertain messages appear here after you analyse them.</p>
        ) : (
          <ul className="space-y-3">
            {queue.map((item) => (
              <li key={item.id} className="border rounded-lg p-3">
                <p className="break-words">“{item.text}”</p>
                <p className="text-xs text-gray-500 mt-1">{pct(item.spam_probability, 0)} spam · {item.reason}</p>
                <div className="flex flex-wrap gap-2 mt-2 text-sm">
                  <button onClick={() => label(item, "spam")} className="px-3 py-1 rounded-lg border hover:bg-red-50">It&apos;s spam</button>
                  <button onClick={() => label(item, "ham")} className="px-3 py-1 rounded-lg border hover:bg-green-50">It&apos;s ham</button>
                  <button onClick={() => dismiss(item)} className="px-3 py-1 rounded-lg text-gray-500 hover:bg-gray-100">Skip</button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
