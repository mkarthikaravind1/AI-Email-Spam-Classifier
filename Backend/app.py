import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AfterValidator, BaseModel

from classifiers import batch, distilbert_classifier, ensemble, feedback_store
from classifiers.email_parser import check_sender, email_text_for_models, parse_eml
from classifiers.link_analyzer import analyze_links
from classifiers.llm_classifier import predict_llm
from classifiers.normalize import ATTACK_DESCRIPTIONS, ATTACKS, normalize
from classifiers.regex_classifier import find_spans, predict_regex, reload_calibration
from classifiers.statistical_ml_classifier import explain_ml, load_model, predict_ml
from classifiers.text_utils import BACKEND_DIR, MODEL_DIR, load_metrics

logger = logging.getLogger("spam_classifier")

# Fail at startup with a clear message if the ML model hasn't been trained yet
load_model()
ensemble.reload_config()
feedback_store.init()

MODEL_TIMEOUT_SECONDS = float(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
BENCHMARK_PATH = MODEL_DIR / "benchmark.json"

FAST_CLASSIFIERS = {"regex": predict_regex, "ml": predict_ml}
if distilbert_classifier.available():
    FAST_CLASSIFIERS["distilbert"] = distilbert_classifier.predict_distilbert
    distilbert_classifier.spam_probabilities(["warm up"])  # load now, not on the first request
SLOW_CLASSIFIERS = {"llm": predict_llm}

app = FastAPI(title="AI Email Spam Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
executor = ThreadPoolExecutor(max_workers=8)

# Running average of how long the slow model takes, to report time the cascade
# saved; starts from the latency measured by train_model.py --with-llm
def _measured_llm_ms():
    try:
        return float(json.loads((MODEL_DIR / "llm_preds.json").read_text())["ms_per_message"])
    except (OSError, KeyError, ValueError):
        return 1000.0


_slow_ms = {"llm": _measured_llm_ms()}


def _validate(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("email_text cannot be empty")
    if len(v) > 5000:
        raise ValueError("email_text exceeds 5000 character limit")
    return v


# Message text: stripped, non-empty, at most 5000 characters
EmailText = Annotated[str, AfterValidator(_validate)]


class EmailRequest(BaseModel):
    email_text: EmailText
    mode: Literal["cascade", "full"] = "cascade"
    ensemble: Literal["stacking", "weighted"] = "stacking"
    normalize: bool = True


class TextRequest(BaseModel):
    email_text: EmailText


class FeedbackRequest(BaseModel):
    email_text: EmailText
    label: Literal["spam", "ham"]
    predicted: Literal["spam", "ham"] | None = None


def _run(classifiers, text, deadline):
    """Runs classifiers in parallel; a failing or slow one becomes an error entry."""
    futures = {name: (executor.submit(fn, text), time.perf_counter()) for name, fn in classifiers.items()}
    results = {}
    for name, (future, started) in futures.items():
        try:
            results[name] = {"status": "ok", **future.result(timeout=max(0.0, deadline - time.perf_counter()))}
            if name in _slow_ms:
                _slow_ms[name] = 0.8 * _slow_ms[name] + 0.2 * (time.perf_counter() - started) * 1000
        except FutureTimeoutError:
            results[name] = {"status": "error", "error": f"timed out after {MODEL_TIMEOUT_SECONDS:g}s"}
        except Exception as exc:  # one broken model must not take down the request
            logger.exception("%s classifier failed", name)
            results[name] = {"status": "error", "error": str(exc) or type(exc).__name__}
    return results


@app.get("/")
def home():
    cfg = ensemble.config()
    return {
        "message": "Backend running",
        "models": list(FAST_CLASSIFIERS) + list(SLOW_CLASSIFIERS),
        "weights": cfg["weights"],
        "stacking": cfg["stacking"],
        "cascade": cfg["cascade"],
    }


@app.post("/analyze")
def analyze_email(request: EmailRequest):
    return _analyze(request.email_text, request.mode, request.ensemble, request.normalize)


def _analyze(original, mode="cascade", method="stacking", undo_obfuscation=True, link_text=None):
    """The full single-message pipeline, shared by /analyze and /analyze-eml."""
    start = time.perf_counter()
    deadline = start + MODEL_TIMEOUT_SECONDS

    # Feature #4: undo obfuscation before classifying
    text, tricks = normalize(original) if undo_obfuscation else (original, [])

    if mode == "full":
        results = _run({**FAST_CLASSIFIERS, **SLOW_CLASSIFIERS}, text, deadline)
        cascade = {"mode": "full", "escalated": True}
    else:
        # Feature #2: fast models first; the slow model only if they are unsure
        results = _run(FAST_CLASSIFIERS, text, deadline)
        fast_verdict = ensemble.combine(results, method)
        if fast_verdict is None or ensemble.needs_slow_models(fast_verdict):
            results.update(_run(SLOW_CLASSIFIERS, text, deadline))
            cascade = {"mode": "cascade", "escalated": True,
                       "reason": "fast models were uncertain" if fast_verdict else "fast models failed"}
        else:
            for name in SLOW_CLASSIFIERS:
                results[name] = {"status": "skipped", "reason": "fast models were confident"}
            cascade = {"mode": "cascade", "escalated": False,
                       "fast_spam_probability": fast_verdict["spam_probability"],
                       "time_saved_ms": round(sum(_slow_ms.values()))}
        cascade.update(ensemble.config()["cascade"])

    verdict = ensemble.combine(results, method)
    if verdict is None:
        raise HTTPException(status_code=503, detail="All classifiers failed; check the backend logs.")
    other = ensemble.combine(results, "weighted" if method == "stacking" else "stacking")

    # Feature #1: explanations on the text the models actually saw
    explanation = {"analyzed_text": text, "normalization": tricks, "regex_spans": find_spans(text)}
    try:
        explanation["ml_words"] = explain_ml(text)
    except Exception:
        logger.exception("ML explanation failed")
        explanation["ml_words"] = []

    # Feature #8: link checks on the original text (normalising would hide look-alike domains)
    links = analyze_links(link_text if link_text is not None else original)

    # Feature #5: queue uncertain/disputed messages for labelling
    predictions = [r["prediction"] for r in results.values() if r["status"] == "ok"]
    queued = feedback_store.queue_if_uncertain(original, verdict["spam_probability"], predictions)

    return {
        **results,
        **verdict,
        "alternative": other,
        "cascade": cascade,
        "explanation": explanation,
        "links": links,
        "queued_for_review": queued,
        "processing_time": round(time.perf_counter() - start, 2),
    }


# ------------------------------------- feature #9: real emails and batches ---

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read_upload(file: UploadFile, kinds):
    if not file.filename or not file.filename.lower().endswith(kinds):
        raise HTTPException(status_code=422, detail=f"Please upload a {' or '.join(kinds)} file.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 10 MB.")
    if not data.strip():
        raise HTTPException(status_code=422, detail="The file is empty.")
    return data


@app.post("/analyze-eml")
async def analyze_eml(
    file: UploadFile = File(...),
    mode: Literal["cascade", "full"] = Form("cascade"),
    ensemble_method: Literal["stacking", "weighted"] = Form("stacking", alias="ensemble"),
    normalize_text: bool = Form(True, alias="normalize"),
):
    """Analyses a saved email (.eml): text models on subject + body, plus
    sender/header checks, attachments and the real targets of HTML links."""
    data = await _read_upload(file, (".eml",))
    try:
        email = parse_eml(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read this .eml file: {exc}") from exc
    if not (email["from"]["address"] or email["subject"] or email["date"]):
        raise HTTPException(status_code=422, detail="This doesn't look like an email: no From, Subject or "
                                                    "Date header. Save the message as .eml from your mail app.")
    text = email_text_for_models(email)
    if not text:
        raise HTTPException(status_code=422, detail="This email has no subject or text body to analyse.")

    sender = check_sender(email)
    # Check visible text *and* the hidden targets of HTML links
    link_text = text + "\n" + "\n".join(link["href"] for link in email["html_links"] if link["href"])
    # Sync model work runs in the thread pool so the event loop stays free
    result = await run_in_threadpool(_analyze, text, mode, ensemble_method, normalize_text, link_text)
    result["email"] = {
        **{k: email[k] for k in ("from", "reply_to", "return_path", "to", "subject", "date", "auth",
                                 "body_format", "attachments")},
        "filename": file.filename,
        "text": text,  # subject + body as given to the models (before de-obfuscation)
        "sender_checks": sender,
    }
    return result


@app.post("/analyze-batch")
async def analyze_batch(
    file: UploadFile = File(...),
    column: str | None = Form(None),
    ensemble_method: Literal["stacking", "weighted"] = Form("stacking", alias="ensemble"),
    normalize_text: bool = Form(True, alias="normalize"),
):
    """Classifies every row of a CSV with the cascade (fast models for all rows,
    the slow model only for uncertain ones). Rows are not added to the queue."""
    data = await _read_upload(file, (".csv",))
    try:
        df = batch.read_csv(data)
        if df.empty:
            raise batch.BatchError("The CSV has no rows.")
        text_column = batch.pick_text_column(df, column or None)
        label_column = batch.pick_label_column(df, text_column)
    except batch.BatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await run_in_threadpool(_score_batch, df, text_column, label_column, ensemble_method, normalize_text)


def _score_batch(df, text_column, label_column, method, undo_obfuscation):
    from classifiers import statistical_ml_classifier
    from classifiers.regex_classifier import get_matched_rules, spam_probability, spam_score

    start = time.perf_counter()
    total = len(df)
    df = df.head(batch.MAX_ROWS)
    raw = df[text_column].astype(str).str.strip()
    keep = raw != ""
    rows_index = list(df.index[keep])
    originals = [raw[i][:batch.MAX_TEXT] for i in rows_index]
    texts = [normalize(t)[0] if undo_obfuscation else t for t in originals]
    if not texts:
        raise HTTPException(status_code=422, detail=f"Column “{text_column}” has no text.")

    # Fast models, vectorised over all rows
    probs = {
        "regex": [spam_probability(spam_score(get_matched_rules(t))) * 100 for t in texts],
        "ml": list(statistical_ml_classifier.spam_probabilities(texts) * 100),
    }
    if "distilbert" in FAST_CLASSIFIERS:
        probs["distilbert"] = [p * 100 for p in distilbert_classifier.spam_probabilities(texts)]

    labels = batch.normalise_labels(df.loc[rows_index, label_column]).tolist() if label_column else None
    rows, escalated = [], 0
    for k, (i, original, text) in enumerate(zip(rows_index, originals, texts)):
        results = {m: {"status": "ok", "prediction": "spam" if p[k] >= 50 else "ham",
                       "spam_probability": round(float(p[k]), 2)} for m, p in probs.items()}
        verdict = ensemble.combine(results, method)
        if ensemble.needs_slow_models(verdict):  # cascade
            escalated += 1
            try:
                results["llm"] = {"status": "ok", **predict_llm(text)}
            except Exception as exc:
                logger.exception("zero-shot model failed on a batch row")
                results["llm"] = {"status": "error", "error": str(exc)}
            verdict = ensemble.combine(results, method)
        link_report = analyze_links(original)
        row = {
            "row": int(i) + 2,  # spreadsheet row number (header is row 1)
            "text": original,
            "verdict": verdict["final_verdict"],
            "spam_probability": verdict["spam_probability"],
            "models": {m: r["spam_probability"] for m, r in results.items() if r["status"] == "ok"},
            "escalated": "llm" in results,
            "link_risk": link_report["level"],
        }
        if labels is not None:
            row["label"] = labels[k]
            row["correct"] = labels[k] == verdict["final_verdict"]
        rows.append(row)

    spam = sum(r["verdict"] == "spam" for r in rows)
    return {
        "text_column": text_column,
        "columns": [str(c) for c in df.columns],
        "label_column": label_column,
        "total_rows": total,
        "analyzed_rows": len(rows),
        "truncated": total > batch.MAX_ROWS,
        "skipped_empty": int((~keep).sum()),
        "summary": {
            "spam": spam, "ham": len(rows) - spam, "spam_share": round(spam / len(rows), 4),
            "phishing_links": sum(r["link_risk"] == "high" for r in rows),
            "escalated": escalated, "method": method,
            "seconds": round(time.perf_counter() - start, 2),
        },
        "evaluation": batch.evaluate(labels, [r["verdict"] for r in rows]) if labels else None,
        "rows": rows,
    }


# ------------------------------------------------- feature #4: robustness ---

@app.post("/robustness")
def robustness(request: TextRequest):
    """Attacks the message with each obfuscation trick and shows which fast
    models still catch it, with and without de-obfuscation."""
    from classifiers import statistical_ml_classifier
    from classifiers.regex_classifier import get_matched_rules, spam_probability, spam_score

    base = request.email_text
    variants = [("original", "No obfuscation", base)] + [
        (name, ATTACK_DESCRIPTIONS[name], attack(base)) for name, attack in ATTACKS.items()
    ]
    texts = [v[2] for v in variants]
    fixed = [normalize(t)[0] for t in texts]

    def score(batch):
        probs = {
            "regex": [spam_probability(spam_score(get_matched_rules(t))) * 100 for t in batch],
            "ml": list(statistical_ml_classifier.spam_probabilities(batch) * 100),
        }
        if "distilbert" in FAST_CLASSIFIERS:
            probs["distilbert"] = list(np.array(distilbert_classifier.spam_probabilities(batch)) * 100)
        rows = []
        for i in range(len(batch)):
            results = {m: {"status": "ok", "prediction": "spam" if p[i] >= 50 else "ham",
                           "spam_probability": round(float(p[i]), 2)} for m, p in probs.items()}
            rows.append({"models": {m: r["spam_probability"] for m, r in results.items()},
                         "ensemble": ensemble.combine_stacking(results)["spam_probability"]})
        return rows

    raw_scores, fixed_scores = score(texts), score(fixed)
    return {"variants": [
        {"name": name, "description": desc, "text": text, "normalized_text": norm,
         "raw": raw, "normalized": fix}
        for (name, desc, text), norm, raw, fix in zip(variants, fixed, raw_scores, fixed_scores)
    ]}


# --------------------------------------------------- feature #3: benchmark ---

@app.get("/benchmark")
def benchmark():
    if not BENCHMARK_PATH.exists():
        raise HTTPException(status_code=404, detail="No benchmark yet: run python train_model.py")
    return json.loads(BENCHMARK_PATH.read_text())


# ------------------------------------------------ feature #5: feedback loop ---

@app.post("/feedback")
def add_feedback(request: FeedbackRequest):
    feedback_store.add_feedback(request.email_text, request.label, request.predicted)
    return feedback_store.stats()


@app.get("/feedback/stats")
def feedback_stats():
    return {**feedback_store.stats(), "used_in_model": load_metrics().get("feedback_labels_used", 0)}


@app.get("/feedback/queue")
def feedback_queue():
    return feedback_store.get_queue()


@app.delete("/feedback/queue/{queue_id}")
def dismiss_queued(queue_id: int):
    feedback_store.dismiss(queue_id)
    return feedback_store.stats()


_retrain = {"state": "idle", "started_at": None, "finished_at": None, "log": "", "error": None}
_retrain_lock = threading.Lock()


def _retrain_job():
    try:
        proc = subprocess.run(
            [sys.executable, "train_model.py", "--include-feedback"],
            cwd=BACKEND_DIR, capture_output=True, text=True, timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        _retrain["log"] = (proc.stdout + proc.stderr)[-4000:]
        if proc.returncode != 0:
            raise RuntimeError(f"train_model.py exited with code {proc.returncode}")
        # Hot-reload everything the training run rewrote
        load_model(force=True)
        reload_calibration()
        ensemble.reload_config()
        _retrain["state"] = "done"
    except Exception as exc:
        logger.exception("Retraining failed")
        _retrain.update(state="failed", error=str(exc))
    finally:
        _retrain["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.post("/retrain")
def retrain():
    with _retrain_lock:
        if _retrain["state"] == "running":
            raise HTTPException(status_code=409, detail="Retraining is already running")
        _retrain.update(state="running", started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        finished_at=None, log="", error=None)
    threading.Thread(target=_retrain_job, daemon=True).start()
    return _retrain


@app.get("/retrain/status")
def retrain_status():
    return {**_retrain, "feedback_labels_used": load_metrics().get("feedback_labels_used", 0)}
