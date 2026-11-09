"""Trains the learned parts of the system, evaluates everything, writes the benchmark.

    python train_transformer.py            # once, optional: fine-tunes DistilBERT (slow)
    python train_model.py                  # ML model, regex calibration, stacking, benchmark
    python train_model.py --with-llm       # also (re)runs the zero-shot model (a few minutes)
    python train_model.py --include-feedback   # also learns from user feedback labels

Everything reported is measured on a held-out 20% test split that no model,
calibration, weight or stacker is fitted on. Stackers and weights are fitted
on out-of-fold predictions from the training split.

Saves:
  saved_models/spam_model.pkl   - statistical ML pipeline (refit on all data)
  saved_models/metrics.json     - weights, regex calibration, stackers, cascade thresholds
  saved_models/benchmark.json   - everything the Benchmark tab shows
  saved_models/llm_preds.json   - cached zero-shot predictions (reused without --with-llm)
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from classifiers import ensemble
from classifiers.normalize import ATTACK_DESCRIPTIONS, ATTACKS, normalize
from classifiers.regex_classifier import fit_calibration, get_matched_rules, spam_score
from classifiers.text_utils import (METRICS_PATH, MODEL_DIR, MODEL_PATH, clean_text, load_dataset)

BENCHMARK_PATH = MODEL_DIR / "benchmark.json"
DISTILBERT_PREDS_PATH = MODEL_DIR / "distilbert_preds.json"
LLM_PREDS_PATH = MODEL_DIR / "llm_preds.json"
LABELS = {"regex": "Regex rules", "ml": "Statistical ML", "distilbert": "DistilBERT (fine-tuned)",
          "llm": "Zero-shot NLI (BART)"}
CASCADE = {"low": 15.0, "high": 85.0}


# ------------------------------------------------------------------ models ---

def build_model():
    """Character n-grams + class-balanced logistic regression.

    Char n-grams are robust to misspellings ("fr33", "c@sh") and
    class_weight="balanced" compensates for spam being only ~13% of the data.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb",
                                  ngram_range=(2, 5), sublinear_tf=True)),
        ("clf", LogisticRegression(class_weight="balanced", C=10, max_iter=2000)),
    ])


def build_baseline():
    """The original TF-IDF + Multinomial Naive Bayes model, kept for comparison."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def ml_spam_proba(model, texts):
    return model.predict_proba(list(texts))[:, list(model.classes_).index("spam")] * 100


def regex_scores(texts):
    return np.array([spam_score(get_matched_rules(t)) for t in texts])


def regex_proba(scores, calibration):
    table = np.array(calibration)
    return table[np.minimum(scores, len(table) - 1)] * 100


# -------------------------------------------------------------- evaluation ---

def metrics_for(y_true, p_spam):
    """Full evaluation of spam probabilities (0-100) against labels."""
    y = (np.asarray(y_true) == "spam").astype(int)
    p = np.asarray(p_spam, dtype=float) / 100
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int))
    bins = []
    for lo in np.arange(0, 1, 0.1):
        mask = (p >= lo) & (p < lo + 0.1 if lo < 0.9 else p <= 1.0)
        if mask.sum():
            bins.append({"predicted": round(float(p[mask].mean()) * 100, 1),
                         "actual": round(float(y[mask].mean()) * 100, 1), "count": int(mask.sum())})
    return {
        "n": int(len(y)), "accuracy": round(float((pred == y).mean()), 4),
        "spam_precision": round(precision, 4), "spam_recall": round(recall, 4), "spam_f1": round(f1, 4),
        "false_alarms": int(fp), "missed_spam": int(fn),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
        "roc": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)] for i in keep],
        "calibration": bins,
    }


def print_metrics(name, m):
    print(f"  {name:<42} F1 {m['spam_f1']:.3f}  P {m['spam_precision']:.3f}  R {m['spam_recall']:.3f}"
          f"  AUC {m['auc'] or 0:.3f}  false alarms {m['false_alarms']:>3}  missed {m['missed_spam']:>3}"
          f"  (n={m['n']})")


def results_row(probs, names):
    return {n: {"prediction": "spam" if probs[n] >= 50 else "ham", "spam_probability": float(probs[n])}
            for n in names}


def fit_stacker(features, X, y):
    """Logistic regression on the models' log-odds."""
    Z = np.vectorize(ensemble.logit)(X)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z, (np.asarray(y) == "spam").astype(int))
    return {"features": list(features), "coef": [round(float(c), 4) for c in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 4), "n_train": int(len(y))}


def stack_proba(stacker, prob_table, idx):
    return np.array([ensemble.combine_stacking(results_row(prob_table.loc[i], stacker["features"]),
                                               {"fast": stacker})["spam_probability"] for i in idx])


# --------------------------------------------------------------------- main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="(re)run the zero-shot model (slow)")
    parser.add_argument("--llm-train-sample", type=int, default=600)
    parser.add_argument("--llm-test-sample", type=int, default=400)
    parser.add_argument("--include-feedback", action="store_true",
                        help="add user feedback labels to the training data")
    args = parser.parse_args()

    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"{len(df)} unique messages: {len(X_train)} train / {len(X_test)} test "
          f"({(df['label'] == 'spam').mean():.1%} spam)")

    # --- Feedback labels (feature #5) only ever join the *training* side -----
    feedback_rows = pd.DataFrame(columns=["text", "label"])
    if args.include_feedback:
        from classifiers import feedback_store
        feedback_store.init()
        feedback_rows = pd.DataFrame(feedback_store.all_feedback(), columns=["text", "label", "predicted",
                                                                             "created_at"])[["text", "label"]]
        test_texts = set(X_test)
        feedback_rows = feedback_rows[~feedback_rows["text"].isin(test_texts)]
        # A user label overrides the dataset label for the same message
        overridden = X_train.isin(set(feedback_rows["text"]))
        X_train, y_train = X_train[~overridden], y_train[~overridden]
        fb_index = pd.RangeIndex(-len(feedback_rows), 0)  # negative ids: not in the dataset
        X_train = pd.concat([X_train, pd.Series(feedback_rows["text"].values, index=fb_index)])
        y_train = pd.concat([y_train, pd.Series(feedback_rows["label"].values, index=fb_index)])
        print(f"Added {len(feedback_rows)} feedback labels to the training split")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = pd.DataFrame(index=X_train.index, dtype=float)    # out-of-fold P(spam) on train
    test = pd.DataFrame(index=X_test.index, dtype=float)    # P(spam) on test
    weights, latency_ms = {}, {}

    # --- Baseline -----------------------------------------------------------
    baseline = metrics_for(y_test, ml_spam_proba(build_baseline().fit(X_train, y_train), X_test))

    # --- Statistical ML -----------------------------------------------------
    oof["ml"] = cross_val_predict(build_model(), X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1] * 100  # classes sorted: ham, spam
    ml_model = build_model().fit(X_train, y_train)
    test["ml"] = ml_spam_proba(ml_model, X_test)
    t0 = time.perf_counter()
    for t in X_test[:200]:
        ml_spam_proba(ml_model, [t])
    latency_ms["ml"] = (time.perf_counter() - t0) / 200 * 1000

    # --- Regex (calibration fitted out-of-fold) -----------------------------
    train_scores, test_scores = regex_scores(X_train), regex_scores(X_test)
    for fit_idx, pred_idx in cv.split(X_train, y_train):
        cal = fit_calibration(list(train_scores[fit_idx]), list(y_train.iloc[fit_idx]))
        oof.loc[X_train.index[pred_idx], "regex"] = regex_proba(train_scores[pred_idx], cal)
    calibration = fit_calibration(list(train_scores), list(y_train))
    test["regex"] = regex_proba(test_scores, calibration)
    latency_ms["regex"] = 0.05

    # --- DistilBERT (predictions produced by train_transformer.py) ----------
    if DISTILBERT_PREDS_PATH.exists():
        dpreds = json.loads(DISTILBERT_PREDS_PATH.read_text())
        # Feedback rows (negative ids) have no DistilBERT prediction -> NaN, so
        # they are left out of stacker training but still used for the ML model
        oof["distilbert"] = [dpreds["oof"][str(i)] * 100 if str(i) in dpreds["oof"] else np.nan
                             for i in X_train.index]
        test["distilbert"] = [dpreds["test"][str(i)] * 100 for i in X_test.index]
        from classifiers import distilbert_classifier
        distilbert_classifier.spam_probabilities(["warm up"])
        t0 = time.perf_counter()
        for t in X_test[:50]:
            distilbert_classifier.spam_probabilities([t])
        latency_ms["distilbert"] = (time.perf_counter() - t0) / 50 * 1000
    else:
        print("DistilBERT not trained (run train_transformer.py); continuing without it.")

    # --- Zero-shot model: run or reuse cached predictions --------------------
    llm_train, llm_test = {}, {}
    if args.with_llm:
        from tqdm import tqdm

        from classifiers.llm_classifier import predict_llm

        def sample(X, y, n):
            if n >= len(X):
                return X
            return train_test_split(X, y, train_size=n, random_state=0, stratify=y)[0]

        Xs_train = sample(X_train[X_train.index >= 0], y_train[X_train.index >= 0], args.llm_train_sample)
        Xs_test = sample(X_test, y_test, args.llm_test_sample)
        print(f"Running the zero-shot model on {len(Xs_train)} train + {len(Xs_test)} test messages...")
        t0 = time.perf_counter()
        llm_train = {int(i): predict_llm(t)["spam_probability"] for i, t in tqdm(Xs_train.items(),
                                                                               total=len(Xs_train))}
        llm_ms = (time.perf_counter() - t0) / max(len(Xs_train), 1) * 1000
        llm_test = {int(i): predict_llm(t)["spam_probability"] for i, t in tqdm(Xs_test.items(),
                                                                              total=len(Xs_test))}
        LLM_PREDS_PATH.write_text(json.dumps({"train": llm_train, "test": llm_test, "ms_per_message": llm_ms}))
    elif LLM_PREDS_PATH.exists():
        cached = json.loads(LLM_PREDS_PATH.read_text())
        llm_train = {int(k): v for k, v in cached["train"].items()}
        llm_test = {int(k): v for k, v in cached["test"].items()}
        llm_ms = cached["ms_per_message"]
        print(f"Reusing cached zero-shot predictions ({len(llm_train)} train / {len(llm_test)} test).")
    if llm_test:
        oof["llm"] = [llm_train.get(int(i), np.nan) for i in X_train.index]
        test["llm"] = [llm_test.get(int(i), np.nan) for i in X_test.index]
        latency_ms["llm"] = llm_ms

    available = [m for m in ("regex", "ml", "distilbert", "llm") if m in test.columns]
    fast = [m for m in ensemble.FAST_MODELS if m in available]

    # --- Weighted-vote weights = out-of-fold spam F1 -------------------------
    for m in available:
        rows = oof[m].notna()
        weights[m] = round(f1_score(y_train[rows], np.where(oof.loc[rows, m] >= 50, "spam", "ham"),
                                    pos_label="spam"), 4)

    # --- Stacking (feature #6) ----------------------------------------------
    fast_rows = oof[fast].notna().all(axis=1)
    stacking = {"fast": fit_stacker(fast, oof.loc[fast_rows, fast].values, y_train[fast_rows])}
    if "llm" in available:
        full_feats = fast + ["llm"]
        full_rows = oof[full_feats].notna().all(axis=1)
        stacking["full"] = fit_stacker(full_feats, oof.loc[full_rows, full_feats].values, y_train[full_rows])

    # --- Evaluate everything on the test split -------------------------------
    print("\n=== Held-out test split ===")
    print_metrics("Baseline: TF-IDF + Naive Bayes", baseline)
    model_metrics = {}
    for m in available:
        rows = test[m].notna()
        model_metrics[m] = {"label": LABELS[m], "weight": weights[m],
                            "eval_set": "full test split" if rows.all() else f"test sample ({rows.sum()})",
                            "latency_ms": round(latency_ms.get(m, 0), 2),
                            **metrics_for(y_test[rows], test.loc[rows, m])}
        print_metrics(LABELS[m], model_metrics[m])

    def ensemble_metrics(name, idx, p):
        m = {"name": name, **metrics_for(y_test.loc[idx], p)}
        print_metrics(name, m)
        return m

    def majority(idx, models):
        votes = (test.loc[idx, models] >= 50).sum(axis=1)
        return np.where(votes > len(models) / 2, 100.0, 0.0)

    def weighted(idx, models):
        w = np.array([weights[m] for m in models])
        return (test.loc[idx, models].values * w).sum(axis=1) / w.sum()

    all_idx = X_test.index
    ensembles = [
        ensemble_metrics(f"Stacking ({' + '.join(fast)})", all_idx, stack_proba(stacking["fast"], test, all_idx)),
        ensemble_metrics(f"Weighted vote ({' + '.join(fast)})", all_idx, weighted(all_idx, fast)),
        ensemble_metrics(f"Majority vote ({' + '.join(fast)})", all_idx, majority(all_idx, fast)),
    ]
    cascade_report = None
    if "full" in stacking:
        s_idx = test.index[test["llm"].notna()]
        full_feats = stacking["full"]["features"]
        p_full = stack_proba(stacking["full"], test, s_idx)
        p_fast = stack_proba(stacking["fast"], test, s_idx)
        escalate = (p_fast >= CASCADE["low"]) & (p_fast <= CASCADE["high"])
        p_cascade = np.where(escalate, p_full, p_fast)
        ensembles += [
            ensemble_metrics("Stacking, all 4 models (sample)", s_idx, p_full),
            ensemble_metrics("Weighted vote, all 4 models (sample)", s_idx, weighted(s_idx, full_feats)),
            ensemble_metrics("Majority vote, all 4 models (sample)", s_idx, majority(s_idx, full_feats)),
            ensemble_metrics("Fast stacking only (sample)", s_idx, p_fast),
            ensemble_metrics("Cascade (sample)", s_idx, p_cascade),
        ]
        fast_ms = max(latency_ms[m] for m in fast)
        cascade_report = {
            **CASCADE, "n": int(len(s_idx)), "escalation_rate": round(float(escalate.mean()), 4),
            "f1_cascade": ensembles[-1]["spam_f1"], "f1_all_models": ensembles[-5]["spam_f1"],
            "f1_fast_only": ensembles[-2]["spam_f1"],
            "latency_ms_fast": round(fast_ms, 1),
            "latency_ms_all_models": round(max(fast_ms, latency_ms["llm"]), 1),
            "latency_ms_cascade": round(fast_ms + float(escalate.mean()) * latency_ms["llm"], 1),
        }
        print(f"  Cascade escalated {escalate.mean():.1%} of messages to the zero-shot model")

    # --- Where the fast models disagree (most instructive examples) ----------
    preds = test[fast] >= 50
    disagree_idx = preds.index[preds.nunique(axis=1) > 1]
    stack_test = pd.Series(stack_proba(stacking["fast"], test, all_idx), index=all_idx)
    disagreements = [{
        "text": X_test[i], "label": y_test[i],
        "models": {m: round(float(test.loc[i, m]), 1) for m in available if not np.isnan(test.loc[i, m])},
        "stacking": round(float(stack_test[i]), 1),
    } for i in disagree_idx[:15]]

    # --- Robustness to obfuscation (feature #4) ------------------------------
    print("\n=== Robustness: spam recall under obfuscation attacks ===")
    spam_texts = X_test[y_test == "spam"].tolist()
    robust_models = {
        "regex": lambda ts: regex_proba(regex_scores(ts), calibration),
        "ml": lambda ts: ml_spam_proba(ml_model, ts),
    }
    if "distilbert" in fast:
        robust_models["distilbert"] = lambda ts: np.array(distilbert_classifier.spam_probabilities(ts)) * 100

    def recall_all(texts):
        probs = {m: fn(texts) for m, fn in robust_models.items()}
        out = {m: round(float((p >= 50).mean()), 4) for m, p in probs.items()}
        table = pd.DataFrame(probs)
        out["stacking"] = round(float((stack_proba(stacking["fast"], table, table.index) >= 50).mean()), 4) \
            if set(stacking["fast"]["features"]) <= set(table.columns) else None
        return out

    robustness = {"n_spam": len(spam_texts), "clean": recall_all(spam_texts), "attacks": []}
    print(f"  {'clean':<16} {robustness['clean']}")
    for name, attack in ATTACKS.items():
        attacked = [attack(t) for t in spam_texts]
        raw = recall_all(attacked)
        fixed = recall_all([normalize(t)[0] for t in attacked])
        robustness["attacks"].append({"name": name, "description": ATTACK_DESCRIPTIONS[name],
                                      "example": attack("free cash prize"), "raw": raw, "normalized": fixed})
        print(f"  {name:<16} raw {raw}\n  {'':<16} fixed {fixed}")

    # --- Final fit on all data and save ----------------------------------------
    all_text = pd.concat([df["text"], feedback_rows["text"]]).drop_duplicates(keep="last")
    all_label = pd.concat([df.set_index("text")["label"], feedback_rows.set_index("text")["label"]])
    all_label = all_label[~all_label.index.duplicated(keep="last")].loc[all_text]
    final_model = build_model().fit(all_text, all_label.values)
    final_calibration = fit_calibration(list(regex_scores(all_text)), list(all_label.values))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "ml_model": "char n-gram TF-IDF + Logistic Regression",
        "models": {m: {"weight": weights[m], "test": {k: model_metrics[m][k] for k in
                   ("n", "accuracy", "spam_precision", "spam_recall", "spam_f1", "auc")}} for m in available},
        "regex_calibration": final_calibration,
        "stacking": stacking,
        "cascade": CASCADE,
        "feedback_labels_used": int(len(feedback_rows)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    BENCHMARK_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "spam_share": round(float((df["label"] == "spam").mean()), 4),
                    "feedback_labels_used": int(len(feedback_rows))},
        "baseline": {"label": "TF-IDF + Naive Bayes (original model)", **baseline},
        "models": model_metrics,
        "ensembles": ensembles,
        "stacking": stacking,
        "cascade": cascade_report,
        "disagreements": disagreements,
        "robustness": robustness,
    }))
    print(f"\nWeights (OOF F1): {weights}")
    print(f"Stacking: {json.dumps(stacking)}")
    print(f"Saved {MODEL_PATH.name}, {METRICS_PATH.name}, {BENCHMARK_PATH.name}")


if __name__ == "__main__":
    main()
