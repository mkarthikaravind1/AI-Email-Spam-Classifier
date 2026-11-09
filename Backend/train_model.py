"""Trains, evaluates and saves the spam classifiers' learned parts.

    python train_model.py              # ML model + regex calibration (fast)
    python train_model.py --with-llm   # also measures the zero-shot model (a few minutes)

Saves:
  saved_models/spam_model.pkl  - the statistical ML pipeline
  saved_models/metrics.json    - test-set metrics, the regex score->probability
                                 table, and the ensemble weight of each model
"""
import argparse
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from classifiers.ensemble import combine
from classifiers.regex_classifier import fit_calibration, get_matched_rules, predict_regex, spam_score
from classifiers.text_utils import METRICS_PATH, MODEL_DIR, MODEL_PATH, clean_text, load_dataset, load_metrics


def build_model():
    """Character n-grams + class-balanced logistic regression.

    Char n-grams are robust to misspellings/obfuscation ("fr33", "c@sh") and
    class_weight="balanced" compensates for spam being only ~13% of the data,
    which made the old Naive Bayes model miss over a third of spam.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(class_weight="balanced", C=10, max_iter=2000)),
    ])


def build_baseline():
    """The original TF-IDF + Multinomial Naive Bayes model, kept for comparison."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def report(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion matrix [rows = true ham/spam, cols = predicted ham/spam]:")
    print(confusion_matrix(y_true, y_pred, labels=["ham", "spam"]))
    r = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": round(r["accuracy"], 4),
        "spam_precision": round(r["spam"]["precision"], 4),
        "spam_recall": round(r["spam"]["recall"], 4),
        "spam_f1": round(r["spam"]["f1-score"], 4),
        "n": len(y_true),
    }


def spam_f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, pos_label="spam"), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true",
                        help="also evaluate the zero-shot model (slow on CPU)")
    parser.add_argument("--llm-sample", type=int, default=200,
                        help="messages per split to run the zero-shot model on")
    args = parser.parse_args()

    df = load_dataset()
    print(f"Loaded {len(df)} unique messages: {df['label'].value_counts().to_dict()}")

    # 80/20 split, stratified to preserve the spam/ham ratio. The test split is
    # only used for the reported numbers; ensemble weights come from the
    # training split, so the test numbers are not tuned on the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    models = {}

    # --- Baseline (for comparison only) ---
    report("Baseline: TF-IDF + Naive Bayes (test)",
           y_test, build_baseline().fit(X_train, y_train).predict(X_test))

    # --- Statistical ML ---
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    ml_val_f1 = spam_f1(y_train, cross_val_predict(build_model(), X_train, y_train, cv=cv))
    ml_model = build_model().fit(X_train, y_train)
    ml_test = report("Statistical ML: char n-gram + Logistic Regression (test)",
                     y_test, ml_model.predict(X_test))
    models["ml"] = {"weight": ml_val_f1, "test": ml_test}

    # --- Regex: learn score -> spam probability on the training split ---
    train_scores = [spam_score(get_matched_rules(t)) for t in X_train]
    calibration = fit_calibration(train_scores, list(y_train))
    print("\nRegex calibration (score -> P(spam)):", calibration)
    regex_val_f1 = spam_f1(y_train, [predict_regex(t, calibration)["prediction"] for t in X_train])
    regex_test_results = [predict_regex(t, calibration) for t in X_test]
    regex_test = report("Regex rules, calibrated (test)",
                        y_test, [r["prediction"] for r in regex_test_results])
    models["regex"] = {"weight": regex_val_f1, "test": regex_test}

    # --- Zero-shot model (optional, slow) ---
    llm_test_results = None
    if args.with_llm:
        from tqdm import tqdm

        from classifiers.llm_classifier import predict_llm

        def stratified_sample(X, y, n):
            if n >= len(X):
                return X, y
            Xs, _, ys, _ = train_test_split(X, y, train_size=n, random_state=0, stratify=y)
            return Xs, ys

        X_lv, y_lv = stratified_sample(X_train, y_train, args.llm_sample)
        X_lt, y_lt = stratified_sample(X_test, y_test, args.llm_sample)
        print(f"\nRunning the zero-shot model on {len(X_lv)} validation + {len(X_lt)} test messages...")
        llm_val_f1 = spam_f1(y_lv, [predict_llm(t)["prediction"] for t in tqdm(X_lv)])
        llm_test_results = dict(zip(X_lt.index, (predict_llm(t) for t in tqdm(X_lt))))
        llm_test = report("Zero-shot BART (test sample)",
                          y_lt, [r["prediction"] for r in llm_test_results.values()])
        models["llm"] = {"weight": llm_val_f1, "test": llm_test}
    elif "llm" in load_metrics().get("models", {}):
        models["llm"] = load_metrics()["models"]["llm"]
        print("\nKeeping previously measured zero-shot metrics (run with --with-llm to refresh).")
    else:
        print("\nZero-shot model not evaluated; it gets weight 1.0 until you run --with-llm.")

    # --- Ensemble on the test split ---
    from classifiers.statistical_ml_classifier import predict_ml

    weights = {name: m["weight"] for name, m in models.items()}
    ensemble_preds, majority_preds, ensemble_true = [], [], []
    for (idx, text), regex_result in zip(X_test.items(), regex_test_results):
        results = {"regex": regex_result, "ml": predict_ml(text, ml_model)}
        if llm_test_results is not None:
            if idx not in llm_test_results:
                continue
            results["llm"] = llm_test_results[idx]
        ensemble_preds.append(combine(results, weights)["final_verdict"])
        spam_votes = sum(r["prediction"] == "spam" for r in results.values())
        majority_preds.append("spam" if spam_votes > len(results) / 2 else "ham")
        ensemble_true.append(y_test[idx])
    used = "regex + ML + zero-shot" if llm_test_results is not None else "regex + ML"
    report(f"Old majority vote: {used} (same test rows, for comparison)", ensemble_true, majority_preds)
    ensemble_test = report(f"Weighted ensemble: {used} (test)", ensemble_true, ensemble_preds)

    # Metrics are measured on held-out data; the saved model is then refit on
    # all data so no labelled examples are wasted.
    final_model = build_model().fit(df["text"], df["label"])
    final_calibration = fit_calibration([spam_score(get_matched_rules(t)) for t in df["text"]],
                                        list(df["label"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "ml_model": "char n-gram TF-IDF + Logistic Regression",
        "models": models,
        "regex_calibration": final_calibration,
        "ensemble_test": {"models": used, **ensemble_test},
    }, indent=2))
    print("\nEnsemble weights (validation spam F1):", weights)
    print(f"Saved model to {MODEL_PATH}\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
