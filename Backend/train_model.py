"""Trains, evaluates and saves the statistical ML spam classifier.

Run from anywhere:  python Backend/train_model.py
"""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from classifiers.text_utils import METRICS_PATH, MODEL_DIR, MODEL_PATH, clean_text, load_dataset


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


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n===== {name} (held-out test set) =====")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix [rows = true ham/spam, cols = predicted ham/spam]:")
    print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))
    return classification_report(y_test, y_pred, output_dict=True)


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} unique messages: {df['label'].value_counts().to_dict()}")

    # 80/20 split, stratified to preserve the spam/ham ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    evaluate("Baseline: TF-IDF + Naive Bayes", build_baseline().fit(X_train, y_train), X_test, y_test)
    report = evaluate("Model: char n-gram + Logistic Regression",
                      build_model().fit(X_train, y_train), X_test, y_test)

    # Metrics are measured on the held-out split; the saved model is then
    # refit on all data so no labelled examples are wasted.
    final_model = build_model().fit(df["text"], df["label"])

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "model": "char n-gram TF-IDF + Logistic Regression",
        "test_accuracy": round(report["accuracy"], 4),
        "spam_precision": round(report["spam"]["precision"], 4),
        "spam_recall": round(report["spam"]["recall"], 4),
        "spam_f1": round(report["spam"]["f1-score"], 4),
    }, indent=2))
    print(f"\nSaved model to {MODEL_PATH}\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
