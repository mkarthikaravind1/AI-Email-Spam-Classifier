"""Batch CSV analysis helpers (feature #9): reading the file, finding the text
and (optional) label columns, and scoring results against the labels."""
import io

import pandas as pd

MAX_ROWS = 2000
MAX_TEXT = 5000
TEXT_COLUMN_NAMES = ("text", "message", "messages", "email", "body", "content", "sms", "email_text", "v2")
LABEL_COLUMN_NAMES = ("label", "labels", "class", "category", "type", "is_spam", "spam", "target", "v1")
_LABEL_VALUES = {
    "spam": "spam", "1": "spam", "true": "spam", "yes": "spam", "junk": "spam", "phishing": "spam",
    "ham": "ham", "0": "ham", "false": "ham", "no": "ham", "not spam": "ham", "legit": "ham", "normal": "ham",
}


class BatchError(ValueError):
    pass


def read_csv(data: bytes):
    """Reads a CSV with unknown encoding and delimiter."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding,
                               dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise BatchError(f"Could not read the CSV: {exc}") from exc
    raise BatchError("Could not decode the CSV (try saving it as UTF-8).")


def pick_text_column(df, requested=None):
    if requested:
        if requested not in df.columns:
            raise BatchError(f"Column “{requested}” not found. Columns: {', '.join(map(str, df.columns))}")
        return requested
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in TEXT_COLUMN_NAMES:
        if name in by_lower:
            return by_lower[name]
    # Otherwise: the column with the longest text on average
    lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns}
    return max(lengths, key=lengths.get)


def pick_label_column(df, text_column):
    """A column whose values all look like spam/ham labels, or None."""
    by_lower = {str(c).strip().lower(): c for c in df.columns if c != text_column}
    for name in LABEL_COLUMN_NAMES:
        column = by_lower.get(name)
        if column is not None:
            values = df[column].astype(str).str.strip().str.lower()
            if len(values) and values.isin(_LABEL_VALUES).all():
                return column
    return None


def normalise_labels(series):
    return series.astype(str).str.strip().str.lower().map(_LABEL_VALUES)


def evaluate(labels, verdicts):
    tp = sum(1 for y, p in zip(labels, verdicts) if y == "spam" and p == "spam")
    fp = sum(1 for y, p in zip(labels, verdicts) if y == "ham" and p == "spam")
    fn = sum(1 for y, p in zip(labels, verdicts) if y == "spam" and p == "ham")
    tn = sum(1 for y, p in zip(labels, verdicts) if y == "ham" and p == "ham")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(labels), "accuracy": round((tp + tn) / len(labels), 4) if labels else None,
        "spam_precision": round(precision, 4), "spam_recall": round(recall, 4),
        "spam_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "confusion": [[tn, fp], [fn, tp]],
    }
