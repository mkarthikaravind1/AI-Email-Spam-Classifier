"""Combines the individual classifier outputs into one verdict.

Instead of a plain majority vote (where a weak model counts as much as a
strong one), each model's spam probability is averaged with a weight equal to
its spam F1 score measured on validation data by train_model.py.
"""
from classifiers.text_utils import load_metrics

MODEL_NAMES = ("regex", "ml", "llm")


def load_weights():
    """Returns {model: weight} from metrics.json; equal weights if not trained yet."""
    models = load_metrics().get("models", {})
    return {name: models.get(name, {}).get("weight", 1.0) for name in MODEL_NAMES}


def combine(results, weights):
    """results: {model: result dict}. A result with status "error" is skipped.

    Returns the final verdict, the weighted spam probability, how many of the
    models that ran agree with the verdict, and the weights actually used.
    """
    ok = {name: r for name, r in results.items() if r.get("status", "ok") == "ok"}
    if not ok:
        return None

    used = {name: weights.get(name, 1.0) for name in ok}
    total = sum(used.values()) or 1.0
    p_spam = sum(used[name] * ok[name]["spam_probability"] for name in ok) / total
    verdict = "spam" if p_spam >= 50 else "ham"
    agree = sum(1 for r in ok.values() if r["prediction"] == verdict)

    return {
        "final_verdict": verdict,
        "spam_probability": round(p_spam, 2),
        "agreement": f"{agree}/{len(ok)}",
        "weights": {name: round(w / total, 3) for name, w in used.items()},
    }
