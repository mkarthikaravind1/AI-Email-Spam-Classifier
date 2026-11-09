"""Combines the individual classifier outputs into one verdict.

Two methods (selectable in the UI):

- "stacking" (default, feature #6): a small logistic regression trained by
  train_model.py on the models' out-of-fold spam probabilities. It learns how
  much to trust each model, and in which direction, from data.
- "weighted": weighted average of spam probabilities, weight = each model's
  validation spam F1.

Two stackers are trained: "fast" uses only the fast models, and "full" also
uses the slow zero-shot model. The cascade (feature #2) runs the fast models
first and only calls the slow model when the fast verdict is uncertain.
"""
import math

from classifiers.text_utils import load_metrics

MODEL_NAMES = ("regex", "ml", "distilbert", "llm")
FAST_MODELS = ("regex", "ml", "distilbert")
SLOW_MODELS = ("llm",)
DEFAULT_CASCADE = {"low": 15.0, "high": 85.0}

_config = {}


def reload_config():
    """Re-reads metrics.json (after training or retraining)."""
    global _config
    metrics = load_metrics()
    models = metrics.get("models", {})
    _config = {
        "weights": {name: models.get(name, {}).get("weight", 1.0) for name in MODEL_NAMES},
        "stacking": metrics.get("stacking", {}),
        "cascade": metrics.get("cascade", DEFAULT_CASCADE),
    }
    return _config


def config():
    return _config or reload_config()


def logit(p_percent):
    p = min(max(p_percent / 100, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _ok(results):
    return {name: r for name, r in results.items() if r.get("status", "ok") == "ok"}


def _summary(p_spam, ok, method, detail):
    verdict = "spam" if p_spam >= 50 else "ham"
    agree = sum(1 for r in ok.values() if r["prediction"] == verdict)
    return {
        "method": method,
        "final_verdict": verdict,
        "spam_probability": round(p_spam, 2),
        "agreement": f"{agree}/{len(ok)}",
        **detail,
    }


def combine_weighted(results, weights=None):
    ok = _ok(results)
    if not ok:
        return None
    weights = weights or config()["weights"]
    used = {name: weights.get(name, 1.0) for name in ok}
    total = sum(used.values()) or 1.0
    p_spam = sum(used[name] * ok[name]["spam_probability"] for name in ok) / total
    return _summary(p_spam, ok, "weighted",
                    {"weights": {name: round(w / total, 3) for name, w in used.items()}})


def _pick_stacker(ok, stacking):
    """The full stacker if all its models ran, else the fast one, else None."""
    for key in ("full", "fast"):
        stacker = stacking.get(key)
        if stacker and all(name in ok for name in stacker["features"]):
            return key, stacker
    return None, None


def combine_stacking(results, stacking=None):
    ok = _ok(results)
    if not ok:
        return None
    stacking = stacking if stacking is not None else config()["stacking"]
    key, stacker = _pick_stacker(ok, stacking)
    if stacker is None:
        # Not trained yet, or a model it needs failed: fall back to weighted vote
        fallback = combine_weighted(results)
        fallback["method"] = "weighted (stacking unavailable)"
        return fallback

    z = stacker["intercept"]
    contributions = {}
    for name, coef in zip(stacker["features"], stacker["coef"]):
        contributions[name] = coef * logit(ok[name]["spam_probability"])
        z += contributions[name]
    p_spam = 100 / (1 + math.exp(-z))
    return _summary(p_spam, ok, f"stacking ({key})", {
        # Learned coefficient per model, and how much it pushed this verdict
        # (in log-odds; positive = towards spam)
        "coefficients": dict(zip(stacker["features"], stacker["coef"])),
        "contributions": {k: round(v, 3) for k, v in contributions.items()},
        "ignored": sorted(set(ok) - set(stacker["features"])),
    })


def combine(results, method="stacking"):
    return combine_stacking(results) if method == "stacking" else combine_weighted(results)


def needs_slow_models(fast_verdict, cascade=None):
    """Cascade rule: escalate only if the fast models' verdict is uncertain."""
    cascade = cascade or config()["cascade"]
    return cascade["low"] <= fast_verdict["spam_probability"] <= cascade["high"]
