import re

import joblib
import numpy as np

from classifiers.text_utils import MODEL_PATH

_model = None


def load_model(force=False):
    """Loads the trained sklearn Pipeline (text cleaning + TF-IDF + classifier) once."""
    global _model
    if _model is None or force:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` "
                "from the Backend folder first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def spam_probabilities(texts, ml_model=None):
    model = ml_model or load_model()
    spam_index = list(model.classes_).index("spam")
    return model.predict_proba(list(texts))[:, spam_index]


# Predicts based on the trained ML model (pass ml_model to use an unsaved one)
def predict_ml(custom_text, ml_model=None):
    p_spam = float(spam_probabilities([custom_text], ml_model)[0])
    prediction = "spam" if p_spam >= 0.5 else "ham"
    confidence = p_spam if prediction == "spam" else 1 - p_spam

    return {
        "prediction": prediction,
        "spam_probability": round(p_spam * 100, 2),
        "confidence": round(confidence * 100, 2),
    }


_WORD = re.compile(r"[\w£$€@']+")
MAX_EXPLAINED_WORDS = 120


def explain_ml(text, ml_model=None):
    """Word-level explanation by occlusion (feature #1).

    Each distinct word is removed in turn and the message re-scored; the drop
    in spam probability is that word's contribution (positive = pushed towards
    spam, negative = pushed towards ham). Works for any model, and for this
    linear model it costs one batched prediction.
    """
    matches = list(_WORD.finditer(text))
    words = list(dict.fromkeys(m.group(0).lower() for m in matches))[:MAX_EXPLAINED_WORDS]
    if not words:
        return []

    variants = [text]
    for word in words:
        variants.append(re.sub(rf"(?<![\w£$€@']){re.escape(word)}(?![\w£$€@'])", " ", text, flags=re.I))
    # Log-odds instead of probability: on a 99.5%-spam message removing one word
    # barely moves the probability, but its log-odds effect is still clear.
    probs = np.clip(spam_probabilities(variants, ml_model), 1e-6, 1 - 1e-6)
    logits = np.log(probs / (1 - probs))
    contribution = {w: round(float(logits[0] - lo), 3) for w, lo in zip(words, logits[1:])}
    biggest = max(abs(c) for c in contribution.values()) or 1.0

    return [
        {"start": m.start(), "end": m.end(), "contribution": contribution[m.group(0).lower()]}
        for m in matches
        if m.group(0).lower() in contribution
        and abs(contribution[m.group(0).lower()]) >= 0.05 * biggest
    ]


# Tests the model with a ham and spam example: python -m classifiers.statistical_ml_classifier
if __name__ == "__main__":
    print(predict_ml("Hey, are we still meeting for lunch at 1 PM?"))
    print(predict_ml("URGENT! Click here to claim your free $500 cash prize now!"))
    print(explain_ml("URGENT! Click here to claim your free $500 cash prize now!"))
