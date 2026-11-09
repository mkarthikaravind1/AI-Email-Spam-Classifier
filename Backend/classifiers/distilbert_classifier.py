"""Fine-tuned DistilBERT spam classifier (feature #7).

Train it with `python train_transformer.py`. If it hasn't been trained, the
app simply runs without it.
"""
import threading

from classifiers.text_utils import MODEL_DIR, clean_text

MODEL_PATH = MODEL_DIR / "distilbert"
MAX_LEN = 64

_lock = threading.Lock()
_model = None
_tokenizer = None


def available():
    return (MODEL_PATH / "config.json").exists()


def _load():
    global _model, _tokenizer
    with _lock:
        if _model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval()
    return _model, _tokenizer


def spam_probabilities(texts, batch_size=64):
    import torch

    model, tokenizer = _load()
    texts = [clean_text(t) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                              padding=True, return_tensors="pt")
            probs.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].tolist())
    return probs


def predict_distilbert(text):
    p_spam = spam_probabilities([text])[0]
    prediction = "spam" if p_spam >= 0.5 else "ham"
    return {
        "prediction": prediction,
        "spam_probability": round(p_spam * 100, 2),
        "confidence": round(max(p_spam, 1 - p_spam) * 100, 2),
    }
