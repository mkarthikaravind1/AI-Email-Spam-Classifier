"""Fine-tunes DistilBERT as the 4th spam classifier (feature #7).

    python train_transformer.py            # ~10-20 min on a laptop CPU

Uses the same 80/20 train/test split as train_model.py, so the test numbers
are comparable. It also produces out-of-fold predictions on the training split
(2-fold), which train_model.py uses to train the stacking ensemble without
leaking test data.

Saves:
  saved_models/distilbert/              - fine-tuned model + tokenizer
  saved_models/distilbert_preds.json    - out-of-fold + test predictions, metrics

Run train_model.py afterwards to rebuild the ensemble and benchmark.
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifiers.text_utils import MODEL_DIR, clean_text, load_dataset

BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = MODEL_DIR / "distilbert"
PREDS_PATH = MODEL_DIR / "distilbert_preds.json"
MAX_LEN = 64  # SMS messages are short; 64 tokens covers ~99% of them


def train(texts, labels, epochs, lr=5e-5, batch_size=32, seed=42):
    """Fine-tunes a fresh DistilBERT on (texts, labels: 0=ham, 1=spam)."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Spam is only ~13% of the data, so up-weight its loss
    labels = np.asarray(labels)
    class_weights = torch.tensor(len(labels) / (2 * np.bincount(labels, minlength=2)), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    texts = [clean_text(t) for t in texts]
    rng = np.random.default_rng(seed)
    steps = epochs * int(np.ceil(len(texts) / batch_size))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, total_iters=steps)

    model.train()
    start, step = time.perf_counter(), 0
    for epoch in range(epochs):
        order = rng.permutation(len(texts))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = tokenizer([texts[j] for j in idx], truncation=True, max_length=MAX_LEN,
                              padding=True, return_tensors="pt")
            loss = loss_fn(model(**batch).logits, torch.tensor(labels[idx]))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0 or step == steps:
                print(f"  epoch {epoch + 1}/{epochs} step {step}/{steps} "
                      f"loss {loss.item():.4f} ({time.perf_counter() - start:.0f}s)", flush=True)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_proba(model, tokenizer, texts, batch_size=64):
    """Returns P(spam) for each text."""
    texts = [clean_text(t) for t in texts]
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt")
        probs.extend(torch.softmax(model(**batch).logits, dim=-1)[:, 1].tolist())
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    y_train_num = (y_train == "spam").astype(int).to_numpy()

    # 1) Out-of-fold predictions on the training split, for the stacking ensemble
    oof = {}
    folds = StratifiedKFold(2, shuffle=True, random_state=42)
    for k, (fit_idx, pred_idx) in enumerate(folds.split(X_train, y_train_num), 1):
        print(f"\nFold {k}/2: training on {len(fit_idx)} messages")
        model, tokenizer = train(X_train.iloc[fit_idx].tolist(), y_train_num[fit_idx], args.epochs)
        probs = predict_proba(model, tokenizer, X_train.iloc[pred_idx].tolist())
        oof.update({int(i): round(p, 5) for i, p in zip(X_train.index[pred_idx], probs)})
    oof_f1 = f1_score(y_train, ["spam" if oof[i] >= 0.5 else "ham" for i in X_train.index], pos_label="spam")
    print(f"\nOut-of-fold spam F1 on the training split: {oof_f1:.4f}")

    # 2) Final model on the whole training split, evaluated on the test split
    print(f"\nFinal model: training on {len(X_train)} messages")
    model, tokenizer = train(X_train.tolist(), y_train_num, args.epochs)
    start = time.perf_counter()
    test_probs = predict_proba(model, tokenizer, X_test.tolist())
    ms_per_msg = (time.perf_counter() - start) / len(X_test) * 1000
    test_pred = ["spam" if p >= 0.5 else "ham" for p in test_probs]
    print(classification_report(y_test, test_pred, digits=4))
    r = classification_report(y_test, test_pred, output_dict=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    PREDS_PATH.write_text(json.dumps({
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "oof_spam_f1": round(oof_f1, 4),
        "test_metrics": {
            "accuracy": round(r["accuracy"], 4),
            "spam_precision": round(r["spam"]["precision"], 4),
            "spam_recall": round(r["spam"]["recall"], 4),
            "spam_f1": round(r["spam"]["f1-score"], 4),
            "n": len(y_test),
        },
        "batch_ms_per_message": round(ms_per_msg, 2),
        "oof": oof,
        "test": {int(i): round(p, 5) for i, p in zip(X_test.index, test_probs)},
    }))
    print(f"Saved model to {OUT_DIR} and predictions to {PREDS_PATH}")


if __name__ == "__main__":
    main()
