"""Shared paths and text cleaning used by both training and inference.

The dataset in data/cleaned_spam.csv was pre-cleaned (lowercased, punctuation
and NLTK English stop words removed). Live input arrives raw, so we apply the
same cleaning at inference time to avoid train/serve skew.
"""
import json
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BACKEND_DIR / "data" / "cleaned_spam.csv"
MODEL_DIR = BACKEND_DIR / "saved_models"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

# NLTK English stop-word list (inlined so NLTK is not a dependency)
STOP_WORDS = frozenset("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how
all any both each few more most other some such no nor not only own same so
than too very s t can will just don don't should should've now d ll m o re ve
y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn
hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn
wouldn't
""".split())

_APOSTROPHES = re.compile(r"['’]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_dataset():
    """Loads the labelled dataset as a DataFrame with `label` and `text` columns."""
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    df = df.dropna(subset=["text"])
    # Exact duplicates would leak between train and test splits
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def load_metrics():
    """Returns metrics.json written by train_model.py, or {} if it does not exist."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def clean_text(text):
    """Lowercases, strips punctuation and stop words, collapses whitespace."""
    text = _APOSTROPHES.sub("", str(text).lower())
    tokens = _NON_ALNUM.sub(" ", text).split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)
