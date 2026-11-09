"""Feedback loop storage (feature #5), in a local SQLite file.

- feedback: messages a user labelled ("this was actually spam")
- queue:    messages the models were unsure about or disagreed on, waiting for
            a label (active learning: label the most informative ones first)

Everything stays on this machine in Backend/data/feedback.db.
"""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from classifiers.text_utils import BACKEND_DIR

DB_PATH = BACKEND_DIR / "data" / "feedback.db"
_lock = threading.Lock()


@contextmanager
def _connect():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init():
    DB_PATH.parent.mkdir(exist_ok=True)
    with _connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL CHECK (label IN ('spam', 'ham')),
                predicted TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                spam_probability REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def add_feedback(text, label, predicted=None):
    """Stores (or updates) a user label and removes the message from the queue."""
    with _connect() as db:
        db.execute(
            "INSERT INTO feedback (text, label, predicted, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(text) DO UPDATE SET label = excluded.label, created_at = excluded.created_at",
            (text, label, predicted, _now()),
        )
        db.execute("DELETE FROM queue WHERE text = ?", (text,))


def queue_if_uncertain(text, spam_probability, predictions):
    """Adds a message to the labelling queue if it is informative to label.

    Returns the reason, or None if it was not queued.
    """
    reasons = []
    if 30 <= spam_probability <= 70:
        reasons.append(f"uncertain ({spam_probability:.0f}% spam)")
    if len(set(predictions)) > 1:
        reasons.append("models disagree")
    if not reasons:
        return None
    with _connect() as db:
        if db.execute("SELECT 1 FROM feedback WHERE text = ?", (text,)).fetchone():
            return None  # already labelled
        db.execute(
            "INSERT INTO queue (text, spam_probability, reason, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(text) DO UPDATE SET spam_probability = excluded.spam_probability, "
            "reason = excluded.reason",
            (text, spam_probability, ", ".join(reasons), _now()),
        )
    return ", ".join(reasons)


def get_queue(limit=50):
    """Most informative first: closest to 50% spam probability."""
    with _connect() as db:
        rows = db.execute(
            "SELECT id, text, spam_probability, reason, created_at FROM queue "
            "ORDER BY ABS(spam_probability - 50) ASC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def dismiss(queue_id):
    with _connect() as db:
        db.execute("DELETE FROM queue WHERE id = ?", (queue_id,))


def all_feedback():
    with _connect() as db:
        return [dict(r) for r in db.execute("SELECT text, label, predicted, created_at FROM feedback")]


def stats():
    with _connect() as db:
        total = db.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        by_label = dict(db.execute("SELECT label, COUNT(*) FROM feedback GROUP BY label").fetchall())
        corrections = db.execute(
            "SELECT COUNT(*) FROM feedback WHERE predicted IS NOT NULL AND predicted != label"
        ).fetchone()[0]
        queued = db.execute("SELECT COUNT(*) FROM queue").fetchone()[0]
    return {"total": total, "spam": by_label.get("spam", 0), "ham": by_label.get("ham", 0),
            "corrections": corrections, "queued": queued}
