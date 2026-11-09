import re

from classifiers.text_utils import load_metrics

# Rules for deciding spam: (pattern, weight). Higher weight = stronger signal.
# Patterns run on lowercased text and are written to work both on raw input
# and on the pre-cleaned dataset (no punctuation, no stop words).
SPAM_RULES = {
    # "free entry", "free ringtone", "free 500"... but not "feel free" / "free time"
    "free_offer": (r"\bfree\s+(entry|gift|offer|prize|cash|ringtones?|tones?|msgs?|texts?|txts?|"
                   r"calls?|camera|phone|mobile|video|sms|access|trial|week|[£$€]?\d+)", 2),
    # \bwon\b would also match "won't" — the lookahead excludes that
    "win_prize": (r"\b(win|winner|winning|won(?!')|prize|awarded|selected)\b", 2),
    "urgent_action": (r"\b(act now|limited time|urgent|click here|last chance|expires?)\b", 1),
    "money_offer": (r"[£$€]\s?\d+|\b\d+\s?(pounds|gbp)\b|\b(cash|guaranteed income|earn \$?\d+)\b", 2),
    "suspicious_links": (r"(https?://|www\.)", 2),
    "claim_reward": (r"\b(claim|reward|gift card|voucher|bonus)\b", 2),
    # "txt WIN to 87121", "reply stop 86688"
    "sms_shortcode": (r"\b(txt|text|reply|send|sms)\b.{0,25}?\b\d{5}\b", 2),
    # UK premium-rate / freephone numbers, e.g. 09061701461, 08452810075
    "premium_number": (r"\b0[789]\d{8,9}", 3),
    "subscription": (r"\b(subscri\w*|unsubscribe|opt ?out|ringtones?|per (msg|min|wk|week)|\d+p\b|"
                     r"18\+|over18s?|t ?& ?c|tcs?)\b", 2),
    "mobile_offer": (r"\b(mobile|phone|camera)\b.{0,30}?\b(update|upgrade|latest|colour|free)\b", 1),
}

_COMPILED = {name: re.compile(pattern) for name, (pattern, _) in SPAM_RULES.items()}

# Fallback score -> spam probability table, used only until train_model.py has
# measured the real one from the training data and saved it to metrics.json.
_DEFAULT_CALIBRATION = [0.02, 0.5, 0.65, 0.95]


# Matches the text with the rules
def get_matched_rules(text):
    text = str(text).lower()
    return [name for name, pattern in _COMPILED.items() if pattern.search(text)]


# Calculates the weighted score for a list of matched rules
def spam_score(matches):
    return sum(SPAM_RULES[name][1] for name in matches)


def fit_calibration(scores, labels):
    """Learns P(spam | score) from labelled data.

    Index i of the returned list is the spam probability for score i; scores
    beyond the end use the last value. Laplace smoothing keeps rare scores away
    from 0%/100%, and a running max makes the table never decrease (a higher
    score can't mean "less spammy").
    """
    max_score = max(scores)
    table, running_max = [], 0.0
    for s in range(max_score + 1):
        n = sum(1 for x in scores if x == s)
        spam = sum(1 for x, y in zip(scores, labels) if x == s and y == "spam")
        p = (spam + 1) / (n + 2)
        running_max = max(running_max, p)
        table.append(round(running_max, 4))
    return table


_CALIBRATION = load_metrics().get("regex_calibration", _DEFAULT_CALIBRATION)


def spam_probability(score, calibration=None):
    table = calibration or _CALIBRATION
    return table[min(score, len(table) - 1)]


# Returns full prediction details including score, probability and matched rules
def predict_regex(text, calibration=None):
    matches = get_matched_rules(text)
    score = spam_score(matches)
    p_spam = spam_probability(score, calibration)
    prediction = "spam" if p_spam >= 0.5 else "ham"
    # Confidence is in the *predicted* label, same meaning as for the other models
    confidence = p_spam if prediction == "spam" else 1 - p_spam

    return {
        "prediction": prediction,
        "spam_probability": round(p_spam * 100, 2),
        "confidence": round(confidence * 100, 2),
        "score": score,
        "matched_rules": matches,
    }


# Evaluates the rules on the dataset: python -m classifiers.regex_classifier
if __name__ == "__main__":
    from collections import Counter

    from sklearn.metrics import classification_report

    from classifiers.text_utils import load_dataset

    df = load_dataset()
    df["prediction"] = df["text"].apply(lambda t: predict_regex(t)["prediction"])
    print("Calibration table (score -> P(spam)):", _CALIBRATION)
    print(classification_report(df["label"], df["prediction"], digits=3))

    false_alarms = df[(df["label"] == "ham") & (df["prediction"] == "spam")]
    print(f"\nHam messages wrongly flagged ({len(false_alarms)}):")
    for text in false_alarms["text"].head(15):
        print(f"  {get_matched_rules(text)}  {text[:90]}")

    counter = Counter()
    for msg in df[df["label"] == "spam"]["text"]:
        counter.update(str(msg).lower().split())
    print("\nTop 50 spam words:", counter.most_common(50))
