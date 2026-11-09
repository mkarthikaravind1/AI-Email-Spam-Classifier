import re

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

SPAM_THRESHOLD = 3
_COMPILED = {name: re.compile(pattern) for name, (pattern, _) in SPAM_RULES.items()}


# Matches the text with the rules
def get_matched_rules(text):
    text = str(text).lower()
    return [name for name, pattern in _COMPILED.items() if pattern.search(text)]


# Calculates the weighted score for a list of matched rules
def spam_score(matches):
    return sum(SPAM_RULES[name][1] for name in matches)


# Classifies the text based on the score
def classify_spam_weighted(text):
    return "spam" if spam_score(get_matched_rules(text)) >= SPAM_THRESHOLD else "ham"


# Returns full prediction details including score, confidence, and matched rules
def predict_regex(text):
    matches = get_matched_rules(text)
    score = spam_score(matches)
    prediction = "spam" if score >= SPAM_THRESHOLD else "ham"
    confidence = min((score / 10) * 100, 100)

    return {
        "prediction": prediction,
        "score": score,
        "confidence": round(confidence, 2),
        "matched_rules": matches,
    }


# Evaluates the rules on the dataset: python -m classifiers.regex_classifier
if __name__ == "__main__":
    from collections import Counter

    from sklearn.metrics import classification_report

    from classifiers.text_utils import load_dataset

    df = load_dataset()
    df["prediction"] = df["text"].apply(classify_spam_weighted)
    print(classification_report(df["label"], df["prediction"], digits=3))

    false_alarms = df[(df["label"] == "ham") & (df["prediction"] == "spam")]
    print(f"\nHam messages wrongly flagged ({len(false_alarms)}):")
    for text in false_alarms["text"].head(15):
        print(f"  {get_matched_rules(text)}  {text[:90]}")

    counter = Counter()
    for msg in df[df["label"] == "spam"]["text"]:
        counter.update(str(msg).lower().split())
    print("\nTop 50 spam words:", counter.most_common(50))
