import pandas as pd
from collections import Counter
import re
from sklearn.metrics import classification_report

df = pd.read_csv("data/cleaned_spam.csv")
df = df.rename(columns={
    "v1":"label",
    "v2":"text"
})

# Rules for deciding spam
SPAM_RULES = {
    "free_offer": r"\bfree\b",
    "win_prize": r"\b(win|winner|won|prize)\b",
    "urgent_action": r"\b(act now|limited time|urgent)\b",
    "money_offer": r"\b(cash|earn \$?\d+|guaranteed income)\b",
    "suspicious_links": r"(http[s]?://|www\.)",
    "claim_reward": r"\b(claim|reward|gift card)\b",
}

# Matches the text with the rules
def get_matched_rules(text):
    text = str(text).lower()
    matches = []

    for rule_name, pattern in SPAM_RULES.items():
        if re.search(pattern, text):
            matches.append(rule_name)

    return matches

# Weights assigned to each rule (higher = stronger spam signal)
RULE_WEIGHTS = {
    "free_offer": 1,
    "win_prize": 2,
    "urgent_action": 1,
    "money_offer": 3,
    "suspicious_links": 2,
    "claim_reward": 2
}

# Calculates the score for the no of matches
def spam_score(text):
    matches = get_matched_rules(text)
    return sum(RULE_WEIGHTS[r] for r in matches)

# Classifies the text based on the score
def classify_spam_weighted(text):
    score = spam_score(text)
    return "spam" if score >= 3 else "ham"

# Returns full prediction details including score, confidence, and matched rules
def predict_regex(text):
    matches = get_matched_rules(text)
    score = spam_score(text)
    prediction = classify_spam_weighted(text)
    confidence = min((score / 10) * 100, 100)

    return {
        "prediction": prediction,
        "score": score,
        "confidence": round(confidence, 2),
        "matched_rules": matches
    }

# Runs classifier on full dataset, prints report and top 50 spam words
if __name__ == "__main__":
    df["score"] = df["text"].apply(spam_score)
    df["prediction"] = df["text"].apply(classify_spam_weighted)

    print(
        classification_report(
            df["label"],
            df["prediction"]
        )
    )

    spam_text = df[df["label"] == "spam"]["text"]
    counter = Counter()

    for msg in spam_text:
        words = str(msg).lower().split()
        counter.update(words)

    print(counter.most_common(50))