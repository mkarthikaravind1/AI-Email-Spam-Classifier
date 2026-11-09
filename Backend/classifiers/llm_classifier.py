from transformers import pipeline

# Zero-shot NLI model: it scores how well "This text is {label}." fits the input.
# It does not know the spam-filter jargon "ham", so we give it descriptive
# labels and map them back to spam/ham. ("spam" vs "ham" scored ~50% accuracy
# on a balanced sample; these labels scored ~74% with ~82% spam recall.)
LABEL_MAP = {
    "a promotion or prize offer": "spam",
    "a normal personal message": "ham",
}
HYPOTHESIS_TEMPLATE = "This text is {}."

print("Loading the zero-shot model from Hugging Face...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# Predicts spam/ham label and confidence score for a given text
def predict_llm(text):
    result = classifier(
        text,
        candidate_labels=list(LABEL_MAP),
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )

    return {
        "prediction": LABEL_MAP[result["labels"][0]],
        "confidence": round(result["scores"][0] * 100, 2),
    }


# Evaluates on a balanced sample: python -m classifiers.llm_classifier
if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import classification_report
    from tqdm import tqdm

    from classifiers.text_utils import load_dataset

    df = load_dataset()
    sample = pd.concat([
        df[df["label"] == "spam"].sample(50, random_state=0),
        df[df["label"] == "ham"].sample(50, random_state=0),
    ])

    print("Classifying messages with the zero-shot model...")
    sample["prediction"] = [predict_llm(t)["prediction"] for t in tqdm(sample["text"])]
    print(classification_report(sample["label"], sample["prediction"], digits=3))
