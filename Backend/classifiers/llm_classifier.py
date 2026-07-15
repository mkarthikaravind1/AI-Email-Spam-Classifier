import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv("data/cleaned_spam.csv")
df = df.rename(columns={"v1": "label", "v2": "text"}).dropna(subset=["text"])

# Grab 20 rows for a quick test
df_sample = df.head(20).copy()

# Initialize the Pipeline
print("Loading the model from Hugging Face...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Labels for classification
candidate_labels = ["spam", "ham"]
# Stores predictions for each message
predictions = []

# Predicts spam/ham label and confidence score for a given text
def predict_llm(text):
    result = classifier(
        text,
        candidate_labels=["spam", "ham"]
    )
    prediction = result["labels"][0]
    confidence = result["scores"][0] * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }

# Runs classification on sample and prints accuracy
if __name__ == "__main__":
    print("Classifying messages with the LLM...")

    for text in tqdm(df_sample["text"]):
        result = classifier(text, candidate_labels=candidate_labels)
        top_prediction = result["labels"][0]
        predictions.append(top_prediction)
    df_sample["prediction"] = predictions

    print("\n--- LLM Classifier Results (Sample) ---")
    print(df_sample[["text", "label", "prediction"]].head(10))

    acc = accuracy_score(df_sample["label"], df_sample["prediction"])
    print(f"\nSample Accuracy: {acc:.2%}")