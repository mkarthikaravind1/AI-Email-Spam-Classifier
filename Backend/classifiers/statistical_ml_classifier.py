import joblib

# Loads the saved naive_bayes & tfidf 
model = joblib.load(
    "saved_models/naive_bayes.pkl"
)
tfidf = joblib.load(
    "saved_models/tfidf.pkl"
)

# Predicts based on the trained ML model
def predict_ml(custom_text):
    transformed_text = tfidf.transform(
        [custom_text]
    )
    prediction = model.predict(
        transformed_text
    )[0]
    probabilities = model.predict_proba(
        transformed_text
    )[0]

    confidence = max(probabilities) * 100

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }

# Tests the model with a ham and spam example
if __name__ == "__main__":
    print(predict_ml("Hey, are we still meeting for lunch at 1 PM?"))
    print(predict_ml("URGENT! Click here to claim your free $500 cash prize now!"))