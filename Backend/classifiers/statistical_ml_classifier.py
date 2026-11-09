import joblib

from classifiers.text_utils import MODEL_PATH

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` "
        "from the Backend folder first."
    )

# A single sklearn Pipeline: text cleaning + TF-IDF + classifier
model = joblib.load(MODEL_PATH)


# Predicts based on the trained ML model
def predict_ml(custom_text):
    probabilities = model.predict_proba([custom_text])[0]
    best = probabilities.argmax()

    return {
        "prediction": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]) * 100, 2),
    }


# Tests the model with a ham and spam example
if __name__ == "__main__":
    print(predict_ml("Hey, are we still meeting for lunch at 1 PM?"))
    print(predict_ml("URGENT! Click here to claim your free $500 cash prize now!"))
