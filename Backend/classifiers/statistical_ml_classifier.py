import joblib

from classifiers.text_utils import MODEL_PATH

_model = None


def load_model():
    """Loads the trained sklearn Pipeline (text cleaning + TF-IDF + classifier) once."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` "
                "from the Backend folder first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# Predicts based on the trained ML model (pass ml_model to use an unsaved one)
def predict_ml(custom_text, ml_model=None):
    model = ml_model or load_model()
    spam_index = list(model.classes_).index("spam")
    p_spam = float(model.predict_proba([custom_text])[0][spam_index])
    prediction = "spam" if p_spam >= 0.5 else "ham"
    confidence = p_spam if prediction == "spam" else 1 - p_spam

    return {
        "prediction": prediction,
        "spam_probability": round(p_spam * 100, 2),
        "confidence": round(confidence * 100, 2),
    }


# Tests the model with a ham and spam example: python -m classifiers.statistical_ml_classifier
if __name__ == "__main__":
    print(predict_ml("Hey, are we still meeting for lunch at 1 PM?"))
    print(predict_ml("URGENT! Click here to claim your free $500 cash prize now!"))
