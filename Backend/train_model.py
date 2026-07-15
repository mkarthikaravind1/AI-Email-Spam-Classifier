import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/cleaned_spam.csv")
# Renames the columns in the cleaned_spam.csv
df = df.rename(columns={"v1": "label","v2": "text"})

# Drops null/empty value rows
df = df.dropna(subset=["text"])

 # Separate features and labels
X = df["text"]
y = df["label"]

# Splits into 80% train, 20% test with stratification to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Converts text to lowercase and removes common English stop words (e.g. "the", "is")
tfidf = TfidfVectorizer(stop_words="english",lowercase=True)

# Fits TF-IDF on training data and transforms it into feature vectors
X_train_tfidf = tfidf.fit_transform(X_train)

# Trains a Multinomial Naive Bayes model on the TF-IDF features
model = MultinomialNB()
model.fit(X_train_tfidf,y_train)

# Saves trained model and vectorizer to disk for later use
joblib.dump(model,"saved_models/naive_bayes.pkl")
joblib.dump(tfidf,"saved_models/tfidf.pkl")

print("Model and vectorizer saved successfully!")