import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("LD DA 1.csv")

print("Columns in CSV:", data.columns)

# Remove TRUE NaN
data = data.dropna(subset=["selftext", "title", "Label"])

# Convert to string
data["selftext"] = data["selftext"].astype(str)
data["title"] = data["title"].astype(str)
data["Label"] = data["Label"].astype(str)

# Remove garbage rows: those containing "nan"
data = data[~data["selftext"].str.lower().str.contains("nan")]
data = data[~data["title"].str.lower().str.contains("nan")]

# Combine text
data["text"] = data["title"] + " " + data["selftext"]

# Remove rows with empty text
data = data[data["text"].str.strip() != ""]

# Extract features and labels
text_data = data["text"]
labels = data["Label"]

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(text_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train SVM model with balanced weights
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

# Predict sample outputs
pred = model.predict(X_test[:10])
print("\nPredictions:", pred)

# Calculate accuracy
pred_full = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, pred_full))
import pickle

import pickle

print("Saving model...")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model saved successfully!")
