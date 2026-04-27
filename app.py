import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title("SVM Text Classification App")
st.write("Enter text to predict category")

text = st.text_area("Input Text")

if st.button("Predict"):
    if text.strip() != "":
        vector = tfidf.transform([text])
        prediction = model.predict(vector)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.warning("Please enter text")