import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer (assuming they are already trained)
MODEL_FILE = "fake_news_model.pkl"
VECT_FILE = "tfidf_vectorizer.pkl"

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
except FileNotFoundError:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

# Streamlit App
st.title("📰 Fake News Detection System")
st.write("Paste a news article below to check if it's REAL or FAKE.")

news_input = st.text_area("Enter News Text:", height=200)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vec = vectorizer.transform([news_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        
        result = "REAL 🟢" if pred == 1 else "FAKE 🔴"
        confidence = round(max(prob) * 100, 2)
        
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence}%")

st.write("---")
st.write("Built with Machine Learning. Enjoy detecting fake news!")