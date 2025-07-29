import joblib
import streamlit as st
import joblib
import numpy as np
from data_training import get_embedding

joblib.dump('stock_prediction_agent/model/clf.pkl')

model = joblib.load("model/clf.pkl")

st.set_page_config(page_title="Stock Movement Predictor", layout="centered")
st.title("News Based Stock Movement Prediction")

st.markdown("Enter a news headline and get prediction on stock price movement over the next few days.")

headline = st.text_area("Enter News Headline")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        embedding = get_embedding(headline).reshape(1, -1)
        prediction = model.predict(embedding)[0]
        proba = model.predict_proba(embedding)[0][prediction]

        if prediction == 1:
            st.success(f"Prediction: Price Likely to go UP (Confidence: {proba:.2f})")
        else:
            st.error(f"Prediction: Price Likely to go DOWN (Confidence: {proba:.2f})")
