import streamlit as st
import numpy as np
import pickle
from models.logistic_regression import LogisticRegression
from libs.sense_proc import TextPreprocessor

# Load vocab and idf
vocab = np.load("models_pretrained/vocab.npy", allow_pickle=True).tolist()
idf = np.load("models_pretrained/idf.npy", allow_pickle=True)

# Load trained model
with open("models_pretrained/logistic_regression.pkl", "rb") as f:
    model_params = pickle.load(f)

model = LogisticRegression()
model.W = model_params['weights']
model.B = model_params['bias']

# Setup preprocessor
preprocessor = TextPreprocessor()
preprocessor.vocab = vocab
preprocessor.idf_vector = idf

# Streamlit UI config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analyzer")
st.markdown("Analyze the **sentiment** of any text — determine whether it's **Positive 😊** or **Negative 😞**.")

# User input
user_input = st.text_area("Enter your sentence or paragraph:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and convert to TF-IDF
        processed_text = preprocessor.preprocess(user_input)
        tfidf_vector = preprocessor.tfidf([processed_text])

        # Predict sentiment
        prediction = model.predict(tfidf_vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

        # Display result
        st.subheader("🔍 Sentiment Result:")
        st.success(sentiment)

