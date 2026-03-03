import streamlit as st
import joblib
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix

class StructuredFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        urgency_words = ["urgent", "final", "deadline", "expires", "immediate", "verify"]

        for text in X:
            text_lower = text.lower()

            num_links = len(re.findall(r"http", text_lower))
            has_short = int(any(x in text_lower for x in ["bit.ly", "tinyurl", "t.co"]))
            has_edu = int(any(x in text_lower for x in [".edu", ".ac.in", ".ac.uk"]))
            has_gov = int(".gov" in text_lower)
            urgency_count = sum(word in text_lower for word in urgency_words)
            suspicious_domain = int(any(x in text_lower for x in ["-secure", "-verify", "-validation"]))
            exclamations = text.count("!")
            length = len(text)

            features.append([
                num_links,
                has_short,
                has_edu,
                has_gov,
                urgency_count,
                suspicious_domain,
                exclamations,
                length
            ])

        return csr_matrix(np.array(features))

@st.cache_resource
def load_objects():
    model = joblib.load("education_phishing_model_hybrid.pkl")
    vectorizer = joblib.load("hybrid_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_objects()

st.title("Education Phishing Detector")

user_input = st.text_area("Enter message")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter text")
    else:
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"⚠️ PHISHING (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ SAFE (Confidence: {(1-probability):.2%})")
