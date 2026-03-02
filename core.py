import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import csr_matrix

# =====================================================
# LOAD DATASETS (Reduced UCI version)
# =====================================================

uci_df = pd.read_csv(
    "/content/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)

uci_df = uci_df.sample(frac=0.5, random_state=42)

edu_df = pd.read_csv("/content/education_sms_dataset.csv")
edu_df = edu_df.rename(columns={"message": "text"})
edu_df = edu_df[["label", "text"]]

adv_df = pd.read_csv("/content/education_adversarial_dataset.csv")
adv_df = adv_df[["label", "text"]]

df = pd.concat([uci_df, edu_df, adv_df], ignore_index=True)
df = df.drop_duplicates(subset=["text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df = df.dropna()

# Rebalance 1:2
spam_df = df[df["label"] == 1]
ham_df = df[df["label"] == 0]
ham_sampled = ham_df.sample(n=min(len(spam_df)*2, len(ham_df)), random_state=42)
df = pd.concat([spam_df, ham_sampled]).sample(frac=1, random_state=42)

# =====================================================
# FEATURE ENGINEERING TRANSFORMER
# =====================================================

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

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# =====================================================
# COMBINE TF-IDF + STRUCTURED FEATURES
# =====================================================

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2
)

combined_features = FeatureUnion([
    ("tfidf", tfidf),
    ("structured", StructuredFeatures())
])

X_train_combined = combined_features.fit_transform(X_train)
X_test_combined = combined_features.transform(X_test)

# =====================================================
# TRAIN MODEL
# =====================================================

model = LogisticRegression(
    max_iter=600,
    class_weight="balanced"
)

model.fit(X_train_combined, y_train)

# =====================================================
# EVALUATE
# =====================================================

y_pred = model.predict(X_test_combined)

print("\n===== CLASSIFICATION REPORT =====\n")
print(classification_report(y_test, y_pred))

print("\n===== CONFUSION MATRIX =====\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# SAVE MODEL
# =====================================================

joblib.dump(model, "education_phishing_model_hybrid.pkl")
joblib.dump(combined_features, "hybrid_vectorizer.pkl")

print("\nHybrid model saved successfully.")
