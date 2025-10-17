import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.title("📰 Fake News Detection")

@st.cache_data
def load_data():
    df = pd.read_csv("news.csv")
    df = df.dropna()
    return df

df = load_data()

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

def predict_news(text):
    vector = vectorizer.transform([text])
    pred = model.predict(vector)
    return "Fake" if pred[0] == 1 else "Real"

st.subheader("Enter News Text:")
user_input = st.text_area("Type or paste news here:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_news(user_input)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter some text to predict.")