import streamlit as st
import pickle
import re

# Load trained model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection System")

st.write("Paste a news article below and click **Check News**.")

news_text = st.text_area("Enter News Text")

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned_text = clean_text(news_text)
        vector = tfidf.transform([cleaned_text])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")
