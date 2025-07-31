import streamlit as st
import nltk
nltk.data.path.append("nltk_data")

import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download once
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess
def preprocess(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [w for w in words if w not in stop_words and w not in string.punctuation]
        cleaned.append(" ".join(words))

    return sentences, cleaned

# Get best matching response
def get_most_relevant_sentence(user_input, cleaned_sentences, original_sentences):
    vectorizer = TfidfVectorizer()
    corpus = cleaned_sentences + [user_input]

    try:
        vectors = vectorizer.fit_transform(corpus)
    except ValueError:
        return "Sorry, I didn't understand that."

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    # If all similarities are zero, return a fallback
    if similarity.max() == 0:
        return "Sorry, I couldn't find a relevant answer."

    idx = similarity.argmax()
    return original_sentences[idx]

# Load FAQ file
with open("customer_faq.txt", "r", encoding="utf-8") as f:
    text = f.read()

original_sentences, cleaned_sentences = preprocess(text)

# Streamlit UI
st.title("📦 Customer Support Chatbot")
st.write("Ask me a question about your order, refund, shipping, etc.")

user_input = st.text_input("You:")

if user_input:
    response = get_most_relevant_sentence(user_input, cleaned_sentences, original_sentences)
    st.success(f"💬 {response}")
