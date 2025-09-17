import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
import io
from fpdf import FPDF
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from gtts import gTTS
import pandas as pd
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import logging
import re
import os
from datetime import datetime
import tempfile

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Study Material Summarizer", layout="wide")

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📚 Study Summarizer Bot")
st.sidebar.markdown("""
**👨‍🎓 AIM:** To assist students by automatically summarizing study materials using OCR and NLP.

**🔍 What It Solves:**
- Extracts text from images or audio for quick review
- Summarizes large texts into key points
- Generates quizzes for revision
- Outputs downloadable summaries
""")

# ---------------------- HEADER ----------------------
st.title("📖 Study Material Summarizer")
st.markdown("""
Welcome to the **Study Summarizer Bot**! Upload your **notes (image/text/audio)** and get:
- 📄 Summarized key points
- ❓ Auto-generated quizzes
- 📥 PDF download option
""")

# === Configure Logging ===
logging.basicConfig(
    filename='study_summarizer.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Starting Study Material Summarizer Bot")

# === Download NLTK Data ===
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_resources()

# === Helper Functions ===
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

def extract_text_from_image(image_file):
    simulated_text = "Error"
    return clean_text(simulated_text)

def extract_text_from_speech():
    st.info("🎤 Recording... Please speak now. You have full 60 seconds.")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            # Manually record 60 seconds of audio regardless of silence
            audio = recognizer.record(source, duration=60)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                with open(tmp_audio.name, "wb") as f:
                    f.write(audio.get_wav_data())
                st.audio(tmp_audio.name, format="audio/wav")  # Optional playback
        except Exception as e:
            st.error(f"🎙️ Recording failed: {e}")
            return ""

    try:
        text = recognizer.recognize_google(audio)
        st.success("✅ Transcription successful!")
        return clean_text(text)
    except sr.UnknownValueError:
        st.warning("⚠️ Could not understand your speech. Try again.")
        return ""
    except sr.RequestError:
        st.error("❌ Recognition service unavailable. Check internet.")
        return ""



def generate_summary(text):
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if len(word_tokenize(s)) > 5]
    if len(sentences) < 3:
        return ["Insufficient content to generate a summary."]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    top_indices = np.argsort(scores)[-5:][::-1]
    summary_points = [sentences[i] for i in top_indices]
    return summary_points

def generate_mcq(text):
    sentences = [s for s in sent_tokenize(text) if len(word_tokenize(s)) > 6]
    mcq = []
    for s in random.sample(sentences, min(5, len(sentences))):
        words = word_tokenize(s)
        content_words = [w for w in words if w.lower() not in stopwords.words('english') and w.isalpha() and len(w) > 3]
        if len(content_words) < 3:
            continue
        keyword = random.choice(content_words)
        question = re.sub(r'\b' + re.escape(keyword) + r'\b', "____", s, 1, flags=re.IGNORECASE)
        distractors = random.sample([w for w in content_words if w != keyword], k=3)
        options = [keyword] + distractors
        random.shuffle(options)
        mcq.append({"question": question, "options": options, "answer": keyword})
    return mcq

def generate_subjective(text):
    sentences = [s for s in sent_tokenize(text) if len(word_tokenize(s)) > 6]
    subj = []
    for s in random.sample(sentences, min(5, len(sentences))):
        words = word_tokenize(s)
        content_words = [w for w in words if w.lower() not in stopwords.words('english') and w.isalpha() and len(w) > 3]
        if content_words:
            keyword = random.choice(content_words)
            subj.append(f"Discuss the role of '{keyword}' in the context of the provided material.")
    return subj

# === Main Logic ===
text_source = st.sidebar.radio("Choose Input Type", ["📷 Image (OCR)", "📝 Text Input", "🎙 Speech Input"])

text = ""
if text_source == "📷 Image (OCR)":
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file:
        extracted_text = extract_text_from_image(image_file)
        st.subheader("Extracted Text")
        st.write(extracted_text)
        text = extracted_text

elif text_source == "📝 Text Input":
    user_text = st.text_area("Paste your study material here:", height=200)
    if st.button("Summarize"):
        text = user_text

elif text_source == "🎙 Speech Input":
    if st.button("Record Speech"):
        spoken_text = extract_text_from_speech()
        st.subheader("Transcribed Text")
        st.write(spoken_text)
        text = spoken_text

if text.strip():
    st.subheader("📌 Summary")
    summary_points = generate_summary(text)
    for point in summary_points:
        st.markdown(f"- {point}")

    st.subheader("📋 Multiple Choice Questions")
    mcqs = generate_mcq(text)
    for i, q in enumerate(mcqs):
        st.markdown(f"**Q{i+1}:** {q['question']}")
        for opt in q['options']:
            st.markdown(f"- {opt}")
        st.markdown(f"**Answer:** {q['answer']}")

    st.subheader("✍️ Subjective Questions")
    subjectives = generate_subjective(text)
    for i, q in enumerate(subjectives):
        st.markdown(f"**Q{i+1}:** {q}")

    # === Text-to-Speech Playback Function ===
    def speak_text(text, label):
        st.subheader(f"🔊 Listen to {label}")
        if st.button(f"▶️ Play {label}"):
            try:
                tts = gTTS(text)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format='audio/mp3')
            except Exception as e:
                st.error(f"Speech generation failed: {e}")

    speak_text(" ".join(summary_points), "Summary")

    mcq_text = "\n".join([f"Q{i+1}: {q['question']}. Options: {', '.join(q['options'])}. Answer: {q['answer']}." for i, q in enumerate(mcqs)])
    speak_text(mcq_text, "MCQs")

    subj_text = "\n".join(subjectives)
    speak_text(subj_text, "Subjective Questions")
