Study Material Summarizer Bot

A lightweight yet powerful Streamlit-based application designed to help students make the most out of their study sessions.
The app extracts text from images, speech, or plain text and automatically generates:

Concise summaries

Multiple Choice Questions (MCQs)

Subjective questions for deeper understanding

Audio playback of generated content

This makes it a personal study assistant for quick revision and active learning.

Why this project?

Students often spend too much time reading and filtering notes.

This tool condenses large chunks of information into clear, usable knowledge.

It transforms study material into interactive quizzes and audio summaries, making revision faster and smarter.

Core Features
Input Sources

Upload an image of your notes (OCR-based text extraction)

Record your voice and let speech-to-text convert it into text

Paste your notes directly as text

Processing

Cleans and preprocesses extracted content

Summarizes text using TF-IDF and NLP techniques

Generates MCQs and subjective-type questions

Converts summaries and questions into speech for easy listening

Output

Clean, structured summaries

Auto-generated quizzes with answers

Audio playback of study material

Planned: Export as PDF for offline use

Technology Stack

Frontend & Framework: Streamlit

Optical Character Recognition: EasyOCR

Speech-to-Text: SpeechRecognition + Google Speech API

Natural Language Processing: NLTK, Scikit-learn (TF-IDF)

Text-to-Speech: gTTS

Visualization: Matplotlib, WordCloud

Export: FPDF

Installation Guide

Clone the repository:

git clone https://github.com/your-username/study-material-summarizer.git
cd study-material-summarizer


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

Running the App

Start the Streamlit server:

streamlit run app.py


Then open your browser and navigate to:

http://localhost:8501
