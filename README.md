Study Material Summarizer Bot

An AI-powered Streamlit web app that helps students quickly summarize study materials, generate quizzes, and listen to audio playback.
It supports OCR (image-based text extraction), speech-to-text transcription, and manual text input, making it a versatile learning assistant.

Features

OCR Extraction: Extracts text from uploaded study notes (images).

Speech-to-Text: Record your voice and convert it into text.

Text Summarization: Auto-generates concise key points using NLP + TF-IDF.

MCQ Generator: Creates multiple-choice questions with answers for quick revision.

Subjective Questions: Generates descriptive questions for deeper learning.

Text-to-Speech: Listen to summaries, MCQs, and subjective questions.

Downloadable PDF (planned) for offline use.

Word Cloud & Visualization (optional extension) for better comprehension.

Tech Stack

Frontend & Framework: Streamlit

OCR: EasyOCR

Speech Recognition: SpeechRecognition + Google Speech API

NLP & Summarization: NLTK + TF-IDF (Scikit-learn)

Text-to-Speech: gTTS (Google Text-to-Speech)

Visualization: Matplotlib, WordCloud

Export: FPDF (for future PDF support)

Installation

Clone the repository:

git clone https://github.com/your-username/study-material-summarizer.git
cd study-material-summarizer


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


Install dependencies:

pip install -r requirements.txt


Make sure NLTK resources are downloaded (first run will auto-download).

Usage

Run the Streamlit app:

streamlit run app.py


Then open in your browser:

http://localhost:8501

Project Structure
study-material-summarizer/
│── app.py                # Main Streamlit app
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── study_summarizer.log   # Log file (auto-generated)

Future Enhancements

Save summarized notes & quizzes as downloadable PDF

Multi-language OCR and summarization support

Integration with LLMs for advanced question generation
