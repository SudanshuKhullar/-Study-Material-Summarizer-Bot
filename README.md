readme_content = """
# Study Material Summarizer Bot  

**Study Material Summarizer Bot** is a lightweight yet powerful Streamlit-based application designed to help students make the most out of their study sessions.  

The app extracts text from images, speech, or plain text and automatically generates:  
- Concise summaries  
- Multiple Choice Questions (MCQs)  
- Subjective questions for deeper understanding  
- Audio playback of generated content  

It acts like a **personal study assistant** for quick revision and active learning.  

---

## Why This Project?  

Students often spend too much time reading and filtering notes.  
This tool helps by:  
- Condensing large chunks of information into clear, usable knowledge  
- Transforming study material into interactive quizzes  
- Converting notes into audio summaries for faster revision  

---

## Core Features  

### Input Sources  
- Upload an image of your notes (OCR-based text extraction)  
- Record your voice and let speech-to-text convert it into text  
- Paste your notes directly as plain text  

### Processing  
- Cleans and preprocesses extracted content  
- Summarizes text using TF-IDF and NLP techniques  
- Generates MCQs and subjective-type questions  
- Converts summaries and questions into speech  

### Output  
- Clean, structured summaries  
- Auto-generated quizzes with answers  
- Audio playback of study material  
- Planned: Export as PDF for offline use  

---

## Technology Stack  

- **Frontend & Framework**: Streamlit  
- **Optical Character Recognition**: EasyOCR  
- **Speech-to-Text**: SpeechRecognition + Google Speech API  
- **Natural Language Processing**: NLTK, Scikit-learn (TF-IDF)  
- **Text-to-Speech**: gTTS  
- **Visualization**: Matplotlib, WordCloud  
- **Export**: FPDF  

---

## Installation Guide  

Clone the repository:  
```bash
git clone https://github.com/your-username/study-material-summarizer.git
cd study-material-summarizer
