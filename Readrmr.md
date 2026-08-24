# 🎙️ AI Interviewer

An offline, voice-enabled AI interview assistant that conducts structured interviews using a locally-hosted LLM (Llama 3 via Ollama), transcribes candidate responses with Whisper, and responds with natural speech using Piper or Coqui TTS — all through a simple Streamlit interface.

Built for privacy-conscious use cases (HR screening, mock interviews, research studies) where sending audio or transcripts to third-party cloud APIs is undesirable.

---

## 📋 Overview

AI Interviewer simulates a real interview experience end-to-end:

1. The system generates or selects an interview question.
2. The candidate answers by speaking into their microphone.
3. Whisper transcribes the response locally.
4. Llama 3 (via Ollama) evaluates the answer and generates a relevant follow-up question.
5. The follow-up is converted to speech and played back to the candidate.
6. All questions, answers, and metadata are logged to a local database for later review.

Everything runs on your own machine — no external API keys or internet-dependent services required.

---

## ✨ Features

- 🗣️ **Voice-driven interviews** — full spoken interaction, no typing required
- 🧠 **Local LLM reasoning** — powered by Llama 3 running through Ollama
- 📝 **Automatic transcription** — powered by OpenAI Whisper
- 🔊 **Natural voice responses** — Piper or Coqui TTS for playback
- 💾 **Session persistence** — interview history stored in a local database
- 🎛️ **Configurable prompts** — easily customize interview style and question sets via `prompts.py`
- 🖥️ **Simple web UI** — built with Streamlit, no frontend framework needed
- 🔒 **Fully offline-capable** — no data leaves your machine

---

## 📁 Folder Structure

```
ai-interviewer/
│
├── app.py              # Streamlit application entry point (UI)
├── interviewer.py       # Core interview flow and session orchestration
├── ai_engine.py         # LLM integration — question generation & evaluation
├── speech.py            # Speech-to-text (Whisper) & text-to-speech (Piper/Coqui) handling
├── database.py          # Database models, connection, and initialization
├── prompts.py            # Prompt templates for the LLM
├── utils.py              # Shared helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ai-interviewer.git
cd ai-interviewer
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate it
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Ollama is used to run Llama 3 locally.

**macOS / Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download and run the installer from [https://ollama.com/download](https://ollama.com/download)

Verify installation:
```bash
ollama --version
```

### 5. Pull the Llama 3 model

```bash
ollama pull llama3
```

Ensure the Ollama service is running before starting the app:
```bash
ollama serve
```

### 6. Install Whisper (Speech-to-Text)

```bash
pip install openai-whisper
```

Whisper requires `ffmpeg` to process audio:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

### 7. Install Piper or Coqui TTS (Text-to-Speech)

Choose **one** of the following:

**Option A — Piper TTS** (lightweight, fast)
```bash
pip install piper-tts
```

**Option B — Coqui TTS** (more voice options, higher quality)
```bash
pip install TTS
```

> Update `speech.py` with the engine you choose, along with the path to your preferred voice model.

---

## ▶️ Running the Application

### Step 1: Initialize the database

```bash
python database.py
```

This creates the local database file and sets up the required tables for storing interview sessions and responses.

### Step 2: Launch the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

## 🖼️ Screenshots

> _Add screenshots of the application here to showcase the interview flow._

| Home Screen | Live Interview | Session History |
|:---:|:---:|:---:|
| ![Home Screen](docs/screenshots/home.png) | ![Live Interview](docs/screenshots/interview.png) | ![Session History](docs/screenshots/history.png) |

---

## 🔮 Future Improvements

- [ ] Support for multiple LLM backends (e.g., Mistral, Phi-3) via Ollama
- [ ] Real-time streaming transcription instead of batch processing
- [ ] Candidate scoring and feedback report generation (PDF export)
- [ ] Multi-language interview support
- [ ] Admin dashboard for reviewing and comparing candidate sessions
- [ ] Dockerized deployment for one-command setup
- [ ] Configurable interview templates (technical, behavioral, HR screening)
- [ ] Resume/CV parsing to generate personalized questions

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ using Streamlit, Ollama, Whisper, and open-source TTS engines.</p>