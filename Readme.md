# 🧑‍💼 Resume Screening Agent

An AI-powered resume screening tool that runs **entirely on your local machine**. Upload a batch of candidate resumes (PDF/DOCX) and a job description, and the agent uses a locally-hosted **Llama 3** model (via **Ollama**) to extract candidate information, compare each resume against the job description, and return a ranked, sortable shortlist — no data ever leaves your machine.

---

## Overview

Screening resumes manually is slow and inconsistent. This project automates the first pass: it parses each resume, asks a local LLM to extract structured candidate data (name, skills, experience, education, etc.), asks the same LLM to score that candidate against your job description, and presents the results as a ranked table in a simple Streamlit UI.

Because the model runs locally through Ollama, there's no API key, no per-request cost, and no resume data sent to a third-party cloud service.

---

## Features

- 📄 **Multi-format resume upload** — accepts multiple PDF and DOCX resumes at once
- 📝 **Flexible job description input** — paste text directly, or upload a PDF/DOCX/TXT file
- 🤖 **Local AI analysis** — powered by Llama 3 running through Ollama, fully offline after setup
- 🧠 **Structured extraction** — pulls name, contact info, skills, experience, and education from each resume
- 📊 **Match scoring** — each candidate gets a 0–100 match score, strengths, gaps, and a recommendation
- 🏆 **Automatic ranking** — candidates are sorted best-to-worst by match score, with an explicit rank
- 🖥️ **Sortable results table** — click any column header to re-sort in the UI
- ⬇️ **CSV export** — download the full results table for further review
- 🛡️ **Resilient batch processing** — a corrupted file or a failed AI call for one resume never stops the rest of the batch; it's simply flagged as failed
- 🔁 **Automatic retries** — invalid/malformed JSON from the LLM is automatically retried before giving up

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally (see below)
- ~5 GB free disk space for the Llama 3 model

### 1. Clone the project and install dependencies

```bash
git clone <your-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Install Ollama

Ollama is the local runtime that serves the Llama 3 model to this app.

**macOS**
- Download the official app from [ollama.com/download](https://ollama.com/download) and drag it into Applications, **or**
- Install via Homebrew:
  ```bash
  brew install ollama
  ```

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
This installs the `ollama` binary and, on most distributions, sets it up as a systemd service that starts automatically.

**Windows**
- Download and run `OllamaSetup.exe` from [ollama.com/download](https://ollama.com/download), **or**
- Install via winget:
  ```powershell
  winget install Ollama.Ollama
  ```

**Verify the installation:**
```bash
ollama --version
```

**Start the Ollama server** (if it isn't already running as a background service):
```bash
ollama serve
```

---

## How to Pull Llama 3

Once Ollama is installed and running, download the Llama 3 model:

```bash
ollama pull llama3
```

Confirm it downloaded successfully:

```bash
ollama list
```

You should see `llama3` in the list of available models. This model name is what `ai_engine.py` uses by default — if you pull a different tag (e.g. `llama3:70b`), pass it as the `model` argument in `screener.screen_candidates(...)` or update `DEFAULT_MODEL` in `ai_engine.py`.

> 💡 Tip: You can do a quick sanity check that everything works with `ollama run llama3 "Say hello"` before launching the app.

---

## How to Run Streamlit

With Ollama running and the `llama3` model pulled, start the app from the project root:

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it in your browser. From there:

1. Paste or upload the job description.
2. Upload one or more candidate resumes (PDF/DOCX).
3. Click **Screen Candidates**.
4. Review the ranked, sortable results table, and optionally download it as CSV.

---

## Folder Structure

```
resume-screening-agent/
│
├── app.py              # Streamlit UI: uploads, button, results table, error handling
├── ai_engine.py         # Connects to Ollama, sends prompts, validates/retries JSON responses
├── parser.py            # Extracts plain text from PDF/DOCX resumes
├── screener.py           # Orchestrates parser.py + ai_engine.py, ranks candidates
├── prompts.py            # Reusable JSON-only prompt templates for extraction & comparison
├── utils.py               # Generic helpers: text cleaning, safe JSON parsing, sorting
├── requirements.txt        # Python dependencies
└── README.md                # This file
```

**Responsibility boundaries** (by design, each module has one job):
- `app.py` — UI only, no AI or parsing logic
- `parser.py` — file → plain text, no AI logic
- `prompts.py` — prompt text only, no API calls
- `ai_engine.py` — API calls + JSON validation/retries, no business logic
- `screener.py` — orchestration only, no direct Ollama or file-parsing calls
- `utils.py` — generic, reusable helpers with no domain-specific assumptions

---

## Future Improvements

- 🔍 **OCR support** for scanned/image-only PDF resumes (no embedded text layer)
- 🌐 **Multi-language resume support**
- 📁 **Batch job description support** — screen the same resumes against multiple job openings at once
- 🧩 **Configurable extraction schema** so different industries can capture different fields
- 📈 **Historical tracking** — save past screening runs and compare candidates across sessions
- 🔐 **Authentication/roles** for multi-recruiter use
- ⚡ **Parallel screening** — process multiple resumes concurrently to speed up large batches
- 🧪 **Automated evaluation set** to benchmark scoring consistency across model versions
- 🗂️ **Direct ATS integration** (e.g. Greenhouse, Lever) for pulling/pushing candidate data
- 🎛️ **In-UI model picker** to switch between Llama 3 variants (8B/70B) or other locally-pulled models