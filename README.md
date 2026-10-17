# 🛡️ Private Contract & Lease Decoder

A fast, private, and free AI tool that acts as a first-pass legal reviewer. Upload a contract or lease, and in seconds, get a bulleted list of red flags, a professional negotiation email, and a predatory risk score. 

Built for students, freelancers, and tenants who need to verify documents before signing but can't afford a $300/hr lawyer.

 **Live Demo:** [Insert Your Render URL Here]

## ✨ Features

- **Parallel AI Analysis:** Three AI routers fire simultaneously to analyze the document in ~3 seconds.
- **Self-Healing Fallback Chains:** If a model fails, times out, or gets deprecated, the app instantly hops to the next available model.
- **Multi-Format Support:** Upload PDFs, DOCX files, or paste raw text.
- **Audio Read-Out:** Listen to your negotiation email on the go via ElevenLabs (with a silent fallback to free gTTS).
- **100% Private:** No data is stored. The app processes text in memory and returns the results.

## 🏗️ Architecture

The app is built on a resilient, parallel architecture designed to survive free-tier API throttling and deprecations:

1. **Router 1 (Deep Analysis):** Uses larger models (e.g., `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`) to identify specific legal red flags and hidden traps.
2. **Router 2 (Email Drafting):** Uses fast, structured models to draft a polite, professional negotiation email.
3. **Router 3 (Risk Scoring):** Uses a lightweight model with strict regex parsing to output a 1-10 predatory risk score.

*All routers run concurrently via Python's `ThreadPoolExecutor` and are wrapped in Sentry distributed tracing for real-time observability.*

## 🛠️ Tech Stack

- **Frontend/Backend:** Python, Gradio
- **AI Inference:** Groq API (GPT-OSS, Qwen)
- **File Parsing:** PyPDF2, python-docx
- **Audio:** ElevenLabs, gTTS (Google Text-to-Speech)
- **Observability:** Sentry
- **Hosting:** Render

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/yuvanshankarm007/private-contract-decoder.git
cd private-contract-decoder
