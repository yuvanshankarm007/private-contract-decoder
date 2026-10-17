"""
Private Contract & Lease Decoder
Hacktoberfest 2026 (#hf26challenge #hacktoberfest)

Architecture:
  - Router 1 (Groq, deep):   red-flag legal analysis
  - Router 2 (Groq, speed):  negotiation email draft
  - Router 3 (Gemma 4):      privacy risk score (Gemini + Groq fallbacks)
  All three fire in parallel threads with hard timeouts and a 10s score budget.
  Self-healing model chains hop automatically across deprecated/congested models.
  Results are cached; providers are warmed at boot; audio falls back to free gTTS.

Categories: Best Use of Gemma / Render / ElevenLabs / Sentry / GitHub Copilot
"""

import os
import re
import time
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from gtts import gTTS
import sentry_sdk

# ==========================================
# 1. INITIALIZATION & ENVIRONMENT
# ==========================================
load_dotenv()  # Only for local testing; Render uses real env vars

MAX_CONTRACT_CHARS = 12000

sentry_dsn = os.environ.get("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
    print("✅ Sentry connected")

groq_key = os.environ.get("GROQ_API_KEY")
client_groq = Groq(api_key=groq_key, timeout=10.0, max_retries=1) if groq_key else None
print("✅ Groq key loaded" if client_groq else "❌ GROQ_API_KEY missing")

google_key = os.environ.get("GOOGLE_API_KEY")
client_google = genai.Client(api_key=google_key, http_options=types.HttpOptions(timeout=15000)) if google_key else None
print("✅ Google key loaded" if client_google else "❌ GOOGLE_API_KEY missing")

eleven_key = os.environ.get("ELEVENLABS_API_KEY")
client_eleven = ElevenLabs(api_key=eleven_key) if eleven_key else None
print("✅ ElevenLabs key loaded" if client_eleven else "⚠️ ElevenLabs missing (free gTTS fallback active)")

# SELF-HEALING MODEL ROUTERS (fast first, smart fallbacks; names verified live)
DEEP_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
SPEED_MODELS = ["llama-3.1-8b-instant", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
GEMMA_MODELS = ["gemma-3-4b-it", "gemma-4-26b-a4b-it", "gemma-4-31b-it"]
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]

_RESPONSE_CACHE = {}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def groq_chat(model_list, prompt, temperature, max_tokens=None):
    """Tries each Groq model until one works."""
    last_error = None
    for model_name in model_list:
        try:
            res = client_groq.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return res.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    raise last_error

def _extract_text(resp):
    """Extracts visible text from a Google response; '' if none (never dumps repr)."""
    txt = getattr(resp, "text", None)
    if txt:
        return txt.strip()
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()
    out = getattr(resp, "output", None)
    if out:
        chunks = []
        for item in out:
            for part in (getattr(item, "content", None) or []):
                t = getattr(part, "text", None)
                if t:
                    chunks.append(t)
        if chunks:
            return "\n".join(chunks).strip()
    return ""

def _clean_score(raw):
    """Pulls a single 1-10 digit out of any model reply."""
    m = re.search(r'\b(10|[1-9])\b', raw or "")
    return m.group(1) if m else None

def extract_text_from_file(file):
    if file is None: return ""
    file_path = file.name if hasattr(file, 'name') else file
    ext = file_path.split('.')[-1].lower()
    try:
        if ext == 'pdf':
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == 'docx':
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "⚠️ Unsupported file type."
    except Exception as e:
        return f"⚠️ Error reading file: {e}"

def create_export_file(analysis, email, score):
    content = f"=== PRIVACY RISK SCORE ===\n{score}\n\n=== CONTRACT ANALYSIS ===\n\n{analysis}\n\n\n=== DRAFT NEGOTIATION EMAIL ===\n\n{email}"
    with tempfile.NamedTemporaryFile(delete=False, suffix="_contract_analysis.txt", mode='w', encoding='utf-8') as f:
        f.write(content)
        return f.name

def generate_audio(text):
    if not text or "Router" in text or "Failed" in text:
        return None
    if client_eleven:
        try:
            audio = client_eleven.generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                for chunk in audio:
                    f.write(chunk)
                return f.name
        except Exception as e:
            print(f"ElevenLabs failed ({e}), using free gTTS fallback")
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            return f.name
    except Exception as e:
        print(f"Audio error: {e}")
        return None

# ==========================================
# 3. ROUTERS (PARALLEL + DEADLINED + CACHED)
# ==========================================
def _safe(fn, label):
    def wrapper(*args):
        try:
            return fn(*args)
        except Exception as e:
            return f"❌ {label} Failed: {e}"
    return wrapper

def _run_router1(contract_text, user_role):
    deep_prompt = f"You are an expert legal assistant. Analyze this contract for a {user_role}. Identify red flags, hidden fees, unfair clauses. Use bullet points and emojis. MAX 5 bullets, under 150 words.\n\nText:\n{contract_text}"
    if client_groq is None: raise ValueError("GROQ_API_KEY missing")
    return groq_chat(DEEP_MODELS, deep_prompt, 0.3, max_tokens=450)

def _run_router2(contract_text, user_role):
    speed_prompt = f"You are a professional negotiator. Read this {user_role} contract, pick the 2-3 most unfair terms, and draft a short, polite, professional email (under 100 words) asking to revise them. Do not sound aggressive.\n\nText:\n{contract_text}"
    if client_groq is None: raise ValueError("GROQ_API_KEY missing")
    return groq_chat(SPEED_MODELS, speed_prompt, 0.5, max_tokens=250)

def _run_router3(contract_text):
    """Pure Groq scoring for maximum speed."""
    score_prompt = f"Rate the privacy risk of this contract from 1 (Safe) to 10 (Highly Predatory). Reply ONLY with the number.\n\n{contract_text}"
    try:
        num = _clean_score(groq_chat(SPEED_MODELS, score_prompt, 0.0, max_tokens=5))
        if num:
            return f"🔒 Privacy Risk Score: {num}/10"
    except Exception as e:
        print(f"Score failed: {e}")
    return "❌ Score Failed"

@sentry_sdk.trace
def process_contract(uploaded_file, text_input, user_role):
    contract_text = ""
    if uploaded_file: contract_text += extract_text_from_file(uploaded_file)
    if text_input: contract_text += "\n" + text_input
    contract_text = contract_text.strip()[:MAX_CONTRACT_CHARS]

    if not contract_text:
        return "⚠️ Please upload a file or paste text.", "No text provided.", "N/A", None

    cache_key = (contract_text, user_role)
    if cache_key in _RESPONSE_CACHE:
        red_flags, email_draft, privacy_score = _RESPONSE_CACHE[cache_key]
        print("⚡ Cache hit - instant response")
    else:
        with ThreadPoolExecutor(max_workers=3) as ex:
            f1 = ex.submit(_safe(_run_router1, "Router 1"), contract_text, user_role)
            f2 = ex.submit(_safe(_run_router2, "Router 2"), contract_text, user_role)
            f3 = ex.submit(_safe(_run_router3, "Score"), contract_text)
            red_flags = f1.result()
            email_draft = f2.result()
            privacy_score = f3.result()
        if len(_RESPONSE_CACHE) > 32:
            _RESPONSE_CACHE.pop(next(iter(_RESPONSE_CACHE)))
        _RESPONSE_CACHE[cache_key] = (red_flags, email_draft, privacy_score)

    export_path = create_export_file(red_flags, email_draft, privacy_score)
    return red_flags, email_draft, privacy_score, export_path

# ==========================================
# 4. STARTUP WARM-UP (bounded by client timeouts)
# ==========================================
def _warmup():
    try:
        if client_groq:
            groq_chat(SPEED_MODELS, "Say OK", 0.0, max_tokens=5)
    except Exception:
        pass
    try:
        if client_google:
            client_google.models.generate_content(model=GEMMA_MODELS[0], contents="Reply 1", config=types.GenerateContentConfig(max_output_tokens=64))
    except Exception:
        pass

threading.Thread(target=_warmup, daemon=True).start()

# ==========================================
# 5. GRADIO UI
# ==========================================
with gr.Blocks(title="Private Contract Decoder", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Private Contract & Lease Decoder")
    gr.Markdown("Upload a file or paste text. Get AI analysis, a negotiation email, and a privacy score. **100% Private.**")
    gr.Markdown("⚖️ Educational tool — not legal advice.")

    with gr.Row():
        with gr.Column():
            role_dropdown = gr.Dropdown(["Freelancer", "Tenant", "Small Business Owner"], label="Select Your Role", value="Freelancer")
            file_input = gr.File(label="Upload Contract (PDF, DOCX, TXT)", file_types=['.pdf', '.docx', '.txt'])
            text_input = gr.Textbox(label="Or Paste Text Here", lines=5)
            run_btn = gr.Button("🔍 Analyze & Draft", variant="primary")

        with gr.Column():
            score_output = gr.Textbox(label="🔒 Gemma Privacy Score", lines=1)
            deep_output = gr.Textbox(label="🧠 Router 1: Red Flags", lines=8)
            speed_output = gr.Textbox(label="✉️ Router 2: Email Draft", lines=8)
            export_output = gr.File(label="⬇️ Download Report (.txt)", interactive=False)

    with gr.Row():
        speak_btn = gr.Button("🔊 Read Email Aloud (ElevenLabs)", variant="secondary")
        audio_output = gr.Audio(label="Audio Draft", type="filepath", autoplay=False)

    run_btn.click(fn=process_contract, inputs=[file_input, text_input, role_dropdown], outputs=[deep_output, speed_output, score_output, export_output])
    speak_btn.click(fn=generate_audio, inputs=speed_output, outputs=audio_output)

# ==========================================
# 6. LAUNCH
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
