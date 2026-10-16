import os
import tempfile
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from google import genai
from elevenlabs.client import ElevenLabs
from gtts import gTTS
import sentry_sdk

# ==========================================
# 1. INITIALIZATION & ENVIRONMENT
# ==========================================
load_dotenv()  # Only for local testing; Render uses real env vars

sentry_dsn = os.environ.get("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
    print("✅ Sentry connected")

groq_key = os.environ.get("GROQ_API_KEY")
client_groq = Groq(api_key=groq_key) if groq_key else None
print("✅ Groq key loaded" if client_groq else "❌ GROQ_API_KEY missing")

google_key = os.environ.get("GOOGLE_API_KEY")
client_google = genai.Client(api_key=google_key) if google_key else None
print("✅ Google key loaded" if client_google else "❌ GOOGLE_API_KEY missing")

eleven_key = os.environ.get("ELEVENLABS_API_KEY")
client_eleven = ElevenLabs(api_key=eleven_key) if eleven_key else None
print("✅ ElevenLabs key loaded" if client_eleven else "⚠️ ElevenLabs missing (free gTTS fallback active)")

# SELF-HEALING MODEL ROUTERS
DEEP_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"]
SPEED_MODELS = ["qwen/qwen3.8-27b", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def groq_chat(model_list, prompt, temperature):
    last_error = None
    for model_name in model_list:
        try:
            res = client_groq.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return res.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    raise last_error

def _extract_text(resp):
    """Works with BOTH the new Interactions API and old generate_content."""
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
    return str(resp).strip()

def gemini_generate(prompt):
    """Tries the NEW Interactions API (Gemini 3.x), then falls back to old method."""
    last_error = None
    # 1) New Interactions API — required for Gemini 3.x models
    for model_name in ["gemini-3.8-flash", "gemini-3-flash"]:
        try:
            resp = client_google.interactions.create(model=model_name, input=prompt)
            return _extract_text(resp)
        except Exception as e:
            last_error = e
            continue
    # 2) Old generate_content method — for older models
    for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            resp = client_google.models.generate_content(model=model_name, contents=prompt)
            return _extract_text(resp)
        except Exception as e:
            last_error = e
            continue
    raise last_error

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
# 3. MAIN PROCESSING FUNCTION
# ==========================================
@sentry_sdk.trace
def process_contract(uploaded_file, text_input, user_role):
    contract_text = ""
    if uploaded_file: contract_text += extract_text_from_file(uploaded_file)
    if text_input: contract_text += "\n" + text_input
    contract_text = contract_text.strip()

    if not contract_text:
        return "⚠️ Please upload a file or paste text.", "No text provided.", "N/A", None

    deep_prompt = f"You are an expert legal assistant. Analyze this contract for a {user_role}. Identify red flags, hidden fees, unfair clauses. Use bullet points and emojis.\n\nText:\n{contract_text}"
    try:
        if client_groq is None: raise ValueError("GROQ_API_KEY missing")
        red_flags = groq_chat(DEEP_MODELS, deep_prompt, 0.3)
    except Exception as e:
        red_flags = f"❌ Router 1 Failed: {e}"

    speed_prompt = f"Based on these red flags: {red_flags}\n\nDraft a short, polite, professional email to the other party negotiating these points. Do not sound aggressive."
    try:
        if client_groq is None: raise ValueError("GROQ_API_KEY missing")
        email_draft = groq_chat(SPEED_MODELS, speed_prompt, 0.5)
    except Exception as e:
        email_draft = f"❌ Router 2 Failed: {e}"

    try:
        if client_google is None: raise ValueError("GOOGLE_API_KEY missing")
        score_prompt = f"Rate the privacy risk of this contract from 1 (Safe) to 10 (Highly Predatory). Reply ONLY with the number.\n\n{contract_text}"
        privacy_score = f"🔒 Gemini Privacy Risk Score: {gemini_generate(score_prompt)}/10"
    except Exception as e:
        privacy_score = f"❌ Gemini Score Failed: {e}"

    export_path = create_export_file(red_flags, email_draft, privacy_score)
    return red_flags, email_draft, privacy_score, export_path

# ==========================================
# 4. GRADIO UI SETUP
# ==========================================
with gr.Blocks(title="Private Contract Decoder", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Private Contract & Lease Decoder")
    gr.Markdown("Upload a file or paste text. Get AI analysis, a negotiation email, and a privacy score. **100% Private.**")

    with gr.Row():
        with gr.Column():
            role_dropdown = gr.Dropdown(["Freelancer", "Tenant", "Small Business Owner"], label="Select Your Role", value="Freelancer")
            file_input = gr.File(label="Upload Contract (PDF, DOCX, TXT)", file_types=['.pdf', '.docx', '.txt'])
            text_input = gr.Textbox(label="Or Paste Text Here", lines=5)
            run_btn = gr.Button("🔍 Analyze & Draft", variant="primary")

        with gr.Column():
            score_output = gr.Textbox(label="🔒 Gemini Privacy Score", lines=1)
            deep_output = gr.Textbox(label="🧠 Router 1: Red Flags", lines=8)
            speed_output = gr.Textbox(label="✉️ Router 2: Email Draft", lines=8)
            export_output = gr.File(label="⬇️ Download Report (.txt)", interactive=False)

    with gr.Row():
        speak_btn = gr.Button("🔊 Read Email Aloud (ElevenLabs)", variant="secondary")
        audio_output = gr.Audio(label="Audio Draft", type="filepath", autoplay=False)

    run_btn.click(fn=process_contract, inputs=[file_input, text_input, role_dropdown], outputs=[deep_output, speed_output, score_output, export_output])
    speak_btn.click(fn=generate_audio, inputs=speed_output, outputs=audio_output)

# ==========================================
# 5. LAUNCH
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
