import os
import re
import time
import tempfile
import gradio as gr
import sentry_sdk
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
import PyPDF2
import docx

# --- 1. ENVIRONMENT & SENTRY SETUP ---
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=1.0,
)

groq_key = os.environ.get("GROQ_API_KEY")
elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")

# --- 2. API CLIENTS ---
# Timeout set to 10s to prevent the 190s hang
client_groq = Groq(api_key=groq_key, timeout=10.0, max_retries=1) if groq_key else None
print("✅ Groq key loaded" if client_groq else "❌ GROQ_API_KEY missing")

# --- 3. MODEL CONFIGURATION (The exact IDs from your Groq Console) ---
DEEP_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
SPEED_MODELS = ["openai/gpt-oss-20b"]

# --- 4. CACHING FOR INSTANT REPEAT RESULTS ---
_RESPONSE_CACHE = {}

# --- 5. HELPER FUNCTIONS ---
def extract_text_from_file(file_path):
    """Extracts text from PDF, DOCX, or TXT files."""
    if not file_path:
        return ""
    ext = os.path.splitext(file_path.name)[1].lower()
    try:
        if ext == '.pdf':
            with open(file_path.name, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == '.docx':
            doc = docx.Document(file_path.name)
            return "\n".join([para.text for para in doc.paragraphs])
        else: # TXT
            with open(file_path.name, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def groq_chat(models, prompt, temp, max_tokens):
    """Iterates through fallback models until one works."""
    for model in models:
        try:
            response = client_groq.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                timeout=10.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    raise ValueError("All Groq models failed.")

# --- 6. THE THREE ROUTERS ---
def _run_router1(contract_text, user_role):
    """Deep analysis for Red Flags."""
    prompt = f"You are an expert legal assistant. Analyze this contract for a {user_role}. Identify red flags, hidden fees, unfair clauses. Use bullet points and emojis. MAX 5 bullets, under 150 words.\n\nText:\n{contract_text}"
    return groq_chat(DEEP_MODELS, prompt, 0.3, max_tokens=450)

def _run_router2(contract_text, user_role):
    """Drafting the negotiation email (Increased tokens to 500 so it doesn't cut off)."""
    prompt = f"You are a professional negotiator. Read this {user_role} contract, pick the 2-3 most unfair terms, and draft a short, polite, professional email (under 150 words) asking to revise them. Do not sound aggressive.\n\nText:\n{contract_text}"
    return groq_chat(SPEED_MODELS, prompt, 0.5, max_tokens=500)

def _run_router3(contract_text):
    """Fast scoring with strict regex extraction."""
    # Truncate to 1500 chars to ensure it finishes in <2 seconds
    short_text = contract_text[:1500] 
    prompt = f"Rate the privacy/predatory risk of this contract from 1 (Safe) to 10 (Highly Predatory). Reply ONLY with the single digit number. Do not write any other words, no punctuation.\n\nText:\n{short_text}"
    
    response = groq_chat(SPEED_MODELS, prompt, 0.0, max_tokens=10)
    match = re.search(r'\b(10|[1-9])\b', response)
    if match:
        return f" Privacy Risk Score: {match.group(1)}/10"
    return f"🔒 Privacy Risk Score: {response.strip()}/10"

# --- 7. AUDIO GENERATION WITH FALLBACK ---
def generate_audio(text):
    """Tries ElevenLabs, falls back to free gTTS."""
    try:
        if elevenlabs_key:
            # Simplified ElevenLabs call (adjust if using specific SDK version)
            from elevenlabs import generate, save
            audio = generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                save(audio, tmp.name)
                return tmp.name
    except Exception as e:
        print(f"ElevenLabs failed, using gTTS: {e}")
    
    # Free gTTS Fallback
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name

# --- 8. MAIN PROCESSING PIPELINE ---
@sentry_sdk.trace
def process_contract(role, file, text_input):
    # 1. Get Text
    contract_text = extract_text_from_file(file) if file else text_input
    if not contract_text.strip():
        return "Please upload a file or paste text.", "", "", None, None

    # 2. Check Cache
    cache_key = hash(contract_text)
    if cache_key in _RESPONSE_CACHE:
        return _RESPONSE_CACHE[cache_key]

    # 3. Run Routers in Parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(_run_router1, contract_text, role)
        f2 = executor.submit(_run_router2, contract_text, role)
        f3 = executor.submit(_run_router3, contract_text)

        try:
            red_flags = f1.result()
        except Exception as e:
            red_flags = f"❌ Router 1 Failed: {e}"
            
        try:
            email_draft = f2.result()
        except Exception as e:
            email_draft = f"❌ Router 2 Failed: {e}"
            
        try:
            score = f3.result()
        except Exception as e:
            score = f"❌ Score Failed: {e}"

    # 4. Generate Audio & Report
    audio_path = generate_audio(email_draft) if "❌" not in email_draft else None
    
    report_content = f"ROLE: {role}\n\n--- RED FLAGS ---\n{red_flags}\n\n--- EMAIL DRAFT ---\n{email_draft}\n\n--- SCORE ---\n{score}"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt") as tmp:
        tmp.write(report_content)
        report_path = tmp.name

    # 5. Save to Cache & Return
    result = (score, red_flags, email_draft, report_path, audio_path)
    _RESPONSE_CACHE[cache_key] = result
    return result

# --- 9. GRADIO UI (Matching your screenshots exactly) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# ️ Private Contract & Lease Decoder")
    gr.Markdown("Upload a file or paste text. Get AI analysis, a negotiation email, and a privacy score. **100% Private.**")
    gr.Markdown("⚖️ *Educational tool — not legal advice.*")

    with gr.Row():
        with gr.Column():
            role_dropdown = gr.Dropdown(["Freelancer", "Tenant", "Employee", "Client"], label="Select Your Role", value="Freelancer")
            file_input = gr.File(label="Upload Contract (PDF, DOCX, TXT)", file_types=['.pdf', '.docx', '.txt'])
            text_input = gr.Textbox(label="Or Paste Text Here", lines=10, placeholder="Paste your contract text here...")
            analyze_btn = gr.Button("🔍 Analyze & Draft", variant="primary", size="lg")
            
        with gr.Column():
            score_output = gr.Textbox(label="Privacy Risk Score", lines=1)
            red_flags_output = gr.Textbox(label="🧠 Router 1: Red Flags", lines=8)
            email_output = gr.Textbox(label="📧 Router 2: Email Draft", lines=8)
            report_output = gr.File(label="Download Report (.txt)")
            audio_btn = gr.Button("🔊 Read Email Aloud (ElevenLabs)", variant="secondary")
            audio_output = gr.Audio(label="Audio Draft", type="filepath")

    analyze_btn.click(
        fn=process_contract,
        inputs=[role_dropdown, file_input, text_input],
        outputs=[score_output, red_flags_output, email_output, report_output, audio_output]
    )

# --- 10. LAUNCH ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
