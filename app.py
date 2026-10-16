import os
import tempfile
import gradio as gr
from groq import Groq
from pypdf import PdfReader
from docx import Document
from google import genai
from elevenlabs.client import ElevenLabs
import sentry_sdk

# ==========================================
# 1. INITIALIZATION & ENVIRONMENT
# ==========================================
# Note: load_dotenv() is skipped because Render uses real Environment Variables

# 1. GROQ SETUP
groq_key = os.environ.get("GROQ_API_KEY")
if groq_key and groq_key.startswith("gsk_"):
    client_groq = Groq(api_key=groq_key)
    print("✅ Groq API Key loaded successfully")
else:
    print("❌ WARNING: GROQ_API_KEY is missing or invalid!")
    client_groq = None

# 2. GOOGLE SETUP (Fixed for new 'google-genai' library)
google_key = os.environ.get("GOOGLE_API_KEY")
if google_key and google_key.startswith("AIza"):
    client_google = genai.Client(api_key=google_key)
    print("✅ Google API Key loaded successfully")
else:
    print(" WARNING: GOOGLE_API_KEY is missing or invalid!")
    client_google = None

# 3. ELEVENLABS SETUP
eleven_key = os.environ.get("ELEVENLABS_API_KEY")
if eleven_key and eleven_key.startswith("sk_"):
    client_eleven = ElevenLabs(api_key=eleven_key)
    print("✅ ElevenLabs API Key loaded successfully")
else:
    print("❌ WARNING: ELEVENLABS_API_KEY is missing or invalid!")
    client_eleven = None

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
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
    if not text or "Router" in text or "Failed" in text or client_eleven is None:
        return None
    try:
        audio = client_eleven.generate(text=text, voice="Rachel", model="eleven_multilingual_v2")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            for chunk in audio:
                f.write(chunk)
            return f.name
    except Exception as e:
        print(f"ElevenLabs Error: {e}")
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

    # --- ROUTER 1: DEEP MODE (Fixed Model Name) ---
    deep_prompt = f"You are an expert legal assistant. Analyze this contract for a {user_role}. Identify red flags, hidden fees, unfair clauses. Use bullet points and emojis.\n\nText:\n{contract_text}"
    try:
        if client_groq is None: raise ValueError("Groq client not initialized")
        res1 = client_groq.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[{"role": "user", "content": deep_prompt}], 
            temperature=0.3
        )
        red_flags = res1.choices[0].message.content
    except Exception as e:
        red_flags = f"❌ Router 1 Failed: {e}"

    # --- ROUTER 2: SPEED MODE (Fixed Model Name) ---
    speed_prompt = f"Based on these red flags: {red_flags}\n\nDraft a short, polite, professional email to the other party negotiating these points. Do not sound aggressive."
    try:
        if client_groq is None: raise ValueError("Groq client not initialized")
        res2 = client_groq.chat.completions.create(
            model="llama3-8b-8192", 
            messages=[{"role": "user", "content": speed_prompt}], 
            temperature=0.5
        )
        email_draft = res2.choices[0].message.content
    except Exception as e:
        email_draft = f"❌ Router 2 Failed: {e}"

    # --- ROUTER 3: GEMINI PRIVACY SCORE (Fixed Syntax) ---
    try:
        if client_google is None: raise ValueError("Google client not initialized")
        score_prompt = f"Rate the privacy risk of this contract from 1 (Safe) to 10 (Highly Predatory). Reply ONLY with the number.\n\n{contract_text}"
        res3 = client_google.models.generate_content(model='gemini-1.5-flash', contents=score_prompt)
        privacy_score = f" Gemini Privacy Risk Score: {res3.text.strip()}/10"
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
