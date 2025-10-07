# app.py — Enhanced Mindful Wellness Voice Chatbot 🌿
import os
import base64
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import streamlit as st

# ---------- Optional Dependencies ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import asyncio
    import edge_tts
except ImportError:
    edge_tts = None
    asyncio = None
# --------------------------------------------

# ---------- Configuration ----------
AUDIO_DIR = Path("audio_responses")
AUDIO_DIR.mkdir(exist_ok=True)
AUDIO_TTL_HOURS = 6
MAX_CONTEXT_MESSAGES = 15
MODEL_LIST = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.2-1b-instruct"
]
LISTEN_TIMEOUT_DEFAULT = 5
PHRASE_TIME_LIMIT_DEFAULT = 15

# Intent patterns for better context
INTENT_PATTERNS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "meditation": ["meditate", "meditation", "breathing", "breath", "relax", "calm"],
    "mood_check": ["how am i", "feeling", "mood", "emotions", "anxious", "stressed"],
    "gratitude": ["grateful", "thankful", "appreciate", "gratitude"],
    "sleep": ["sleep", "rest", "tired", "insomnia", "can't sleep"],
    "exercise": ["exercise", "workout", "movement", "yoga", "walk"],
    "farewell": ["goodbye", "bye", "see you", "talk later"]
}

# Available voices for TTS
VOICE_OPTIONS = {
    "Jenny (Female, Calm)": "en-US-JennyNeural",
    "Guy (Male, Warm)": "en-US-GuyNeural",
    "Aria (Female, Friendly)": "en-US-AriaNeural",
    "Davis (Male, Professional)": "en-US-DavisNeural"
}
# -------------------------------------

st.set_page_config(
    page_title="🌿 Mindful Wellness AI Assistant", 
    page_icon="🪷", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Session State Initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "continuous_listening" not in st.session_state:
    st.session_state.continuous_listening = False

if "interrupt_signal" not in st.session_state:
    st.session_state.interrupt_signal = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "voice_preference": "en-US-JennyNeural",
        "response_length": "medium",
        "topics_discussed": [],
        "total_conversations": 0
    }

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "idle"  # idle, listening, thinking, speaking

if "audio_response_path" not in st.session_state:
    st.session_state.audio_response_path = None

# ---------- Helper Functions ----------
def cleanup_old_audio():
    """Remove audio files older than TTL"""
    cutoff = datetime.now() - timedelta(hours=AUDIO_TTL_HOURS)
    for f in AUDIO_DIR.glob("response_*.mp3"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink(missing_ok=True)
        except Exception:
            pass

def autoplay_audio_html(file_path):
    """Auto-play audio in browser"""
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

def detect_intent(user_input):
    """Detect user intent from input text"""
    user_input_lower = user_input.lower()
    
    for intent, patterns in INTENT_PATTERNS.items():
        if any(pattern in user_input_lower for pattern in patterns):
            return intent
    
    return "general"

def update_conversation_stats(intent):
    """Track conversation topics"""
    if intent not in st.session_state.user_profile["topics_discussed"]:
        st.session_state.user_profile["topics_discussed"].append(intent)
    st.session_state.user_profile["total_conversations"] += 1

# ---------- OpenRouter Init ----------
@st.cache_resource
def init_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("❌ Please add your OpenRouter API key to `.env`.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

openai_client = init_openai_client()

# ---------- Enhanced System Prompt ----------
def get_system_prompt(intent="general", user_profile=None):
    """Dynamic system prompt based on intent and user profile"""
    
    base_prompt = """
You are a warm, gentle mindfulness companion named 'Mindful'. 
You speak kindly and encourage calm self-reflection and peace. 
Be supportive, empathetic, and human-like. 
Keep responses brief, grounding, and emotionally mindful.
Never give medical advice. Focus on calm and presence.
"""
    
    # Add intent-specific guidance
    intent_prompts = {
        "greeting": "\n\nUser is greeting you. Respond warmly and ask how they're feeling today.",
        "meditation": "\n\nUser wants meditation guidance. Offer a short, calming breathing exercise or mindfulness tip.",
        "mood_check": "\n\nUser is checking in about emotions. Be empathetic, validate their feelings, and offer gentle support.",
        "gratitude": "\n\nUser is expressing gratitude. Acknowledge it warmly and encourage reflection on positive moments.",
        "sleep": "\n\nUser has sleep concerns. Offer calming techniques, avoid medical advice, suggest relaxation practices.",
        "farewell": "\n\nUser is ending the conversation. Wish them well with a peaceful closing."
    }
    
    prompt = base_prompt + intent_prompts.get(intent, "")
    
    # Add personalization
    if user_profile:
        length_map = {
            "short": "\n\nKeep responses very brief (1-2 sentences).",
            "medium": "\n\nKeep responses moderate length (2-4 sentences).",
            "long": "\n\nProvide detailed, thoughtful responses (4-6 sentences)."
        }
        prompt += length_map.get(user_profile.get("response_length", "medium"), "")
    
    return prompt

def build_context(intent="general"):
    """Build conversation context with summarization for long chats"""
    msgs = st.session_state.messages.copy()
    
    # Summarize if conversation is too long
    if len(msgs) > 25:
        # Keep recent messages
        recent = msgs[-15:]
        old = msgs[:-15]
        
        # Create summary of older messages
        summary = f"Earlier in this conversation, we discussed: "
        topics = [detect_intent(m["content"]) for m in old if m["role"] == "user"]
        unique_topics = list(set(topics))
        summary += ", ".join(unique_topics[:5])
        
        msgs = [
            {"role": "system", "content": get_system_prompt(intent, st.session_state.user_profile)},
            {"role": "system", "content": summary}
        ] + recent
    else:
        msgs.insert(0, {"role": "system", "content": get_system_prompt(intent, st.session_state.user_profile)})
    
    # Limit total context
    if len(msgs) > MAX_CONTEXT_MESSAGES:
        msgs = [msgs[0]] + msgs[-MAX_CONTEXT_MESSAGES:]
    
    return msgs

# ---------- AI Reply with Intent ----------
def get_ai_reply(user_message):
    """Get AI response with intent detection"""
    st.session_state.conversation_state = "thinking"
    
    # Detect intent
    intent = detect_intent(user_message)
    update_conversation_stats(intent)
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": user_message})
    
    # Build context
    context = build_context(intent)
    
    # Try models in order
    for model in MODEL_LIST:
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=context,
                max_tokens=250,
                temperature=0.7,
                extra_headers={
                    "HTTP-Referer": "https://mindful-bot.streamlit.app",
                    "X-Title": "Mindful Wellness Chatbot",
                },
            )
            reply = resp.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            st.warning(f"⚠️ Model {model} failed: {e}")
            continue
    
    return "I'm having trouble connecting — but take a deep breath and try again soon. 🌿"

# ---------- Enhanced TTS with Multiple Voices ----------
def text_to_speech(text, voice=None):
    """Convert text to speech with voice selection"""
    st.session_state.conversation_state = "speaking"
    
    if voice is None:
        voice = st.session_state.user_profile.get("voice_preference", "en-US-JennyNeural")
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = AUDIO_DIR / f"response_{ts}.mp3"
    
    try:
        if edge_tts and asyncio:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(edge_tts.Communicate(text, voice).save(str(out)))
            loop.close()
        elif gTTS:
            gTTS(text=text, lang="en").save(str(out))
        else:
            st.warning("⚠️ No text-to-speech engine available.")
            return None
        
        st.session_state.conversation_state = "idle"
        return str(out)
    except Exception as e:
        st.error(f"TTS error: {e}")
        st.session_state.conversation_state = "idle"
        return None

# ---------- Streaming TTS (Sentence by Sentence) ----------
def stream_tts_response(text, voice=None):
    st.session_state.is_speaking = True
    """Stream audio response sentence by sentence"""
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    audio_paths = []
    
    for i, sentence in enumerate(sentences):
        if st.session_state.interrupt_signal:
            st.info("🛑 Response interrupted")
            st.session_state.interrupt_signal = False
            break
        
        audio_path = text_to_speech(sentence, voice)
        if audio_path:
            audio_paths.append(audio_path)
            autoplay_audio_html(audio_path)
            
            # Small pause between sentences
            if i < len(sentences) - 1:
                time.sleep(0.5)
    
    st.session_state.is_speaking = False
    return audio_paths

# ---------- Enhanced Speech to Text ----------
def speech_to_text_microphone(timeout=LISTEN_TIMEOUT_DEFAULT, phrase_time_limit=PHRASE_TIME_LIMIT_DEFAULT):
    """Capture speech from microphone with better error handling"""
    if sr is None:
        st.error("❌ speech_recognition not installed. Install: pip install SpeechRecognition")
        return None
    
    r = sr.Recognizer()
    r.energy_threshold = 4000
    r.dynamic_energy_threshold = True
    
    try:
        with sr.Microphone() as src:
            st.session_state.conversation_state = "listening"
            st.info("🎤 Listening... Speak naturally.")
            r.adjust_for_ambient_noise(src, duration=1)
            audio = r.listen(src, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
        st.session_state.conversation_state = "thinking"
        text = r.recognize_google(audio, language="en-US")
        return text
        
    except sr.WaitTimeoutError:
        st.warning("⏱️ No speech detected. Please try again.")
    except sr.UnknownValueError:
        st.warning("🤷 Could not understand. Please speak clearly.")
    except Exception as e:
        st.error(f"❌ Speech error: {e}")
    finally:
        st.session_state.conversation_state = "idle"
    
    return None

# ---------- Continuous Listening Mode ----------
def continuous_listening():
    """Improved continuous voice interaction mode with cooldown and feedback prevention"""
    if sr is None:
        st.error("❌ speech_recognition not installed.")
        return

    r = sr.Recognizer()
    r.energy_threshold = 4000
    r.dynamic_energy_threshold = True
    st.session_state.is_speaking = False

    st.info("🎤 **Continuous mode active** - Speak anytime!")

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        silence_count = 0

        while st.session_state.continuous_listening:
            try:
                if st.session_state.is_speaking:
                    time.sleep(0.5)
                    continue

                st.session_state.conversation_state = "listening"
                audio = r.listen(source, timeout=3, phrase_time_limit=12)

                st.session_state.conversation_state = "thinking"
                text = r.recognize_google(audio, language="en-US").strip()

                if not text:
                    silence_count += 1
                    if silence_count >= 3:
                        st.info("🕊️ No speech detected for a while. Stopping continuous mode.")
                        st.session_state.continuous_listening = False
                    continue

                silence_count = 0
                st.success(f"💬 You said: {text}")

                reply = get_ai_reply(text)

                st.session_state.is_speaking = True
                audio_paths = stream_tts_response(reply)
                st.session_state.is_speaking = False

                time.sleep(2)
                st.session_state.conversation_state = "idle"

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Listening error: {e}")
                time.sleep(1)

def audio_upload_to_text(uploaded):
    """Convert uploaded audio to text"""
    if sr is None:
        st.error("❌ speech_recognition not installed.")
        return None
    
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getbuffer())
        
        with sr.AudioFile(tmp.name) as src:
            audio = r.record(src)
            text = r.recognize_google(audio, language="en-US")
        
        os.unlink(tmp.name)
        return text
    except Exception as e:
        st.error(f"🎧 Couldn't process audio: {e}")
        return None

# ---------- UI: Status Indicator ----------
def show_status_indicator():
    """Visual status indicator"""
    status_map = {
        "idle": "💤 Ready - Say something!",
        "listening": "🎤 Listening...",
        "thinking": "🤔 Thinking...",
        "speaking": "💬 Speaking..."
    }
    
    status = st.session_state.conversation_state
    return status_map.get(status, "💤 Ready")

# ========================================
# MAIN UI
# ========================================

st.markdown("""
<div style='text-align:center; padding: 20px;'>
    <h1 style='color:#4a7856;'>🌿 Mindful Wellness AI Assistant</h1>
    <p style='color:#666; font-size:18px;'>Your compassionate voice companion for peace and presence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Voice selection
    st.markdown("### 🎵 Voice Selection")
    selected_voice = st.selectbox(
        "Choose AI voice:",
        options=list(VOICE_OPTIONS.keys()),
        index=0
    )
    st.session_state.user_profile["voice_preference"] = VOICE_OPTIONS[selected_voice]
    
    # Response length
    st.markdown("### 📏 Response Length")
    response_length = st.radio(
        "Preferred response style:",
        options=["short", "medium", "long"],
        index=1,
        horizontal=True
    )
    st.session_state.user_profile["response_length"] = response_length
    
    st.markdown("---")
    
    # Conversation stats
    st.markdown("### 📊 Your Journey")
    st.metric("Conversations", st.session_state.user_profile["total_conversations"])
    if st.session_state.user_profile["topics_discussed"]:
        st.write("**Topics explored:**")
        for topic in st.session_state.user_profile["topics_discussed"][:5]:
            st.write(f"• {topic.capitalize()}")
    
    st.markdown("---")
    
    # Reset button
    if st.button("🧹 Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip:** Use headphones for best voice experience")
    st.caption("🌐 Powered by OpenRouter API")

# Main Layout
col1, col2 = st.columns([2, 1])

# Left Column: Chat Interface
with col1:
    st.markdown(f"### 💬 Conversation - {show_status_indicator()}")
    
    # Chat history container
    chat_container = st.container(height=400)
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧘"):
                    st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🌿"):
                    st.markdown(msg["content"])
    
    # Text input
    if prompt := st.chat_input("🪷 Type your message here..."):
        with st.spinner("✨ Reflecting mindfully..."):
            reply = get_ai_reply(prompt)
        
        with st.spinner("🎵 Preparing voice response..."):
            audio_paths = stream_tts_response(reply)
            if audio_paths:
                st.session_state.audio_response_path = audio_paths[-1]
        
        st.rerun()

# Right Column: Voice Controls
with col2:
    st.markdown("### 🎤 Voice Controls")
    
    # Single voice input
    st.markdown("**Quick Voice Input**")
    if st.button("🎙️ Press & Speak", use_container_width=True, type="primary"):
        if sr:
            with st.spinner("🎤 Listening..."):
                text = speech_to_text_microphone(timeout=8, phrase_time_limit=20)
            
            if text:
                st.success(f"✅ You said: *{text}*")
                
                with st.spinner("💭 Processing..."):
                    reply = get_ai_reply(text)
                
                with st.spinner("🔊 Speaking..."):
                    audio_paths = stream_tts_response(reply)
                    if audio_paths:
                        st.session_state.audio_response_path = audio_paths[-1]
                
                st.rerun()
        else:
            st.error("❌ Speech recognition not available")
    
    st.markdown("---")
    
    # Continuous mode
    st.markdown("**Continuous Conversation**")
    
    continuous_col1, continuous_col2 = st.columns(2)
    
    with continuous_col1:
        if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.continuous_listening):
            st.session_state.continuous_listening = True
            threading.Thread(target=continuous_listening, daemon=True).start()
            st.rerun()
    
    with continuous_col2:
        if st.button("⏸️ Stop", use_container_width=True, disabled=not st.session_state.continuous_listening):
            st.session_state.continuous_listening = False
            st.session_state.conversation_state = "idle"
            st.rerun()
    
    if st.session_state.continuous_listening:
        st.info("🔴 **Live** - Speak naturally, I'm listening!")
    
    st.markdown("---")
    
    # Upload audio file
    st.markdown("**Upload Voice Note**")
    uploaded_audio = st.file_uploader(
        "Upload audio (wav/mp3/m4a)",
        type=["wav", "mp3", "m4a", "flac"],
        label_visibility="collapsed"
    )
    


    # ✅ Prevent repeated processing of the same uploaded file
    if uploaded_audio:
        # Hash file buffer to detect same file reupload
        import hashlib
        file_hash = hashlib.md5(uploaded_audio.getbuffer()).hexdigest()

        # Process only if new or changed file
        if st.session_state.get("last_uploaded_hash") != file_hash:
            st.session_state.last_uploaded_hash = file_hash

            with st.spinner("🎧 Processing your voice..."):
                text = audio_upload_to_text(uploaded_audio)

            if text:
                st.success(f"✅ Transcribed: *{text}*")

                with st.spinner("💭 Reflecting..."):
                    reply = get_ai_reply(text)

                with st.spinner("🔊 Responding..."):
                    audio_paths = stream_tts_response(reply)
                    if audio_paths:
                        st.session_state.audio_response_path = audio_paths[-1]

                st.rerun()
        else:
            st.info("🪷 This audio has already been processed. Upload a new one to continue.")



    
    st.markdown("---")
    
    # Interrupt button
    if st.session_state.conversation_state == "speaking":
        if st.button("🛑 Interrupt Response", use_container_width=True, type="secondary"):
            st.session_state.interrupt_signal = True
            st.info("Response stopped!")

# Audio playback
if st.session_state.audio_response_path and os.path.exists(st.session_state.audio_response_path):
    st.audio(st.session_state.audio_response_path, format="audio/mp3")

# Footer
cleanup_old_audio()

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#777; padding: 20px;'>
    <p>🌼 Remember: Each breath is a new beginning</p>
    <p style='font-size:12px;'>💚 Take care of yourself • 🧘 Practice mindfulness • 🌿 Stay present</p>
</div>
""", unsafe_allow_html=True)