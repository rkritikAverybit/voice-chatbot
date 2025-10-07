# app1_streamlit_cloud.py — Streamlit Cloud–ready voice chatbot
import os, io, base64, tempfile, time, hashlib, wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av


from pydub.utils import which
from pydub import AudioSegment

# Ensure ffmpeg is available (for Streamlit Cloud)
if not which("ffmpeg"):
    import os
    import subprocess
    ffmpeg_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    os.system(f"wget -q {ffmpeg_url} -O /tmp/ffmpeg.tar.xz")
    os.system("tar -xf /tmp/ffmpeg.tar.xz -C /tmp")
    ffmpeg_path = next(p for p in os.listdir('/tmp') if p.startswith('ffmpeg') and os.path.isdir(f'/tmp/{p}'))
    os.environ["PATH"] += os.pathsep + f"/tmp/{ffmpeg_path}"
    AudioSegment.converter = f"/tmp/{ffmpeg_path}/ffmpeg"


try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import edge_tts, asyncio
except Exception:
    edge_tts = None
    asyncio = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

AUDIO_DIR = Path("audio_responses"); AUDIO_DIR.mkdir(exist_ok=True)
AUDIO_TTL_HOURS = 6
MAX_CONTEXT_MESSAGES = 15
MODEL_LIST = ["openai/gpt-4o-mini","meta-llama/llama-3.2-3b-instruct","meta-llama/llama-3.2-1b-instruct"]

VOICE_OPTIONS = {
    "Jenny (Female, Calm)": "en-US-JennyNeural",
    "Guy (Male, Warm)": "en-US-GuyNeural",
    "Aria (Female, Friendly)": "en-US-AriaNeural",
    "Davis (Male, Professional)": "en-US-DavisNeural",
    "Neerja (Female, Indian)": "en-IN-NeerjaNeural",
    "Prabhat (Male, Indian)": "en-IN-PrabhatNeural",
}
INTENT_PATTERNS = {
    "greeting": ["hello","hi","hey","good morning","good evening"],
    "meditation": ["meditate","meditation","breathing","breath","relax","calm"],
    "mood_check": ["how am i","feeling","mood","emotions","anxious","stressed"],
    "gratitude": ["grateful","thankful","appreciate","gratitude"],
    "sleep": ["sleep","rest","tired","insomnia","can't sleep"],
    "exercise": ["exercise","workout","movement","yoga","walk"],
    "farewell": ["goodbye","bye","see you","talk later"],
}

st.set_page_config(page_title="Mindful Wellness (Cloud)", page_icon="🪷", layout="wide")

def ensure_state():
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("user_profile", {"voice_preference":"en-US-JennyNeural","response_length":"medium","topics_discussed":[],"total_conversations":0})
    ss.setdefault("conversation_state","idle")
    ss.setdefault("audio_response_path", None)
    ss.setdefault("webrtc_frames", [])
    ss.setdefault("last_uploaded_hash", None)
ensure_state()

def cleanup_old_audio():
    cutoff = datetime.now() - timedelta(hours=AUDIO_TTL_HOURS)
    for f in AUDIO_DIR.glob("response_*.mp3"):
        try:
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
                f.unlink(missing_ok=True)
        except Exception: pass

def autoplay_audio_html(file_path:str):
    try:
        with open(file_path,"rb") as f: b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>", unsafe_allow_html=True)
    except Exception: pass

def detect_intent(txt:str)->str:
    t = txt.lower()
    for intent, pats in INTENT_PATTERNS.items():
        if any(p in t for p in pats): return intent
    return "general"

def update_conversation_stats(intent:str):
    prof = st.session_state.user_profile
    if intent not in prof["topics_discussed"]: prof["topics_discussed"].append(intent)
    prof["total_conversations"] += 1

@st.cache_resource
def init_openai_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key: st.error("Missing OPENROUTER_API_KEY in Secrets."); st.stop()
    if OpenAI is None: st.error("openai package missing."); st.stop()
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
openai_client = init_openai_client()

def get_system_prompt(intent="general", user_profile=None):
    base = ("You are a warm, gentle mindfulness companion named 'Mindful'. "
            "Speak kindly. Keep responses brief and calming. No medical advice.")
    intent_map = {
        "greeting":" Respond warmly and ask how they're feeling today.",
        "meditation":" Offer a short calming breathing tip (1-2 mins).",
        "mood_check":" Acknowledge feelings and suggest one gentle next step.",
        "gratitude":" Encourage a small gratitude reflection.",
        "sleep":" Share simple wind-down advice (no medical).",
        "farewell":" Close kindly and invite them back."
    }
    extra = intent_map.get(intent,"")
    if user_profile:
        rl = user_profile.get("response_length","medium")
        lens = {"short":" Keep it 1–2 sentences.","medium":" Keep it 2–4 sentences.","long":" Use 4–6 sentences."}
        extra += lens.get(rl,"")
    return base+extra

def build_context(intent="general"):
    msgs = st.session_state.messages.copy()
    if len(msgs)>25:
        recent = msgs[-15:]; old = msgs[:-15]
        topics = [detect_intent(m["content"]) for m in old if m["role"]=="user"]
        summary = "Earlier we discussed: " + ", ".join(list(dict.fromkeys(topics))[:5])
        msgs = [{"role":"system","content":get_system_prompt(intent, st.session_state.user_profile)},
                {"role":"system","content":summary}] + recent
    else:
        msgs.insert(0, {"role":"system","content":get_system_prompt(intent, st.session_state.user_profile)})
    if len(msgs)>MAX_CONTEXT_MESSAGES:
        msgs = [msgs[0]] + msgs[-MAX_CONTEXT_MESSAGES:]
    return msgs

def get_ai_reply(user_message:str)->str:
    st.session_state.conversation_state="thinking"
    intent = detect_intent(user_message); update_conversation_stats(intent)
    st.session_state.messages.append({"role":"user","content":user_message})
    ctx = build_context(intent)
    for model in MODEL_LIST:
        try:
            resp = openai_client.chat.completions.create(
                model=model, messages=ctx, max_tokens=250, temperature=0.7,
                extra_headers={"HTTP-Referer":"https://mindful-bot.streamlit.app","X-Title":"Mindful Wellness Chatbot"}
            )
            reply = resp.choices[0].message.content
            st.session_state.messages.append({"role":"assistant","content":reply})
            return reply
        except Exception as e:
            st.warning(f"Model {model} failed: {e}")
            continue
    return "I'm having trouble connecting — but take a deep breath and try again soon."

def text_to_speech(text:str, voice:Optional[str]=None)->Optional[str]:
    st.session_state.conversation_state="speaking"
    voice = voice or st.session_state.user_profile.get("voice_preference","en-US-JennyNeural")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = AUDIO_DIR / f"response_{ts}.mp3"
    try:
        if edge_tts and asyncio:
            async def _run(): await edge_tts.Communicate(text, voice=voice, rate="-10%", pitch="-2Hz").save(str(out))


            try: asyncio.run(_run())
            except RuntimeError:
                import asyncio as _a; loop=_a.new_event_loop(); loop.run_until_complete(_run()); loop.close()
        elif gTTS:
            gTTS(text=text, lang="en").save(str(out))
        else:
            st.warning("No TTS engine available."); st.session_state.conversation_state="idle"; return None
        st.session_state.conversation_state="idle"; return str(out)
    except Exception as e:
        st.error(f"TTS error: {e}"); st.session_state.conversation_state="idle"; return None

from pydub import AudioSegment
import re

def stream_tts_response(text: str, voice: Optional[str] = None):
    """Generate combined audio (Edge TTS + gTTS fallback) and play once cleanly."""
    st.session_state.conversation_state = "speaking"

    # Split text smartly by punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    combined_audio = AudioSegment.silent(duration=300)  # small intro pause
    temp_files = []

    for sentence in sentences:
        path = text_to_speech(sentence, voice)
        if path and os.path.exists(path):
            try:
                seg = AudioSegment.from_file(path, format="mp3")
                combined_audio += seg + AudioSegment.silent(duration=800)  # add pause between sentences
                temp_files.append(path)
            except Exception as e:
                st.warning(f"Audio merge error: {e}")

    # Export combined audio
    final_path = AUDIO_DIR / f"response_combined_{int(time.time())}.mp3"
    combined_audio.export(final_path, format="mp3")

    # Play single audio cleanly
    st.session_state.audio_response_path = str(final_path)
    st.audio(str(final_path), format="audio/mp3")

    # Optional: cleanup temp sentence clips
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass

    st.session_state.conversation_state = "idle"
    return [str(final_path)]

def transcribe_audio_bytes(raw_wav_bytes:bytes)->Optional[str]:
    if sr is None: st.error("SpeechRecognition not installed."); return None
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_wav_bytes); tmp_path = tmp.name
        with sr.AudioFile(tmp_path) as src:
            audio = r.record(src); text = r.recognize_google(audio, language="en-US")
        os.unlink(tmp_path); return text
    except Exception as e:
        st.error(f"Transcription error: {e}"); return None

def audio_upload_to_text(uploaded)->Optional[str]:
    if sr is None: st.error("SpeechRecognition not installed."); return None
    try:
        r = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getbuffer()); tmp_path = tmp.name
        with sr.AudioFile(tmp_path) as src:
            audio = r.record(src); text = r.recognize_google(audio, language="en-US")
        os.unlink(tmp_path); return text
    except Exception as e:
        st.error(f"Couldn't process audio: {e}"); return None

def record_browser_audio_ui():
    st.markdown("#### Browser Voice Recorder")
    st.caption("Click Start to record, Stop when done, then press Transcribe.")
    ctx = webrtc_streamer(key="webrtc-audio", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024,
                          media_stream_constraints={"audio":True,"video":False})
    if ctx and ctx.state.playing:
        frames = ctx.audio_receiver.get_frames(timeout=1)
        for f in frames:
            arr = f.to_ndarray()
            if arr.ndim==2: arr = arr[0]
            pcm = arr.astype("int16").tobytes()
            st.session_state.webrtc_frames.append(pcm)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("Clear Recording Buffer", use_container_width=True):
            st.session_state.webrtc_frames = []; st.success("Cleared recorded audio.")
    with c2:
        if st.button("Transcribe Recording", use_container_width=True):
            if not st.session_state.webrtc_frames: st.warning("No audio recorded yet.")
            else:
                raw_pcm = b"".join(st.session_state.webrtc_frames)
                sample_rate, channels, sampwidth = 48000, 1, 2
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(channels); wf.setsampwidth(sampwidth); wf.setframerate(sample_rate)
                    wf.writeframes(raw_pcm)
                wav_bytes = buf.getvalue()
                with st.spinner("Transcribing recording..."):
                    text = transcribe_audio_bytes(wav_bytes)
                if text:
                    st.success(f"Transcribed: {text}")
                    with st.spinner("Thinking..."): reply = get_ai_reply(text)
                    with st.spinner("Responding..."):
                        paths = stream_tts_response(reply)
                        if paths: st.session_state.audio_response_path = paths[-1]
                    st.rerun()

st.markdown("<div style='text-align:center; padding: 20px;'><h1>Mindful Wellness AI Assistant</h1><p>Streamlit Cloud–ready: browser mic + file upload</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Settings")
    st.markdown("### Voice Selection")
    selected_voice = st.selectbox("Choose AI voice:", options=list(VOICE_OPTIONS.keys()), index=0)
    st.session_state.user_profile["voice_preference"] = VOICE_OPTIONS[selected_voice]
    st.markdown("### Response Length")
    rl = st.radio("Style:", options=["short","medium","long"], index=1, horizontal=True)
    st.session_state.user_profile["response_length"] = rl
    st.markdown("---"); st.markdown("### Your Journey")
    st.metric("Conversations", st.session_state.user_profile["total_conversations"])
    if st.session_state.user_profile["topics_discussed"]:
        st.write("**Topics explored:**")
        for t in st.session_state.user_profile["topics_discussed"][:5]: st.write(f"• {t.capitalize()}")
    st.markdown("---")
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    st.caption("Powered by OpenRouter API")

col1, col2 = st.columns([2,1])
with col1:
    def status(): return {"idle":"Ready","listening":"Listening","thinking":"Thinking","speaking":"Speaking"}.get(st.session_state.conversation_state,"Ready")
    st.markdown(f"### Conversation — {status()}")
    chat = st.container(height=420)
    with chat:
        for m in st.session_state.messages:
            if m["role"]=="user":
                with st.chat_message("user", avatar="🧘"): st.markdown(f"**You:** {m['content']}")
            else:
                with st.chat_message("assistant", avatar="🌿"): st.markdown(m["content"])
    if prompt := st.chat_input("Type your message..."):
        with st.spinner("Reflecting..."): reply = get_ai_reply(prompt)
        with st.spinner("Preparing voice response..."):
            paths = stream_tts_response(reply)
            if paths: st.session_state.audio_response_path = paths[-1]
        st.rerun()

with col2:
    st.markdown("### Voice Controls (Cloud-friendly)")
    record_browser_audio_ui()
    st.markdown("---"); st.markdown("**Or upload a voice note**")
    uploaded_audio = st.file_uploader("Upload audio (wav/mp3/m4a/flac)", type=["wav","mp3","m4a","flac"], label_visibility="collapsed")
    if uploaded_audio:
        file_hash = hashlib.md5(uploaded_audio.getbuffer()).hexdigest()
        if st.session_state.get("last_uploaded_hash") != file_hash:
            st.session_state.last_uploaded_hash = file_hash
            with st.spinner("Processing your voice..."): text = audio_upload_to_text(uploaded_audio)
            if text:
                st.success(f"Transcribed: {text}")
                with st.spinner("Thinking..."): reply = get_ai_reply(text)
                with st.spinner("Responding..."):
                    paths = stream_tts_response(reply)
                    if paths: st.session_state.audio_response_path = paths[-1]
            st.rerun()
        else:
            st.info("This audio has already been processed. Upload a new one to continue.")

if st.session_state.audio_response_path and os.path.exists(st.session_state.audio_response_path):
    st.audio(st.session_state.audio_response_path, format="audio/mp3")

cleanup_old_audio()

st.markdown("<hr/><div style='text-align:center; color:#777; padding: 20px;'><p>Each breath is a new beginning</p></div>", unsafe_allow_html=True)