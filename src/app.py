import streamlit as st
import cv2
import numpy as np
from vision import LipDetector
from audio import TextToSpeech
import time

# Page configuration
# Page configuration
st.set_page_config(page_title="LipSync AI | Premium Assistant", layout="wide", page_icon="👄")

# Premium UI CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Gradient Background */
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1a2e 0%, #0f0f1a 100%);
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 20px;
    }

    /* Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 600;
        background: linear-gradient(90deg, #ff007a, #7a00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Animated Recording Pulse */
    .pulse-container {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }

    .pulse {
        width: 12px;
        height: 12px;
        background: #ff007a;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(255, 0, 122, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 122, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 0, 122, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 122, 0); }
    }

    /* Transcription Box */
    .transcription-area {
        background: rgba(0, 0, 0, 0.3);
        border-left: 4px solid #7a00ff;
        padding: 20px;
        border-radius: 0 15px 15px 0;
        min-height: 100px;
        font-size: 1.5rem;
        color: #fff;
    }

    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.02);
    }
</style>
""", unsafe_content_allowed=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #ff007a;'>LipSync AI</h2>", unsafe_content_allowed=True)
    st.sidebar.info("Streamlit Cloud Free Tier has a **1GB RAM limit**. The full model is ~1GB.")
    
    st.markdown("### ⚙️ Engine Settings")
    model_mode = st.selectbox(
        "Model Processing Engine",
        ["Safe Mode (Stub - High Stability)", "Full Auto-AVSR (May Crash Cloud)"],
        index=0,
        help="Safe mode loads instantly. Full mode requires >1GB RAM."
    )
    stub_mode = (model_mode == "Safe Mode (Stub - High Stability)")
    
    st.divider()
    st.markdown("### ℹ️ Instructions")
    st.write("1. Allow camera access\n2. Click **Start** button\n3. Speak clearly to the camera")

# Header section
st.markdown("<h1 class='main-title'>LipSync AI Assistant</h1>", unsafe_content_allowed=True)
st.markdown("<p style='color: rgba(255,255,255,0.6); font-size: 1.1rem;'>Real-time AI-powered communication from lip movement</p>", unsafe_content_allowed=True)

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import queue
import threading
import av

@st.cache_resource
def get_detector():
    return LipDetector()

@st.cache_resource
def get_reader(is_stub):
    if is_stub:
        class StubReader:
            def __init__(self): self.model = None
            def predict(self, frames): return "Safe Mode (Simulator Active)"
        return StubReader()
    
    try:
        from model import LipReader
        return LipReader()
    except Exception as e:
        st.error(f"Failed to load full model: {e}")
        return None

@st.cache_resource
def get_tts():
    return TextToSpeech()

detector = get_detector()
reader = get_reader(stub_mode)
tts = get_tts()

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# Background Inference Worker
def inference_worker(input_queue, output_queue):
    while True:
        frames = input_queue.get()
        if frames is None: break
        try:
            text = reader.predict(frames)
            if text and text != "...":
                output_queue.put(text)
        except Exception as e:
            print(f"DEBUG: Background Inference Error: {e}")
        finally:
            input_queue.task_done()

if 'inference_thread' not in st.session_state:
    st.session_state.input_queue = queue.Queue()
    thread = threading.Thread(
        target=inference_worker, 
        args=(st.session_state.input_queue, st.session_state.result_queue),
        daemon=True
    )
    thread.start()
    st.session_state.inference_thread = thread

# Dashboard Layout
col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown("<div class='glass-card'>", unsafe_content_allowed=True)
    st.markdown("""
        <div class='pulse-container'>
            <div class='pulse'></div>
            <span style='color: #ff007a; font-weight: 600;'>LIVE FEED ACTIVE</span>
        </div>
    """, unsafe_content_allowed=True)
    
    # WebRTC Video Processor
    class LipReadingProcessor:
        def __init__(self):
            self.frame_buffer = []
            self.frame_count = 0
            self.last_mouth_crop = None
            self.WINDOW_SIZE = 30
            self.STEP_SIZE = 15

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                mouth_crop, box = detector.get_lip_region(img)
                if mouth_crop is not None and mouth_crop.size > 0:
                    self.last_mouth_crop = mouth_crop
                    self.frame_buffer.append(mouth_crop)
                    self.frame_count += 1
                    if len(self.frame_buffer) > self.WINDOW_SIZE:
                        self.frame_buffer.pop(0)
                    if len(self.frame_buffer) == self.WINDOW_SIZE and self.frame_count % self.STEP_SIZE == 0:
                        if st.session_state.input_queue:
                            st.session_state.input_queue.put(list(self.frame_buffer))
                    img = detector.draw_landmarks(img, box)
            except: pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="lip-sync",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LipReadingProcessor,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.markdown("</div>", unsafe_content_allowed=True)

with col_side:
    st.markdown("<div class='glass-card'>", unsafe_content_allowed=True)
    st.markdown("<p style='font-weight: 600; font-size: 0.9rem; color: rgba(255,255,255,0.5);'>MOUTH FOCUS</p>", unsafe_content_allowed=True)
    mouth_feed = st.empty()
    st.markdown("</div>", unsafe_content_allowed=True)
    
    st.markdown("<div class='glass-card'>", unsafe_content_allowed=True)
    st.markdown("<p style='font-weight: 600; font-size: 0.9rem; color: rgba(255,255,255,0.5);'>STABILITY STATUS</p>", unsafe_content_allowed=True)
    if stub_mode:
        st.warning("Simulator Mode")
    else:
        st.success("AI Model Active")
    st.markdown("</div>", unsafe_content_allowed=True)

# Transcription Centerpiece
st.markdown("<div class='glass-card'>", unsafe_content_allowed=True)
st.markdown("<p style='font-weight: 600; font-size: 0.9rem; color: rgba(255,255,255,0.5);'>AI TRANSCRIPTION</p>", unsafe_content_allowed=True)
transcription_area = st.empty()

# Handle Results
if webrtc_ctx.video_processor:
    if webrtc_ctx.video_processor.last_mouth_crop is not None:
        mouth_feed.image(webrtc_ctx.video_processor.last_mouth_crop, channels="BGR", use_container_width=True)
    
    try:
        while not st.session_state.result_queue.empty():
            new_text = st.session_state.result_queue.get_nowait()
            st.session_state.last_text = new_text
            tts.speak(new_text)
    except: pass

last_text = st.session_state.get('last_text', "Ready to translate your lips...")
transcription_area.markdown(f"<div class='transcription-area'>{last_text}</div>", unsafe_content_allowed=True)
st.markdown("</div>", unsafe_content_allowed=True)
