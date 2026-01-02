import streamlit as st
import cv2
import numpy as np
from vision import LipDetector
from model import LipReader
from audio import TextToSpeech
import time

# Page configuration
st.set_page_config(page_title="Lip Reading POC", layout="wide")

st.title("👄 Lip Reading Communication Assistant")
st.markdown("""
This application captures your lip movements and converts them into speech. 
Move your lips clearly in front of the camera.
""")

# Initialize components
if 'detector' not in st.session_state:
    st.session_state.detector = LipDetector()
if 'reader' not in st.session_state:
    st.session_state.reader = LipReader()
if 'tts' not in st.session_state:
    st.session_state.tts = TextToSpeech()
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []

detector = st.session_state.detector
reader = st.session_state.reader
tts = st.session_state.tts

# Dynamic Warning
if reader.model is None:
    st.warning("**Current Model Status:** Using a **Simulated Stub**. Please download weights as per README.")
else:
    st.success("**Model weights loaded!** Running real-time inference.")

# Sidebar info
st.sidebar.header("Settings")
buffer_size = st.sidebar.slider("Frame Buffer Size", 15, 60, 30)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Layout: 2 columns for video feeds
col1, col2 = st.columns(2)

with col1:
    st.subheader("Main Feed")
    main_feed = st.empty()

with col2:
    st.subheader("Extracted Mouth")
    mouth_feed = st.empty()

st.divider()
st.subheader("Transcription")
transcription_box = st.empty()
last_transcription = st.empty()

# Webcam Loop
run_app = st.checkbox("Start Webcam", value=True)
cap = cv2.VideoCapture(0)

# Constants for synchronization and overlapping buffers
TARGET_FPS = 25
FRAME_TIME = 1.0 / TARGET_FPS
WINDOW_SIZE = 30  # 1.2 seconds of context
STEP_SIZE = 15     # Run inference every 0.6 seconds

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

while run_app and cap.isOpened():
    start_time = time.perf_counter()
    
    ret, frame = cap.read()
    if not ret:
        st.error("Could not access webcam.")
        break

    try:
        # Detect mouth with alignment
        mouth_crop, box = detector.get_lip_region(frame)
    except Exception as e:
        if "shutdown" in str(e).lower():
            break
        st.error(f"Detection error: {e}")
        break

    if mouth_crop is not None and mouth_crop.size > 0:
        # Update buffer
        st.session_state.frame_buffer.append(mouth_crop)
        st.session_state.frame_count += 1
        
        # Keep buffer limited to WINDOW_SIZE
        if len(st.session_state.frame_buffer) > WINDOW_SIZE:
            st.session_state.frame_buffer.pop(0)
            
        # Run inference periodically (Overlapping)
        if len(st.session_state.frame_buffer) == WINDOW_SIZE and st.session_state.frame_count % STEP_SIZE == 0:
            with st.spinner("Transcribing..."):
                text = reader.predict(st.session_state.frame_buffer)
                if text and text != "...":
                    transcription_box.success(f"**Predicted:** {text}")
                    tts.speak(text)
        
        # Display mouth crop (now aligned)
        mouth_feed.image(mouth_crop, channels="BGR", width="stretch")
        
        # Draw on main frame
        frame = detector.draw_landmarks(frame, box)

    # Display main feed
    if frame is not None and frame.size > 0:
        main_feed.image(frame, channels="BGR", width="stretch")
    
    # Precise synchronization to 25 FPS
    elapsed = time.perf_counter() - start_time
    sleep_time = max(0, FRAME_TIME - elapsed)
    time.sleep(sleep_time)

cap.release()
