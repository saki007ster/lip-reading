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

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import queue

import threading

# Initialize components once in session_state
if 'detector' not in st.session_state:
    st.session_state.detector = LipDetector()
if 'reader' not in st.session_state:
    st.session_state.reader = LipReader()
if 'tts' not in st.session_state:
    st.session_state.tts = TextToSpeech()
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()

detector = st.session_state.detector
reader = st.session_state.reader
tts = st.session_state.tts
result_queue = st.session_state.result_queue

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

# Dynamic Warning
if reader.model is None:
    st.warning("**Current Model Status:** Using a **Simulated Stub**. Please ensure weights are in `models/`.")
else:
    st.success("**Model weights loaded!** Ready for real-time inference (Background Processing Active).")

# Layout: 2 columns for video feeds
col1, col2 = st.columns(2)
with col1:
    st.subheader("Webcam Stream")
with col2:
    st.subheader("Extracted Mouth")
    mouth_feed = st.empty()

st.divider()
st.subheader("Transcription")
transcription_box = st.empty()

# WebRTC Video Processor
class LipReadingProcessor(VideoTransformerBase):
    def __init__(self, input_queue):
        self.frame_buffer = []
        self.frame_count = 0
        self.last_mouth_crop = None
        self.input_queue = input_queue
        
        # Constants from implementation
        self.WINDOW_SIZE = 30
        self.STEP_SIZE = 15

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        try:
            # Process frame
            mouth_crop, box = detector.get_lip_region(img)
            
            if mouth_crop is not None and mouth_crop.size > 0:
                self.last_mouth_crop = mouth_crop
                self.frame_buffer.append(mouth_crop)
                self.frame_count += 1
                
                if len(self.frame_buffer) > self.WINDOW_SIZE:
                    self.frame_buffer.pop(0)
                
                # Inference trigger: Put a copy of the buffer into the background queue
                if len(self.frame_buffer) == self.WINDOW_SIZE and self.frame_count % self.STEP_SIZE == 0:
                    # We send a COPY of the buffer list to avoid mutation issues in the thread
                    self.input_queue.put(list(self.frame_buffer))
                
                # Draw box on main frame
                img = detector.draw_landmarks(img, box)
        except Exception as e:
            pass

        return img

# Streamer
webrtc_ctx = webrtc_streamer(
    key="lip-reading",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=lambda: LipReadingProcessor(st.session_state.input_queue),
    async_transform=True,
    media_stream_constraints={"video": True, "audio": False},
)

# Handle Results and Display Aligned Mouth
if webrtc_ctx.video_transformer:
    # Display the latest aligned mouth crop in the side feed
    if webrtc_ctx.video_transformer.last_mouth_crop is not None:
        mouth_feed.image(webrtc_ctx.video_transformer.last_mouth_crop, channels="BGR", width="stretch")
    
    # Check for new transcriptions from the background thread
    try:
        while not st.session_state.result_queue.empty():
            new_text = st.session_state.result_queue.get_nowait()
            st.session_state.last_text = new_text
            tts.speak(new_text)
    except:
        pass

if 'last_text' in st.session_state:
    transcription_box.success(f"**Predicted:** {st.session_state.last_text}")

st.sidebar.markdown("""
### Instructions:
1. Allow camera access.
2. Click **Start** in the video player.
3. Speak clearly toward the camera.
4. Aligned mouth view shows what the model 'sees'.
""")
