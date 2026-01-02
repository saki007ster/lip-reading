# Lip Reading Communication Assistant (POC)

This is a proof-of-concept application designed for users who cannot speak but can move their lips. It uses computer vision to detect mouth movements and converts them into synthesized speech.

## Features
- **Real-time Detection:** Uses MediaPipe Face Mesh to extract the mouth region.
- **Lip Reading (Simulated):** A stub for AVSR (Automatic Visual Speech Recognition) that can be extended with models like Auto-AVSR or Chaplin.
- **Text-to-Speech:** Integrated TTS using `pyttsx3` for immediate audio feedback.
- **Interactive UI:** Built with Streamlit for a live dashboard experience.

## Project Structure
```text
.
├── data/           # Storage for sample videos or logged data
├── models/         # Weights for AVSR models (e.g., .pt or .onnx files)
├── src/            # Source code
│   ├── app.py      # Streamlit dashboard
│   ├── vision.py   # Mouth extraction logic
│   ├── model.py    # AVSR model integration
│   └── audio.py    # TTS utilities
├── requirements.txt # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   cd "Lip Reading"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **System Dependencies:**
   - On macOS, you might need to grant camera access to your terminal/IDE.
   - For TTS, `pyttsx3` uses native system engines. Ensure your speakers are on.

## Running the App

Run the Streamlit application from the root directory:

```bash
streamlit run src/app.py
```

## How to Test
1. Start the application.
2. Ensure your face is clearly visible in the webcam feed.
3. You will see a green bounding box around your lips.
4. The "Extracted Mouth" column will show a zoomed-in view of your lips.
5. Once the frame buffer (default 30 frames) is full, the model will "predict" text and speak it aloud.
6. The predicted text will appear in the "Transcription" section.

## Future Model Integration (Auto-AVSR)
To integrate real Auto-AVSR predictions:
1. **Download Weights**: Obtain pre-trained model weights (e.g., `LRS3_V_WER30.7.pth`) from the official [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) repository.
2. **Setup**: Place the `.pth` or `.pt` file in the `models/` directory.
3. **Configure**: Rename the file to `auto_avsr_weights.pt` or update the path in `src/model.py`.
4. **Environment**: Ensure you have `torch` and `torchvision` installed (included in `requirements.txt`).
