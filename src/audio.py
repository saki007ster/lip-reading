import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
        except Exception as e:
            print(f"Warning: Could not initialize TTS engine: {e}")
            self.engine = None

    def speak(self, text):
        """
        Speak the provided text. Runs in a separate thread to avoid blocking.
        """
        if not self.engine:
            return
            
        def run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Warning: TTS speak failed: {e}")
        
        thread = threading.Thread(target=run)
        thread.start()

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Testing the lip reading audio output.")
