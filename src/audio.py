import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        # On macOS, 'nsspeech' is often used. On Windows 'sapi5'.
        # self.engine.setProperty('voice', ...) 

    def speak(self, text):
        """
        Speak the provided text. Runs in a separate thread to avoid blocking.
        """
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        
        thread = threading.Thread(target=run)
        thread.start()

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Testing the lip reading audio output.")
