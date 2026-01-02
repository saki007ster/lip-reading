import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import os
import sentencepiece
import sys
import argparse

# Add Auto-AVSR repository to sys.path
auto_avsr_path = os.path.join(os.path.dirname(__file__), "auto_avsr")
if auto_avsr_path not in sys.path:
    sys.path.insert(0, auto_avsr_path)

class LipReader:
    def __init__(self, model_path="models/auto_avsr_weights.pt"):
        import torch # Lazy import
        import gc
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_module = None
        
        print("DEBUG: Lazy loading heavy model components...")
        try:
            from lightning import ModelModule
            from datamodule.transforms import TextTransform
        except Exception as e:
            print(f"DEBUG: Import failed: {e}")
            ModelModule = None
            TextTransform = None

        if ModelModule and os.path.exists(model_path):
            try:
                # Setup arguments for ModelModule
                parser = argparse.ArgumentParser()
                args, _ = parser.parse_known_args(args=[])
                args.modality = "video"
                args.ctc_weight = 0.1
                args.lr = 1e-3
                args.weight_decay = 0.05
                args.warmup_epochs = 5
                args.max_epochs = 75
                args.pretrained_model_path = model_path
                
                print("DEBUG: Initializing ModelModule (this may take a moment)...")
                self.model_module = ModelModule(args)
                gc.collect()
                
                if self.device.type == "cpu":
                    print("DEBUG: Applying dynamic quantization...")
                    self.model_module = torch.quantization.quantize_dynamic(
                        self.model_module, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    gc.collect()
                
                self.model_module.to(self.device).eval()
                
                # Setup TextTransform
                if hasattr(self.model_module, 'text_transform'):
                    self.text_transform = self.model_module.text_transform
                elif TextTransform:
                    self.text_transform = TextTransform()
                
                gc.collect()
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model_module = None
        
        # Ensure text_transform
        if not hasattr(self, 'text_transform'):
            class BasicTextTransform:
                def post_process(self, token_ids): return f"Tokens: {token_ids.tolist()}"
            self.text_transform = BasicTextTransform()
                print(f"Error loading model architecture: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not ModelModule:
                print("Auto-AVSR architecture not found. Run in STUB mode.")
            else:
                print(f"Model weights not found at {model_path}. Running in STUB mode.")
        
        # Ensure text_transform is always initialized, even in stub mode
        if not hasattr(self, 'text_transform'):
            # Fallback to a basic TextTransform if nothing else worked
            class BasicTextTransform:
                def post_process(self, token_ids):
                    return f"Tokens: {token_ids.tolist()}"
            self.text_transform = BasicTextTransform()


    @property
    def model(self):
        return self.model_module

    def preprocess(self, frames):
        """
        Preprocess frames for Auto-AVSR:
        Input: List of BGR images (96x96 from vision.py)
        Output: Normalized tensor (T, 1, 88, 88)
        """
        processed = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Center Crop to 88x88 from 96x96
            h, w = gray.shape
            ch, cw = 88, 88
            y1 = (h - ch) // 2
            x1 = (w - cw) // 2
            cropped = gray[y1:y1+ch, x1:x1+cw]
            
            # Normalize based on Auto-AVSR training specs (mean=0.421, std=0.165)
            norm = (cropped.astype(np.float32) / 255.0 - 0.421) / 0.165
            processed.append(norm)
        
        # Shape: (T, 1, 88, 88)
        tensor = torch.FloatTensor(np.array(processed)).unsqueeze(1)
        return tensor.to(self.device)

    def predict(self, frames):
        if not frames:
            return ""

        if self.model_module is None:
            import random
            return f"Stub predicted: {random.choice(['Hello', 'How are you', 'Lip reading active'])}"
            
        try:
            input_tensor = self.preprocess(frames)
            with torch.no_grad():
                # ModelModule.forward (LightningModule) returns the decoded string
                prediction = self.model_module(input_tensor)
            return prediction if prediction else "..."
        except Exception as e:
            return f"Inference Error: {str(e)}"

if __name__ == "__main__":
    reader = LipReader()
    dummy_frames = [np.random.randint(0, 255, (50, 50, 3)) for _ in range(30)]
    print(f"Predicted text: {reader.predict(dummy_frames)}")
