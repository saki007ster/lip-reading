import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class LipDetector:
    def __init__(self, model_path="models/face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Lip landmark indices in MediaPipe Face Mesh / Tasks
        self.LIP_INDICES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
        ]

    def get_lip_region(self, frame):
        """
        Detects face and returns a CROPPED AND ALIGNED image of the lip region.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return None, None

        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape

        # Official Auto-AVSR style alignment (Similarity Transform)
        # Stable points: Left Eye, Right Eye, Nose Tip, Mouth Center
        # MediaPipe Indices mapping:
        # Left Eye Center: mean of [33, 133, 159, 145]
        # Right Eye Center: mean of [362, 263, 386, 374]
        # Nose Tip: 1
        # Mouth Center: 13
        
        def get_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        left_eye = np.mean([get_pt(33), get_pt(133), get_pt(159), get_pt(145)], axis=0)
        right_eye = np.mean([get_pt(362), get_pt(263), get_pt(386), get_pt(374)], axis=0)
        nose_tip = get_pt(1)
        mouth_center = get_pt(13)

        src_pts = np.array([left_eye, right_eye, nose_tip, mouth_center], dtype=np.float32)
        
        # Target points (normalized 256x256 reference frame)
        # These are rough estimates matching the '20words_mean_face.npy' logic
        # Where eyes are approx at y=0.3, nose at y=0.5, mouth at y=0.7
        dst_pts = np.array([
            [0.35 * 256, 0.35 * 256], # Left Eye
            [0.65 * 256, 0.35 * 256], # Right Eye
            [0.50 * 256, 0.55 * 256], # Nose Tip
            [0.50 * 256, 0.75 * 256]  # Mouth Center
        ], dtype=np.float32)

        # Calculate Similarity Transform
        tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if tform is None:
            return None, None

        # Warp the full frame to aligned position
        aligned_frame = cv2.warpAffine(frame, tform, (256, 256))
        
        # Crop 96x96 around the now-aligned mouth center (0.5, 0.75) * 256
        # cx = 128, cy = 192
        cx, cy = 128, 192
        half_w, half_h = 48, 48
        
        mouth_crop = aligned_frame[cy-half_h:cy+half_h, cx-half_w:cx+half_w]
        
        # Return mouth crop and the original mouth bounding box for UI drawing
        # For drawing, we'll just return the min/max of the mouth indices
        lip_coords = []
        for idx in self.LIP_INDICES:
            lm = landmarks[idx]
            lip_coords.append((int(lm.x * w), int(lm.y * h)))
        lip_coords = np.array(lip_coords)
        x, y, mouth_w, mouth_h = cv2.boundingRect(lip_coords)
        
        return mouth_crop, (x, y, x+mouth_w, y+mouth_h)

    def draw_landmarks(self, frame, landmarks_box):
        """
        Draws a bounding box around the lips.
        """
        if landmarks_box:
            x1, y1, x2, y2 = landmarks_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = LipDetector()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        mouth_crop, box = detector.get_lip_region(frame)
        if mouth_crop is not None:
            cv2.imshow("Mouth Crop", mouth_crop)
            frame = detector.draw_landmarks(frame, box)
            
        cv2.imshow("Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
