import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeHandler:
    def __init__(self, face_model_path="./models/face_landmarker.task", hand_model_path="./models/hand_landmarker.task"):
        BaseOptions = mp.tasks.BaseOptions
        
        # face landmarking config
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        self.face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=vision.RunningMode.VIDEO, # Use VIDEO for webcam
            num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.face_options)

        # hand landmarking config
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        self.hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(self.hand_options)

    def prepare(self, frame):
        # MediaPipe Tasks requires mp.Image
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    def process_face(self, mp_image, timestamp_ms):
        return self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

    def process_hands(self, mp_image, timestamp_ms):
        return self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.face_landmarker.close()
        self.hand_landmarker.close()