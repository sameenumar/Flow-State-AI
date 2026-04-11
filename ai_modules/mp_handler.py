import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandler:
    def __init__(self, face_model_path=None, hand_model_path=None):
        # Resolve model paths relative to THIS file's directory
        # This works correctly no matter where you run the script from
        module_dir = os.path.dirname(os.path.abspath(__file__))

        if face_model_path is None:
            face_model_path = os.path.join(module_dir, "models", "face_landmarker.task")
        if hand_model_path is None:
            hand_model_path = os.path.join(module_dir, "models", "hand_landmarker.task")

        # Verify files exist before loading — gives a clear error if models are missing
        if not os.path.isfile(face_model_path):
            raise FileNotFoundError(
                f"[MediaPipeHandler] Face model not found at: {face_model_path}\n"
                f"Make sure face_landmarker.task is inside the 'models/' folder next to mp_handler.py"
            )
        if not os.path.isfile(hand_model_path):
            raise FileNotFoundError(
                f"[MediaPipeHandler] Hand model not found at: {hand_model_path}\n"
                f"Make sure hand_landmarker.task is inside the 'models/' folder next to mp_handler.py"
            )

        BaseOptions = mp.tasks.BaseOptions

        # Face landmarker — blendshapes enabled for emotion detection
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        self.face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(self.face_options)

        # Hand landmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        self.hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(self.hand_options)

        print(f"[MediaPipeHandler] Models loaded from: {module_dir}/models/")

    def prepare(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def process_face(self, mp_image, timestamp_ms):
        return self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

    def process_hands(self, mp_image, timestamp_ms):
        return self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.face_landmarker.close()
        self.hand_landmarker.close()