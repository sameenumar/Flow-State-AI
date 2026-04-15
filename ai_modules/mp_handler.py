import os
import cv2
import queue
import threading
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
            running_mode=vision.RunningMode.VIDEO, # Use VIDEO for webcam
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False

            # mediapipe 2-stage architecture:
            # stage 1: the detector (heavy) - scans images from scratch
            # stage 2: the tracker (light) - predicts landmarks
            # relying on the tracker more than the detector significantly increases performance

            # implementation of the concept as mentioned above.
            # min_face_detection_confidence=0.6,  # High Threshold (Heavy Model)
            # min_face_presence_confidence=0.5,   # Tracking Threshold (Light Model)
            # min_tracking_confidence=0.5         # Tracking Threshold (Light Model)
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
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    def process_face(self, mp_image, timestamp_ms):
        return self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

    def process_hands(self, mp_image, timestamp_ms):
        return self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self.face_landmarker.close()
        self.hand_landmarker.close()

class MediaPipeThread(threading.Thread):
    def __init__(self, output_queue, face_model_path=None, hand_model_path=None):
        super().__init__()
        self.daemon = True
        self.mp_agent = MediaPipeHandler(face_model_path, hand_model_path)
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = output_queue
        self.running = False

    def enqueue_frame(self, bgr_frame, ts_ms):
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
        self.input_queue.put((bgr_frame, ts_ms))

    def run(self):
        self.running = True
        while self.running:
            try:
                bgr_frame, ts_ms = self.input_queue.get(timeout=0.1)
                if bgr_frame is None:  # sentinel
                    break

                # BGR -> RGB conversion
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = self.mp_agent.prepare(rgb_frame)

                face_results = self.mp_agent.process_face(mp_image, ts_ms)
                hand_results = self.mp_agent.process_hands(mp_image, ts_ms)

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.output_queue.put((bgr_frame, face_results, hand_results))

            except queue.Empty:
                continue

def stop(self):
    self.running = False
    try:
        self.input_queue.put_nowait(None)
    except queue.Full:
        pass
    self.join(timeout=1.0)
    self.mp_agent.close()