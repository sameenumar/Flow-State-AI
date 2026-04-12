import cv2
import queue
import threading
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
            num_faces=1,

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

class MediaPipeThread(threading.Thread):
    def __init__(self, output_queue, face_model_path="./models/face_landmarker.task", hand_model_path="./models/hand_landmarker.task"):
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
        self.mp_agent.close()