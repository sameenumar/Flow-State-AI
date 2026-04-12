import cv2
import time
import queue
from mp_handler import MediaPipeThread
from rPPG_module.rPPG import rPPG_agent
# from face_module import face_detection
# from gesture_module import gesture


def main():
    # rppg framing variables
    target_fps = 25
    frame_interval = 1.0 / target_fps
    last_sent_time = 0

    cap = cv2.VideoCapture(0)
    
    # memory buffer -> connects mp_handler and main scripts
    mp_results_queue = queue.Queue(maxsize=2)
    mp_thread = MediaPipeThread(mp_results_queue)

    # initializing model objects
    # agent_gesture = gesture_agent()
    # agent_face = face_agent()
    agent_rppg = rPPG_agent()

    # starting the thread-based objects
    mp_thread.start()
    agent_rppg.start()

    face_results = None
    hand_results = None
    latest_bgr = None

    while True:
        ret, bgr_frame = cap.read()
        if not ret: break

        # downscaling the frame:
        # 1. to reduce cpu load
        # 2. to cancel sensor noise
        # 3. to average adjacent pixels
        # (experimental)
        # bgr_frame = cv2.resize(bgr_frame, (640, 480), interpolation=cv2.INTER_AREA)

        ts_ms = int(time.perf_counter() * 1000)
        mp_thread.enqueue_frame(bgr_frame, ts_ms)

        # process_face() & process_hands() expect
        # timestamp in milliseconds as args
        try:
            latest_bgr, face_results, hand_results = mp_results_queue.get_nowait()
        except queue.Empty:
            pass

        if face_results is not None:
            now = time.perf_counter()
            if (now - last_sent_time) >= frame_interval:
                agent_rppg.enqueue_frame(latest_bgr, face_results.face_landmarks)
                last_sent_time = now

        

if __name__ == "__main__":
    main()