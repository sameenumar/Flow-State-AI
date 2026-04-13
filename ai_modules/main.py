import cv2
import time
import sys
import os
import queue
from mp_handler import MediaPipeThread

# Add current directory to path so sibling modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# rPPG module — optional (implemented separately)
try:
    from rPPG_module.rppg import rPPG_agent
    RPPG_AVAILABLE = True
except ImportError:
    RPPG_AVAILABLE = False
    print("[WARNING] rPPG module not yet implemented — skipping heart rate detection")

# Face module — optional (graceful degradation if missing)
try:
    from face_module.face_detection import FaceAgent
    FACE_AVAILABLE = True
except ImportError:
    FACE_AVAILABLE = False
    print("[WARNING] Face module not found — skipping face detection")

# Hand Gesture module
try:
    from gesture_module.gesture import gesture_agent
    GESTURE_AVAILABLE = True
except ImportError:
    GESTURE_AVAILABLE = False
    print("[WARNING] Hand gestures module not found — skipping gesture detection")

def main():
   # ── Open webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that camera index 0 is available.")
        return
    
     # ── rPPG frame sampling config ──
    # actual_fps = cap.get(cv2.CAP_PROP_FPS)
    # target_fps = actual_fps if actual_fps < 30 else 30
    target_fps = 25
    frame_interval  = 1.0 / target_fps
    last_rppg_time  = 0.0

    # memory buffer -> connects mp_handler and main scripts
    mp_results_queue = queue.Queue(maxsize=2)
    mp_thread = MediaPipeThread(mp_results_queue)


    # ── Initialise modules ──
    mp_thread.start()

    agent_gesture = None
    if GESTURE_AVAILABLE:
        agent_gesture = gesture_agent()

    agent_face = None
    if FACE_AVAILABLE:
        agent_face = FaceAgent()

    agent_rppg = None
    if RPPG_AVAILABLE:
        agent_rppg = rPPG_agent(fps=int(target_fps))

    # ── Start daemon threads ──
    if agent_gesture:
        agent_gesture.start()
    if agent_face:
        agent_face.start()
    if agent_rppg:
        agent_rppg.start()

    print("[OK] All modules started — press 'q' to quit")

    face_results = None
    hand_results = None
    latest_bgr = None

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break
 
        # Feed the raw BGR frame to MediaPipeThread with a monotonic timestamp.
        # MediaPipeThread handles BGR→RGB conversion internally via prepare().
        # ts_ms must be strictly increasing — perf_counter() guarantees this.
        ts_ms = int(time.perf_counter() * 1000)
        mp_thread.enqueue_frame(bgr_frame, ts_ms)
 
        # Non-blocking read of the latest MediaPipe results.
        # If MediaPipe hasn't finished yet, queue.Empty is raised and we keep
        # the previous frame's results for display — no blocking, no stalling.
        try:
            latest_bgr, face_results, hand_results = mp_results_queue.get_nowait()
        except queue.Empty:
            pass
 
        # ── Feed face agent ──
        # FaceAgent.enqueue_frame() expects (MediaPipe FaceLandmarkerResult, bgr_frame).
        # It extracts blendshapes from face_results internally.
        if agent_face and face_results is not None:
            agent_face.enqueue_frame(face_results, latest_bgr)
 
        # ── Feed gesture agent ──
        # gesture_agent.enqueue_frame() expects a dict with specific keys.
        # hand_landmarks: list of landmark lists (one per detected hand)
        # face_landmarks: first face's landmark list (or None) — used for
        #                 face-touching proximity detection
        # frame_id: timestamp in ms (used for ordering, not timing math)
        if face_results is not None and hand_results is not None:
            gesture_frame_data = {
                "hand_landmarks": hand_results.hand_landmarks if hand_results.hand_landmarks else [],
                "hand_gestures":  None,
                "face_landmarks": face_results.face_landmarks[0] if face_results.face_landmarks else None,
                "frame_id":       ts_ms,
            }
            agent_gesture.enqueue_frame(gesture_frame_data)
 
        # ── Feed rPPG agent at exactly target_fps ──
        # We send latest_bgr (the frame MediaPipe processed) not bgr_frame
        # (the current raw capture frame). This keeps pixel data and landmark
        # coordinates perfectly in sync — they describe the same moment in time.
        # .copy() prevents a race condition: main thread draws on bgr_frame
        # while rPPG worker reads it for ROI extraction simultaneously.
        if agent_rppg and face_results is not None and latest_bgr is not None:
            now = time.perf_counter()
            if (now - last_rppg_time) >= frame_interval:
                agent_rppg.enqueue_frame(latest_bgr.copy(), face_results.face_landmarks)
                last_rppg_time = now
 
        # ── Read latest results from each module ──
        rppg_result = agent_rppg.latest_result if agent_rppg else None
        gesture_result = agent_gesture.latest_result
        face_result = agent_face.latest_result if agent_face else None
 

    # ── Clean shutdown ──
    # stop() sets the stop event and pushes a sentinel None into the queue
    # so the thread's queue.get() unblocks immediately rather than waiting
    # for the timeout. Order: stop workers first, then release hardware.
    mp_thread.stop()
    if agent_gesture:
        agent_gesture.stop()
    if agent_face:
        agent_face.stop()
    if agent_rppg:
        agent_rppg.stop()
 
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Shutdown complete.")

if __name__ == "__main__":
    main()