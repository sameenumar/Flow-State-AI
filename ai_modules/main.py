import cv2
import time
import sys
import os

# Add current directory to path so sibling modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from mp_handler import MediaPipeHandler
from gesture_module.gesture import gesture_agent

# rPPG module — optional (implemented separately)
try:
    from rppg_module.rppg import rPPG_agent
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


def main():
    # ── rPPG frame sampling config ──
    target_fps      = 25
    frame_interval  = 1.0 / target_fps
    last_rppg_time  = 0.0

    # ── Open webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check that camera index 0 is available.")
        return

    # ── Initialise MediaPipe handler ──
    mp_agent = MediaPipeHandler()

    # ── Initialise agent modules ──
    agent_gesture = gesture_agent()

    agent_face = None
    if FACE_AVAILABLE:
        agent_face = FaceAgent()

    agent_rppg = None
    if RPPG_AVAILABLE:
        agent_rppg = rPPG_agent()

    # ── Start daemon threads ──
    agent_gesture.start()
    if agent_face:
        agent_face.start()
    if agent_rppg:
        agent_rppg.start()

    print("[OK] All modules started — press 'q' to quit")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Convert BGR→RGB once; used by both MediaPipe and rPPG
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Prepare MediaPipe image (already does BGR→RGB internally)
        mp_image = mp_agent.prepare(bgr_frame)

        # Timestamps must be monotonically increasing (milliseconds)
        current_time = time.perf_counter()
        timestamp_ms = int(current_time * 1000)

        # ── MediaPipe detections ──
        face_results = mp_agent.process_face(mp_image, timestamp_ms)
        hand_results = mp_agent.process_hands(mp_image, timestamp_ms)

        # ── Send to face module ──
        if agent_face:
            agent_face.enqueue_frame(face_results, bgr_frame)

        # ── Send to gesture module ──
        gesture_frame_data = {
            "hand_landmarks": hand_results.hand_landmarks if hand_results.hand_landmarks else [],
            "hand_gestures":  None,
            # Pass first face's landmarks (list of NormalizedLandmark) or None
            "face_landmarks": face_results.face_landmarks[0] if face_results.face_landmarks else None,
            # FIX: frame_id should just be the timestamp in ms, not divided by frame_interval
            "frame_id": timestamp_ms,
        }
        agent_gesture.enqueue_frame(gesture_frame_data)

        # ── Send to rPPG module at target_fps rate ──
        if agent_rppg and (current_time - last_rppg_time) >= frame_interval:
            agent_rppg.enqueue_frame(rgb_frame)
            last_rppg_time = current_time

        # ── Read latest results from each module ──
        rppg_result    = agent_rppg.latest_result   if agent_rppg    else None
        gesture_result = agent_gesture.latest_result
        face_result    = agent_face.latest_result    if agent_face    else None

        # ── Visualise face landmarks (green dots) ──
        if face_results.face_landmarks:
            h, w = bgr_frame.shape[:2]
            for face_lms in face_results.face_landmarks:
                for lm in face_lms:
                    cv2.circle(bgr_frame,
                               (int(lm.x * w), int(lm.y * h)),
                               1, (0, 255, 0), -1)

        # ── Visualise hand landmarks (blue dots) ──
        if hand_results.hand_landmarks:
            h, w = bgr_frame.shape[:2]
            for hand_lms in hand_results.hand_landmarks:
                for lm in hand_lms:
                    cv2.circle(bgr_frame,
                               (int(lm.x * w), int(lm.y * h)),
                               1, (255, 0, 0), -1)

        # ── On-screen display ──
        display_y = 30

        # rPPG result
        if rppg_result:
            cv2.putText(bgr_frame, f"rPPG: {rppg_result}",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            display_y += 30

        # Face result
        if face_result and face_result.get("face_detected"):
            emotions = face_result.get("emotion_probabilities", {})
            va       = face_result.get("valence_arousal", {})
            metrics  = face_result.get("raw_metrics", {})

            cv2.putText(bgr_frame,
                        f"Face - Joy: {emotions.get('joy', 0):.2f} | "
                        f"Frust: {emotions.get('frustration', 0):.2f} | "
                        f"Fatigue: {emotions.get('fatigue', 0):.2f}",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 200, 200), 2)
            display_y += 25
            cv2.putText(bgr_frame,
                        f"Blink: {metrics.get('blink_rate', 0)} bpm | "
                        f"EyeClosure: {metrics.get('eye_closure_index', 0):.2f} | "
                        f"V: {va.get('valence', 0):.2f} A: {va.get('arousal', 0):.2f}",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 150, 200), 1)
            display_y += 30

        # Gesture result
        if gesture_result and gesture_result.get("hands_detected"):
            state  = gesture_result.get("state_tags", {})
            probs  = gesture_result.get("probabilities", {})
            raw    = gesture_result.get("raw_metrics", {})

            cv2.putText(bgr_frame,
                        f"Gesture - Fidget: {state.get('fidget_level', 0):.2f} | "
                        f"FaceTouch: {state.get('face_touching', False)} | "
                        f"Typing: {state.get('typing_cadence', 'none')}",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 80, 0), 2)
            display_y += 25
            cv2.putText(bgr_frame,
                        f"Probs - Fidget: {probs.get('prob_fidgeting', 0):.2f} | "
                        f"Typing: {probs.get('prob_typing', 0):.2f} | "
                        f"FaceTouch: {probs.get('prob_face_touching', 0):.2f} | "
                        f"Slump: {state.get('posture_slump', 0):.2f}",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 140, 0), 1)
        else:
            cv2.putText(bgr_frame, "[*] Detecting hands...",
                        (10, display_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (100, 100, 255), 2)

        cv2.imshow("Main Stream", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Clean shutdown ──
    agent_gesture.stop()
    if agent_face:
        agent_face.stop()

    cap.release()
    cv2.destroyAllWindows()
    mp_agent.close()
    print("[OK] Shutdown complete.")


if __name__ == "__main__":
    main()
