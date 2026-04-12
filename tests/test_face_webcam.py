"""
Webcam test for FaceAgent (face_detection.py).
Run from project root:
    python tests/test_face_webcam.py

What you'll see on screen:
  - Green dots on face landmarks
  - Live readout of joy, frustration, fatigue, blink rate, eye closure
  - Valence-Arousal values
  - "No face detected" when face leaves frame

Press 'q' to quit.
"""

import sys
import os
import cv2
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.mp_handler import MediaPipeHandler
from ai_modules.face_module.face_detection import FaceAgent


def main():
    print("[INFO] Starting face webcam test...")
    print("[INFO] Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Is another app using it?")
        return

    mp_agent  = MediaPipeHandler()
    agent     = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    print("[OK] FaceAgent started. Look at the camera!")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        mp_image     = mp_agent.prepare(bgr_frame)
        timestamp_ms = int(time.perf_counter() * 1000)

        face_results = mp_agent.process_face(mp_image, timestamp_ms)

        # Send to FaceAgent thread
        agent.enqueue_frame(face_results, bgr_frame)

        # Draw green dots on face landmarks
        if face_results.face_landmarks:
            h, w = bgr_frame.shape[:2]
            for face_lms in face_results.face_landmarks:
                for lm in face_lms:
                    cv2.circle(bgr_frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

        # Read and display latest result
        result = agent.latest_result
        y = 30

        if result and result.get("face_detected"):
            emotions = result["emotion_probabilities"]
            va       = result["valence_arousal"]
            metrics  = result["raw_metrics"]

            lines = [
                (f"Joy:         {emotions['joy']:.2f}",          (0, 220, 80)),
                (f"Frustration: {emotions['frustration']:.2f}",  (0, 80, 220)),
                (f"Fatigue:     {emotions['fatigue']:.2f}",      (0, 180, 220)),
                (f"Blink Rate:  {metrics['blink_rate']} bpm",    (200, 200, 0)),
                (f"Eye Closure: {metrics['eye_closure_index']:.2f}", (200, 200, 0)),
                (f"Valence:     {va['valence']:.2f}",            (180, 100, 255)),
                (f"Arousal:     {va['arousal']:.2f}",            (180, 100, 255)),
            ]

            # Semi-transparent background panel
            overlay = bgr_frame.copy()
            cv2.rectangle(overlay, (5, 5), (300, 5 + len(lines) * 28 + 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, bgr_frame, 0.55, 0, bgr_frame)

            for text, color in lines:
                cv2.putText(bgr_frame, text, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y += 28

        else:
            cv2.putText(bgr_frame, "No face detected", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)

            # Also print to terminal if result exists but face not detected
            if result:
                print(f"[DECAY] Joy: {result['emotion_probabilities']['joy']:.3f} "
                      f"Frust: {result['emotion_probabilities']['frustration']:.3f} "
                      f"Fatigue: {result['emotion_probabilities']['fatigue']:.3f}")

        # Print packet to terminal every second (when it updates)
        if result:
            emotions = result["emotion_probabilities"]
            metrics  = result["raw_metrics"]
            print(f"\r[LIVE] Joy={emotions['joy']:.2f} "
                  f"Frust={emotions['frustration']:.2f} "
                  f"Fatigue={emotions['fatigue']:.2f} "
                  f"Blink={metrics['blink_rate']}bpm "
                  f"EyeClosure={metrics['eye_closure_index']:.2f}  ",
                  end="", flush=True)

        cv2.imshow("FaceAgent Webcam Test", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\n[INFO] Shutting down...")
    agent.stop()
    cap.release()
    cv2.destroyAllWindows()
    mp_agent.close()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
