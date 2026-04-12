"""
Webcam test for gesture_agent (gesture.py).
Run from project root:
    python tests/test_gesture_webcam.py

What you'll see on screen:
  - Blue dots on hand landmarks
  - Live readout of fidget_level, face_touching, typing_cadence, posture_slump
  - Probabilities for fidgeting, typing, face touching
  - Movement intensity bar

Things to try:
  - Wiggle fingers while keeping wrist still  → fidget_level goes up
  - Touch your face with your hand            → face_touching = True
  - Tap fingers on desk like typing           → typing_cadence appears
  - Keep hands completely still              → posture_slump goes up

Press 'q' to quit.
"""

import sys
import os
import cv2
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.mp_handler import MediaPipeHandler
from ai_modules.gesture_module.gesture import gesture_agent


def draw_bar(frame, x, y, width, value, color, label):
    """Draw a horizontal progress bar with label."""
    bar_h = 14
    filled = int(width * min(max(value, 0.0), 1.0))
    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + bar_h), (120, 120, 120), 1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x + width + 8, y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def main():
    print("[INFO] Starting gesture webcam test...")
    print("[INFO] Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    mp_agent = MediaPipeHandler()
    agent    = gesture_agent(output_frequency=1.0)
    agent.start()

    print("[OK] gesture_agent started. Show your hands to the camera!")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        mp_image     = mp_agent.prepare(bgr_frame)
        timestamp_ms = int(time.perf_counter() * 1000)

        face_results = mp_agent.process_face(mp_image, timestamp_ms)
        hand_results = mp_agent.process_hands(mp_image, timestamp_ms)

        # Draw blue dots on hand landmarks
        if hand_results.hand_landmarks:
            h, w = bgr_frame.shape[:2]
            for hand_lms in hand_results.hand_landmarks:
                for lm in hand_lms:
                    cv2.circle(bgr_frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 100, 0), -1)

        # Draw green dots on face landmarks (small, not distracting)
        if face_results.face_landmarks:
            h, w = bgr_frame.shape[:2]
            for face_lms in face_results.face_landmarks:
                for lm in face_lms:
                    cv2.circle(bgr_frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 200, 0), -1)

        # Build gesture frame data and enqueue
        gesture_frame = {
            "hand_landmarks": hand_results.hand_landmarks if hand_results.hand_landmarks else [],
            "face_landmarks": face_results.face_landmarks[0] if face_results.face_landmarks else None,
            "hand_gestures":  None,
            "frame_id":       timestamp_ms,
        }
        agent.enqueue_frame(gesture_frame)

        # DEBUG - add these lines temporarily
        raw_result = agent.latest_result
        if raw_result:
            rm = raw_result.get("raw_metrics", {})
            print(f"\r[DEBUG] move={rm.get('movement_intensity',0):.4f} "
                  f"fidget={rm.get('fidget_score',0):.4f} "
                  f"typing={rm.get('typing_score',0):.4f} "
                  f"hands={raw_result.get('hands_detected')}", end="", flush=True)
        else:
            print("\r[DEBUG] No packet yet", end="", flush=True)

        # Semi-transparent background panel
        overlay = bgr_frame.copy()
        cv2.rectangle(overlay, (5, 5), (380, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, bgr_frame, 0.55, 0, bgr_frame)

        result = agent.latest_result
        y = 28

        if result and result.get("hands_detected"):
            state  = result["state_tags"]
            probs  = result["probabilities"]
            raw    = result["raw_metrics"]

            # ── State tags ──
            face_touch_color = (0, 80, 255) if state["face_touching"] else (80, 200, 80)
            face_touch_text  = "YES" if state["face_touching"] else "no"
            cadence = state["typing_cadence"] if state["typing_cadence"] else "none"

            cv2.putText(bgr_frame, f"Face touching: {face_touch_text}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, face_touch_color, 2)
            y += 26

            cv2.putText(bgr_frame, f"Typing cadence: {cadence}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 180, 0), 2)
            y += 30

            # ── Probability bars ──
            bar_x = 12
            bar_w = 180

            draw_bar(bgr_frame, bar_x, y, bar_w,
                     state["fidget_level"], (0, 180, 255), "Fidget")
            y += 26

            draw_bar(bgr_frame, bar_x, y, bar_w,
                     probs["prob_typing"], (0, 255, 180), "Typing")
            y += 26

            draw_bar(bgr_frame, bar_x, y, bar_w,
                     probs["prob_face_touching"], (80, 80, 255), "FaceTouch")
            y += 26

            draw_bar(bgr_frame, bar_x, y, bar_w,
                     raw["movement_intensity"], (200, 200, 80), "Movement")
            y += 26

            draw_bar(bgr_frame, bar_x, y, bar_w,
                     state["posture_slump"], (180, 80, 180), "Slump")
            y += 30

            # ── Confidence ──
            conf = result.get("overall_confidence", 0)
            cv2.putText(bgr_frame, f"Confidence: {conf:.2f}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            # ── Print to terminal ──
            print(f"\r[LIVE] Fidget={state['fidget_level']:.2f} "
                  f"FaceTouch={state['face_touching']} "
                  f"Typing={cadence} "
                  f"Slump={state['posture_slump']:.2f} "
                  f"Move={raw['movement_intensity']:.2f}  ",
                  end="", flush=True)

        elif result and not result.get("hands_detected"):
            cv2.putText(bgr_frame, "No hands detected",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
            y += 26
            cv2.putText(bgr_frame, "Show your hands to the camera",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        else:
            cv2.putText(bgr_frame, "Initialising... (1 second warmup)",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 80), 1)

        cv2.imshow("gesture_agent Webcam Test", bgr_frame)

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
