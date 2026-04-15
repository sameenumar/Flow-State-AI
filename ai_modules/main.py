"""
ai_modules/main.py

Entry point for the AI analysis pipeline.

Responsibilities:
  - Opens the webcam
  - Runs MediaPipe face + hand landmarking on a background thread
  - Feeds three agent threads: FaceAgent, gesture_agent, rPPG_agent
  - Runs FusionEngine + decide() at 1 Hz
  - Pushes a structured JSON payload to the backend over WebSocket

This script owns the camera. It never sends raw frames to the backend.
All heavy computation happens locally. The backend is only a relay.

DEBUG = True  → shows OpenCV window with overlays (development)
DEBUG = False → no window, no drawing overhead (production)
"""

import cv2
import time
import queue
import os
import sys
import json
import threading
import websocket
import base64

# ── Path setup ──────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# ── Local imports ────────────────────────────────────────────────────────────
from mp_handler import MediaPipeThread
from fusion.fusion   import FusionEngine
from fusion.decision import decide

FACE_AVAILABLE    = False
GESTURE_AVAILABLE = False
RPPG_AVAILABLE    = False

try:
    from face_module.face_detection import FaceAgent
    FACE_AVAILABLE = True
except ImportError:
    print("[WARNING] Face module not found — skipping emotion detection")

try:
    from gesture_module.gesture import gesture_agent
    GESTURE_AVAILABLE = True
except ImportError:
    print("[WARNING] Gesture module not found — skipping gesture detection")

try:
    from rPPG_module.rppg import rPPG_agent
    RPPG_AVAILABLE = True
except ImportError:
    print("[WARNING] rPPG module not found — skipping heart rate detection")


# ── Configuration ─────────────────────────────────────────────────────────────

DEBUG          = False                    # Set False in production
WS_URL         = "ws://localhost:8000/ws/ai"
FUSION_INTERVAL = 1.0                   # seconds between fusion + push
RPPG_TARGET_FPS = 30                    # capped by camera's actual FPS


# ── WebSocket sender ──────────────────────────────────────────────────────────

class WebSocketSender:
    def __init__(self, url: str):
        self.url  = url
        self.ws   = None
        self.lock = threading.Lock()
        self._reconnecting = False          # guard flag
        self._retry_after  = 0.0           # earliest time to retry
        self._backoff      = 1.0           # current backoff in seconds
        self._MAX_BACKOFF  = 30.0
        self._connect()

    def _connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.url)
            self._backoff = 1.0            # reset on success
            print(f"[WebSocket] Connected to {self.url}")
        except Exception as e:
            print(f"[WebSocket] Could not connect: {e} — retry in {self._backoff}s")
            self._retry_after = time.time() + self._backoff
            self._backoff = min(self._backoff * 2, self._MAX_BACKOFF)
            self.ws = None

    def send(self, payload: dict):
        def _do_send():
            with self.lock:
                # If we're in backoff, skip until retry window opens
                if self.ws is None:
                    if time.time() < self._retry_after:
                        return                  # still cooling down, drop this send
                    if self._reconnecting:
                        return                  # another thread is already reconnecting
                    self._reconnecting = True
                    try:
                        self._connect()
                    finally:
                        self._reconnecting = False

                if self.ws is None:
                    return                      # connect failed, give up this send

                try:
                    self.ws.send(json.dumps(payload))
                except Exception as e:
                    print(f"[WebSocket] Send failed: {e}")
                    self.ws = None
                    self._retry_after = time.time() + self._backoff
                    self._backoff = min(self._backoff * 2, self._MAX_BACKOFF)

        threading.Thread(target=_do_send, daemon=True).start()

    def close(self):
        with self.lock:
            if self.ws:
                try:
                    self.ws.close()
                except Exception:
                    pass
            self.ws = None

# ── Payload builder ───────────────────────────────────────────────────────────

def build_payload(fusion_result, decision_result, rppg_result,
                  face_result, gesture_result, pulse_sample, bgr_frame=None):
    """
    Assembles the complete JSON payload pushed to the backend every second.

    All sections have safe defaults so the payload is always a complete,
    well-typed object even during the 5-second rPPG warmup period or when
    individual agents haven't produced output yet.

    pulse_sample:
        A single float — the most recent value from the rPPG combined
        pulse waveform. The frontend accumulates one per push to draw
        a live scrolling heart rate waveform without receiving the full
        125-frame buffer every second.
    """

    # ── Fusion ────────────────────────────────────────────────────────────────
    fusion_block = {
        "cognitive_state":   "unknown",
        "emotional_state":   "unknown",
        "stress_level":      "unknown",
        "engagement":        "unknown",
        "focus_score":       0.0,
        "fusion_confidence": 0.0,
    }
    if fusion_result:
        fusion_block.update({k: fusion_result[k] for k in fusion_block if k in fusion_result})

    # ── Decision ──────────────────────────────────────────────────────────────
    decision_block = {
        "message":    "Initializing...",
        "suggestion": "Please remain visible to the camera.",
        "priority":   "passive",
        "alert":      False,
    }
    if decision_result:
        decision_block.update(decision_result)

    # ── Vitals (rPPG) ─────────────────────────────────────────────────────────
    vitals_block = {
        "bpm":            None,
        "hrv_sdnn":       None,
        "stress_index":   None,
        "signal_quality": "warming_up",
        "confidence":     0.0,
        "pulse_sample":   pulse_sample,
    }
    if rppg_result:
        vitals_block.update({
            "bpm":            rppg_result.get("bpm"),
            "hrv_sdnn":       rppg_result.get("hrv_sdnn"),
            "stress_index":   rppg_result.get("stress_index"),
            "signal_quality": rppg_result.get("signal_quality", "unknown"),
            "confidence":     rppg_result.get("confidence", 0.0),
            "pulse_sample":   pulse_sample,
        })

    # ── Face ──────────────────────────────────────────────────────────────────
    face_block = {
        "joy":               0.0,
        "frustration":       0.0,
        "fatigue":           0.0,
        "blink_rate":        0,
        "eye_closure_index": 0.0,
        "valence":           0.0,
        "arousal":           0.5,
        "face_detected":     False,
    }
    if face_result:
        emo = face_result.get("emotion_probabilities", {}) or {}
        va  = face_result.get("valence_arousal",       {}) or {}
        met = face_result.get("raw_metrics",           {}) or {}
        face_block.update({
            "joy":               emo.get("joy",               0.0),
            "frustration":       emo.get("frustration",       0.0),
            "fatigue":           emo.get("fatigue",           0.0),
            "blink_rate":        met.get("blink_rate",        0),
            "eye_closure_index": met.get("eye_closure_index", 0.0),
            "valence":           va.get("valence",            0.0),
            "arousal":           va.get("arousal",            0.5),
            "face_detected":     face_result.get("face_detected", False),
        })

    # ── Gesture ───────────────────────────────────────────────────────────────
    gesture_block = {
        "fidget_level":       0.0,
        "face_touching":      False,
        "typing_cadence":     None,
        "posture_slump":      0.0,
        "prob_fidgeting":     0.0,
        "prob_typing":        0.0,
        "prob_face_touching": 0.0,
        "hands_detected":     False,
    }
    if gesture_result:
        tags  = gesture_result.get("state_tags",    {}) or {}
        probs = gesture_result.get("probabilities", {}) or {}
        gesture_block.update({
            "fidget_level":       tags.get("fidget_level",        0.0),
            "face_touching":      tags.get("face_touching",       False),
            "typing_cadence":     tags.get("typing_cadence",      None),
            "posture_slump":      tags.get("posture_slump",       0.0),
            "prob_fidgeting":     probs.get("prob_fidgeting",     0.0),
            "prob_typing":        probs.get("prob_typing",        0.0),
            "prob_face_touching": probs.get("prob_face_touching", 0.0),
            "hands_detected":     gesture_result.get("hands_detected", False),
        })

    frame_b64 = None
    if bgr_frame is not None:
        try:
            # Encode to JPEG with compression (70% quality)
            ret, buffer = cv2.imencode('.jpg', bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[ERROR] Frame encoding failed: {e}")

    return {
        "timestamp": time.perf_counter(),
        "frame": frame_b64,
        "fusion":    fusion_block,
        "decision":  decision_block,
        "vitals":    vitals_block,
        "face":      face_block,
        "gesture":   gesture_block,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fps = actual_fps if actual_fps > 10 else 30.0
    rppg_fps   = min(actual_fps, RPPG_TARGET_FPS)
    rppg_interval = 1.0 / rppg_fps
    print(f"[Camera] Actual FPS: {actual_fps:.1f} | rPPG target: {rppg_fps}")

    # ── Threads ───────────────────────────────────────────────────────────────
    mp_results_queue = queue.Queue(maxsize=2)
    mp_thread        = MediaPipeThread(mp_results_queue)

    agent_face    = FaceAgent()             if FACE_AVAILABLE    else None
    agent_gesture = gesture_agent()         if GESTURE_AVAILABLE else None
    agent_rppg    = rPPG_agent(fps=int(rppg_fps)) if RPPG_AVAILABLE else None

    mp_thread.start()
    if agent_face:    agent_face.start()
    if agent_gesture: agent_gesture.start()
    if agent_rppg:    agent_rppg.start()

    # ── Fusion + WebSocket ────────────────────────────────────────────────────
    fusion_engine = FusionEngine()
    sender        = WebSocketSender(WS_URL)

    # ── Loop state ────────────────────────────────────────────────────────────
    face_results    = None
    hand_results    = None
    latest_bgr      = None
    fusion_result   = None
    decision_result = None
    last_rppg_time  = 0.0
    last_fusion_time= 0.0
    pulse_sample    = 0.0

    print("[OK] Pipeline running — press 'q' to quit")

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        # ── MediaPipe thread feed ──────────────────────────────────────────
        ts_ms = int(time.perf_counter() * 1000)
        mp_thread.enqueue_frame(bgr_frame, ts_ms)

        # ── Read MediaPipe results (non-blocking) ──────────────────────────
        try:
            latest_bgr, face_results, hand_results = mp_results_queue.get_nowait()
        except queue.Empty:
            pass

        # ── Agent feeds ───────────────────────────────────────────────────
        if face_results is not None and latest_bgr is not None:

            if agent_face:
                agent_face.enqueue_frame(face_results, latest_bgr)

            if agent_gesture and hand_results is not None:
                agent_gesture.enqueue_frame({
                    "hand_landmarks": hand_results.hand_landmarks or [],
                    "hand_gestures":  None,
                    "face_landmarks": (face_results.face_landmarks[0]
                                       if face_results.face_landmarks else None),
                    "frame_id": ts_ms,
                })

            # rPPG rate gate — consistent sample rate for FFT accuracy
            now = time.perf_counter()
            if agent_rppg and (now - last_rppg_time) >= rppg_interval:
                agent_rppg.enqueue_frame(latest_bgr, face_results.face_landmarks)
                last_rppg_time = now

        # ── Read agent results ─────────────────────────────────────────────
        face_result    = agent_face.latest_result    if agent_face    else None
        gesture_result = agent_gesture.latest_result if agent_gesture else None
        rppg_result    = agent_rppg.latest_result    if agent_rppg    else None

        # Single waveform point for live pulse chart on frontend.
        # rPPG_agent exposes latest_pulse_sample after each window computation.
        # See note at bottom of rPPG.py for the one-line addition needed.
        if agent_rppg:
            pulse_sample = getattr(agent_rppg, "latest_pulse_sample", 0.0) or 0.0

        # ── Fusion + push at 1 Hz ──────────────────────────────────────────
        now = time.perf_counter()
        if (now - last_fusion_time) >= FUSION_INTERVAL:
            fusion_result   = fusion_engine.fuse(face_result, gesture_result, rppg_result)
            decision_result = decide(fusion_result)
            last_fusion_time = now

            payload = build_payload(
                fusion_result, decision_result, rppg_result,
                face_result, gesture_result, pulse_sample,
                bgr_frame=latest_bgr  # <--- Pass the frame here
            )
            sender.send(payload)

        # ── Debug window ───────────────────────────────────────────────────
        if DEBUG:
            draw = bgr_frame.copy()
            h, w = draw.shape[:2]

            if face_results and face_results.face_landmarks:
                for lms in face_results.face_landmarks:
                    for lm in lms:
                        cv2.circle(draw, (int(lm.x*w), int(lm.y*h)), 1, (0,255,0), -1)

            if hand_results and hand_results.hand_landmarks:
                for lms in hand_results.hand_landmarks:
                    for lm in lms:
                        cv2.circle(draw, (int(lm.x*w), int(lm.y*h)), 1, (255,0,0), -1)

            y = 25
            def _t(line, col=(255,255,255)):
                nonlocal y
                cv2.putText(draw, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
                y += 22

            if fusion_result:
                _t(f"State: {fusion_result['cognitive_state']}  "
                   f"Emotion: {fusion_result['emotional_state']}  "
                   f"Stress: {fusion_result['stress_level']}", (0,255,255))
                _t(f"Engagement: {fusion_result['engagement']}  "
                   f"Focus: {fusion_result['focus_score']:.2f}  "
                   f"Conf: {fusion_result['fusion_confidence']:.2f}")
            else:
                _t("Fusion: warming up...", (160,160,160))

            if rppg_result:
                _t(f"BPM: {rppg_result.get('bpm','--')}  "
                   f"HRV: {rppg_result.get('hrv_sdnn','--')}ms  "
                   f"{rppg_result.get('signal_quality','')}", (0,220,0))
            else:
                _t("rPPG: warming up (5s)...", (160,160,160))

            if decision_result:
                _t(decision_result.get("message",""), (255,220,0))

            cv2.imshow("Flow AI — Debug", draw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── Shutdown ───────────────────────────────────────────────────────────
    print("[INFO] Shutting down...")
    mp_thread.stop()
    if agent_face:    agent_face.stop()
    if agent_gesture: agent_gesture.stop()
    if agent_rppg:    agent_rppg.stop()
    sender.close()
    cap.release()
    if DEBUG:
        cv2.destroyAllWindows()
    print("[OK] Shutdown complete.")

if __name__ == "__main__":
    main()