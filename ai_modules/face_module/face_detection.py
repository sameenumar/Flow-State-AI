"""
Face detection module.
Detects facial expressions, emotions, blink rate and eye closure.
Outputs standardized JSON packet at 1 Hz.

Required output format:
{
    "joy": 0.1,
    "frustration": 0.6,
    "fatigue": 0.3,
    "blink_rate": 12,
    "eye_closure_index": 0.2
}
Plus valence-arousal, face_detected flag, module metadata.
"""

import cv2
import numpy as np
import queue
import threading
from collections import deque
from typing import Optional, Dict, Any
import time


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


class FaceAgent(threading.Thread):
    """
    Face emotion and activity agent for real-time facial analysis.
    Runs as a daemon thread. Call start() after init.
    Reads latest_result to get the most recent 1 Hz output packet.
    """

    def __init__(self, blink_window: int = 60, fps: int = 30, output_frequency: float = 1.0):
        """
        Args:
            blink_window: Number of frames to keep for blink detection window
            fps: Expected frames per second (used for blink rate calculation)
            output_frequency: How often to emit an output packet (Hz). Default = 1 per second.
        """
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)
        self.daemon = True

        self.fps = fps
        self.blink_threshold = 0.3
        self.blink_window = blink_window

        # Output frequency control
        self.output_frequency = output_frequency
        self.output_interval = 1.0 / output_frequency
        self.last_output_time = 0.0

        # Blink tracking
        self.eye_closure_history = deque(maxlen=blink_window)
        self._blink_times: list = []          # timestamps of blink events
        self.was_closed = False
        self.frame_count = 0

        # Emotion smoothing (rolling window over ~1 second of frames)
        self.joy_history = deque(maxlen=30)
        self.frustration_history = deque(maxlen=30)
        self.fatigue_history = deque(maxlen=30)

        # Public output — read this from main loop
        self.latest_result: Optional[Dict[str, Any]] = None

        # Stop signal for clean shutdown
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def enqueue_frame(self, results, frame: np.ndarray) -> None:
        """
        Send a new frame + MediaPipe results to the processing thread.
        Drops the oldest frame if queue is full (keeps latency low).

        Args:
            results: MediaPipe FaceLandmarkerResult
            frame: BGR numpy frame from OpenCV
        """
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put((results, frame))

    def stop(self) -> None:
        """Signal the thread to exit cleanly."""
        self._stop_event.set()
        # Unblock the queue.get() with a sentinel
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

    # ──────────────────────────────────────────────────────────────
    # THREAD MAIN LOOP
    # ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Daemon thread loop. Pulls frames from queue and processes them.
        Exits cleanly when stop() is called.
        """
        while not self._stop_event.is_set():
            try:
                item = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Sentinel value means stop
            if item is None:
                break

            results, frame = item
            current_time = time.time()
            self._process_frame(results, frame, current_time)

    # ──────────────────────────────────────────────────────────────
    # CORE PROCESSING
    # ──────────────────────────────────────────────────────────────

    def _process_frame(self, results, frame: np.ndarray, current_time: float) -> None:
        """
        Process facial landmarks and blendshapes to compute emotional state metrics.
        Updates self.latest_result at the configured output_frequency.

        Args:
            results: MediaPipe FaceLandmarkerResult (may be None if detection failed)
            frame: BGR numpy frame
            current_time: Current timestamp from time.time()
        """
        # ── No face detected: decay toward neutral and emit if interval elapsed ──
        if results is None or not results.face_landmarks:
            if (current_time - self.last_output_time) >= self.output_interval:
                self.latest_result = self._default_result(current_time)
                self.last_output_time = current_time
            return

        try:
            # ── Extract blendshapes (MediaPipe's trained facial action units) ──
            blendshapes = self._extract_blendshapes(results)

            # ── Eye closure (average of left + right eye blink blendshape scores) ──
            eye_blink_left  = blendshapes.get("eyeBlinkLeft", 0.0)
            eye_blink_right = blendshapes.get("eyeBlinkRight", 0.0)
            eye_closure_index = (eye_blink_left + eye_blink_right) / 2.0

            self.eye_closure_history.append(eye_closure_index)
            self._update_blink_count(eye_closure_index, current_time)
            blink_rate = self._get_blink_rate()

            # ── Joy indicators: mouth smile blendshapes ──
            smile_left  = blendshapes.get("mouthSmileLeft", 0.0)
            smile_right = blendshapes.get("mouthSmileRight", 0.0)
            smile_intensity = (smile_left + smile_right) / 2.0

            # ── Frustration indicators: frown + brow furrow ──
            mouth_frown_left  = blendshapes.get("mouthFrownLeft", 0.0)
            mouth_frown_right = blendshapes.get("mouthFrownRight", 0.0)
            frown_intensity   = (mouth_frown_left + mouth_frown_right) / 2.0

            brow_down_left  = blendshapes.get("browDownLeft", 0.0)
            brow_down_right = blendshapes.get("browDownRight", 0.0)
            brow_down = (brow_down_left + brow_down_right) / 2.0

            # ── Fatigue indicators: squinting + high blink rate + droopy eyes ──
            eye_squint_left  = blendshapes.get("eyeSquintLeft", 0.0)
            eye_squint_right = blendshapes.get("eyeSquintRight", 0.0)
            eye_squint = (eye_squint_left + eye_squint_right) / 2.0

            # ── Compute per-frame emotion scores ──
            # Joy:         high smile, low frown, relaxed brows
            joy = smile_intensity * 0.7 + (1.0 - frown_intensity) * 0.2 + (1.0 - brow_down) * 0.1

            # Frustration: high frown, furrowed brows, low smile
            frustration = frown_intensity * 0.4 + brow_down * 0.4 + (1.0 - smile_intensity) * 0.2

            # Fatigue:     high eye closure, squinting, elevated blink rate
            # Blink rate > 25 bpm starts contributing to fatigue
            fatigue = eye_closure_index * 0.5 + eye_squint * 0.3 + min(blink_rate / 25.0, 1.0) * 0.2

            joy         = _clamp(joy)
            frustration = _clamp(frustration)
            fatigue     = _clamp(fatigue)

            # ── Accumulate into smoothing windows ──
            self.joy_history.append(joy)
            self.frustration_history.append(frustration)
            self.fatigue_history.append(fatigue)

            # ── Emit standardized packet at configured frequency ──
            if (current_time - self.last_output_time) >= self.output_interval:
                smoothed_joy         = float(np.mean(self.joy_history))
                smoothed_frustration = float(np.mean(self.frustration_history))
                smoothed_fatigue     = float(np.mean(self.fatigue_history))
                avg_eye_closure      = float(np.mean(self.eye_closure_history)) \
                                       if self.eye_closure_history else 0.0

                raw = {
                    "joy":               smoothed_joy,
                    "frustration":       smoothed_frustration,
                    "fatigue":           smoothed_fatigue,
                    "blink_rate":        blink_rate,
                    "eye_closure_index": avg_eye_closure,
                }
                self.latest_result = self._build_packet(current_time, raw)
                self.last_output_time = current_time

        except Exception as e:
            print(f"[FaceAgent] Error processing frame: {e}")
            if (current_time - self.last_output_time) >= self.output_interval:
                self.latest_result = self._default_result(current_time)
                self.last_output_time = current_time

    # ──────────────────────────────────────────────────────────────
    # PACKET BUILDERS
    # ──────────────────────────────────────────────────────────────

    def _build_packet(self, current_time: float, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the standardized output JSON packet.

        Valence-Arousal mapping:
          Valence  = joy - frustration           → range -1 to +1
          Arousal  = frustration*0.6 + joy*0.4   → high energy emotions raise arousal
                     fatigue pulls it down via (1-fatigue)*0.3

        Required top-level output (from spec):
            { "joy": ..., "frustration": ..., "fatigue": ...,
              "blink_rate": ..., "eye_closure_index": ... }
        """
        valence = float(np.clip(raw["joy"] - raw["frustration"], -1.0, 1.0))

        # Arousal: frustration and joy both increase arousal; fatigue suppresses it
        arousal = (raw["frustration"] * 0.6 + raw["joy"] * 0.4) + (1.0 - raw["fatigue"]) * 0.3
        arousal = float(np.clip(arousal, 0.0, 1.0))

        return {
            # ── Module metadata ──
            "module":    "face",
            "timestamp": current_time,

            # ── Valence-Arousal (psychology standard) ──
            "valence_arousal": {
                "valence": valence,
                "arousal": arousal,
            },

            # ── Emotion probabilities (REQUIRED by spec) ──
            "emotion_probabilities": {
                "joy":         raw["joy"],
                "frustration": raw["frustration"],
                "fatigue":     raw["fatigue"],
            },

            # ── Raw metrics (REQUIRED by spec) ──
            "raw_metrics": {
                "blink_rate":        raw["blink_rate"],
                "eye_closure_index": raw["eye_closure_index"],
            },

            "face_detected": True,
        }

    def _default_result(self, current_time: float) -> Dict[str, Any]:
        """
        Return a decayed / neutral result when no face is detected.
        Smoothly decays previous values toward zero so the fusion layer
        doesn't see a hard jump.
        """
        if self.latest_result and self.latest_result.get("face_detected"):
            emotions = self.latest_result.get("emotion_probabilities", {})
            metrics  = self.latest_result.get("raw_metrics", {})

            joy         = emotions.get("joy", 0.0)         * 0.9
            frustration = emotions.get("frustration", 0.0) * 0.9
            fatigue     = emotions.get("fatigue", 0.0)     * 0.9
            blink_rate  = max(0, metrics.get("blink_rate", 0) - 1)
            eye_closure = metrics.get("eye_closure_index", 0.0) * 0.9

            raw = {
                "joy":               joy,
                "frustration":       frustration,
                "fatigue":           fatigue,
                "blink_rate":        blink_rate,
                "eye_closure_index": eye_closure,
            }
            packet = self._build_packet(current_time, raw)
            packet["face_detected"] = False
            return packet

        # Truly cold-start neutral values
        return {
            "module":    "face",
            "timestamp": current_time,
            "valence_arousal": {
                "valence": 0.0,
                "arousal": 0.5,
            },
            "emotion_probabilities": {
                "joy":         0.0,
                "frustration": 0.0,
                "fatigue":     0.0,
            },
            "raw_metrics": {
                "blink_rate":        0,
                "eye_closure_index": 0.0,
            },
            "face_detected": False,
        }

    # ──────────────────────────────────────────────────────────────
    # BLENDSHAPE EXTRACTION
    # ──────────────────────────────────────────────────────────────

    def _extract_blendshapes(self, results) -> Dict[str, float]:
        """
        Extract blendshape scores from MediaPipe FaceLandmarkerResult.
        Blendshapes are pre-trained 52 facial action unit scores (0.0 – 1.0).

        Args:
            results: MediaPipe FaceLandmarkerResult with face_blendshapes enabled

        Returns:
            Dict mapping blendshape category_name → score float
        """
        blendshapes_dict: Dict[str, float] = {}
        try:
            if hasattr(results, "face_blendshapes") and results.face_blendshapes:
                # [0] = primary face; each element has .category_name and .score
                for bs in results.face_blendshapes[0]:
                    blendshapes_dict[bs.category_name] = bs.score
        except (AttributeError, IndexError, TypeError) as e:
            print(f"[FaceAgent] Blendshapes unavailable: {e}")
        return blendshapes_dict

    # ──────────────────────────────────────────────────────────────
    # BLINK DETECTION
    # ──────────────────────────────────────────────────────────────

    def _update_blink_count(self, eye_closure_index: float, current_time: float) -> None:
        """
        Detect a blink by watching for a closed→open transition (rising edge).
        Records real timestamps so blink rate is time-accurate.

        Args:
            eye_closure_index: Average eye blink blendshape score (0-1)
            current_time: Current timestamp from time.time()
        """
        is_closed = eye_closure_index > self.blink_threshold

        # Rising edge = eye just closed
        if is_closed and not self.was_closed:
            self._blink_times.append(current_time)

        self.was_closed = is_closed
        self.frame_count += 1

    def _get_blink_rate(self) -> int:
        """
        Calculate blinks per minute using a rolling 3-second window.
        Also prunes old blink timestamps to prevent unbounded memory growth.

        Returns:
            Integer blinks-per-minute, capped at 60.
        """
        window_seconds = 3.0
        now = time.time()
        cutoff = now - window_seconds

        # FIX: Prune old timestamps so the list never grows unboundedly
        self._blink_times = [t for t in self._blink_times if t >= cutoff]

        blinks_in_window = len(self._blink_times)
        blink_rate = int((blinks_in_window / window_seconds) * 60)
        return max(0, min(blink_rate, 60))

    # ──────────────────────────────────────────────────────────────
    # OPTIONAL OVERLAY (for debug display)
    # ──────────────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Draw emotion metrics as text overlay onto a BGR frame (in-place).

        Args:
            frame: OpenCV BGR image

        Returns:
            Same frame with text overlay added
        """
        if self.latest_result is None or frame is None:
            return frame

        try:
            emotions = self.latest_result.get("emotion_probabilities", {})
            va       = self.latest_result.get("valence_arousal", {})
            metrics  = self.latest_result.get("raw_metrics", {})

            lines = [
                f"Joy: {emotions.get('joy', 0):.2f} | Frust: {emotions.get('frustration', 0):.2f} | Fatigue: {emotions.get('fatigue', 0):.2f}",
                f"Valence: {va.get('valence', 0):.2f} | Arousal: {va.get('arousal', 0):.2f}",
                f"Blink Rate: {metrics.get('blink_rate', 0)} bpm | Eye Closure: {metrics.get('eye_closure_index', 0):.2f}",
            ]

            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception:
            pass

        return frame
