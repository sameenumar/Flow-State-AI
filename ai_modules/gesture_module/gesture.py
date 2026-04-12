"""
Gesture recognition module.
Detects and classifies hand gestures and body movements.
Outputs standardized JSON packet at 1 Hz.

Required output format (from spec):
{
    "fidget_level": 0.8,
    "face_touching": true,
    "typing_cadence": "erratic",
    "posture_slump": 0.4
}
Plus probabilities, raw_metrics, module metadata.
"""

import queue
import threading
import numpy as np
import time
from collections import deque
from typing import Dict, Optional, Any


class gesture_agent(threading.Thread):
    """
    Gesture recognition agent. Runs as a daemon thread.
    Call start() after init. Read latest_result for the 1 Hz output packet.
    """

    def __init__(self, smoothing_window: int = 5, output_frequency: float = 1.0):
        """
        Args:
            smoothing_window: Unused legacy param (kept for API compatibility)
            output_frequency: How often to emit a packet in Hz. Default = 1 per second.
        """
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)
        self.daemon = True

        # Public output — read this from main loop
        self.latest_result: Optional[Dict[str, Any]] = None

        # Output frequency control
        self.output_frequency = output_frequency
        self.output_interval  = 1.0 / output_frequency
        self.last_output_time = 0.0

        self.smoothing_window = smoothing_window

        # ── Frame accumulation (rolling 1-second window at ~30fps) ──
        self.frames_buffer      = deque(maxlen=30)
        self.frame_timestamps   = deque(maxlen=30)

        # ── Per-frame metric windows ──
        self.movement_intensity_window = deque(maxlen=30)
        self.fidget_score_window       = deque(maxlen=30)
        self.typing_score_window       = deque(maxlen=30)
        self.face_touching_window      = deque(maxlen=30)
        self.confidence_window         = deque(maxlen=30)

        # FIX: Separate flag to accurately track whether hands were seen
        self._hands_seen_in_window = False

        # ── State tracking ──
        self.previous_hand_positions = None

        # Tap detection
        self.tap_timestamps = deque(maxlen=10)

        # ── Detection thresholds (tuned for MediaPipe normalized coords 0.0–1.0) ──
        # Per-frame movement in normalized coords is tiny: typical finger tap = 0.002–0.008
        self.WRIST_FIDGET_THRESHOLD  = 0.003   # wrist must be THIS still for fidget to trigger
        self.FACE_TOUCHING_THRESHOLD = 0.15    # fingertip within 15% of frame width from face
        self.TYPING_FINGER_THRESHOLD = 0.003   # minimum vertical movement per frame to count as tap
        self.MIN_HAND_CONFIDENCE     = 0.7

        # State classification thresholds (what prob_x must exceed to trigger state tag)
        self.FIDGET_THRESHOLD = 0.2
        self.TYPING_THRESHOLD = 0.2
        self.FACE_THRESHOLD   = 0.2

        # Stop signal
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def enqueue_frame(self, frame: Dict) -> None:
        """
        Add a frame dict to the processing queue.
        Drops oldest frame if queue is full (keeps latency low).

        Args:
            frame: Dict with keys:
                - hand_landmarks: list of MediaPipe NormalizedLandmark lists
                - face_landmarks: MediaPipe NormalizedLandmark list or None
                - hand_gestures: (unused, pass None)
                - frame_id: int
        """
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def stop(self) -> None:
        """Signal the thread to exit cleanly."""
        self._stop_event.set()
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass

    # ──────────────────────────────────────────────────────────────
    # THREAD MAIN LOOP
    # ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Daemon thread loop. Pulls frames, computes per-frame metrics,
        and emits a standardized packet once per output_interval.
        """
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                break

            current_time = time.time()

            # Compute per-frame metrics
            frame_metrics = self.recognize_gesture(frame)

            # Accumulate into rolling windows
            self.frames_buffer.append(frame)
            self.frame_timestamps.append(current_time)
            self.movement_intensity_window.append(frame_metrics.get("movement_intensity", 0.0))
            self.fidget_score_window.append(frame_metrics.get("fidget_score", 0.0))
            self.typing_score_window.append(frame_metrics.get("typing_score", 0.0))
            self.face_touching_window.append(frame_metrics.get("face_touching_score", 0.0))
            self.confidence_window.append(frame_metrics.get("confidence", 0.0))

            # FIX: Track whether any frame in this window actually had hands
            if frame_metrics.get("hand_detected", False):
                self._hands_seen_in_window = True

            # Emit standardized packet at configured frequency
            if (current_time - self.last_output_time) >= self.output_interval:
                packet = self._generate_standardized_packet(current_time)
                if packet is not None:
                    self.latest_result = packet
                self.last_output_time = current_time
                # Reset per-window hand flag after emitting
                self._hands_seen_in_window = False

    # ──────────────────────────────────────────────────────────────
    # PER-FRAME GESTURE RECOGNITION
    # ──────────────────────────────────────────────────────────────

    def recognize_gesture(self, frame_data: Dict) -> Dict:
        """
        Compute gesture metrics for a single frame.

        Args:
            frame_data: Dict with hand_landmarks, face_landmarks, etc.

        Returns:
            Dict with movement_intensity, fidget_score, typing_score,
            face_touching_score, confidence, hand_detected.
        """
        frame_result = {
            "movement_intensity":  0.0,
            "fidget_score":        0.0,
            "typing_score":        0.0,
            "face_touching_score": 0.0,
            "confidence":          0.0,
            "hand_detected":       False,
        }

        hand_landmarks = frame_data.get("hand_landmarks", None)
        face_landmarks = frame_data.get("face_landmarks", None)

        if not hand_landmarks or len(hand_landmarks) == 0:
            self._reset_hand_state()
            return frame_result

        # Movement intensity + current positions for velocity calc
        movement_intensity, hand_positions = self._calculate_movement_intensity(hand_landmarks)
        frame_result["movement_intensity"] = movement_intensity

        # Fidget: fingers wiggling while wrist is relatively still
        fidget_score = self._calculate_fidget_score(hand_landmarks, hand_positions)
        frame_result["fidget_score"] = fidget_score

        # Typing: repetitive vertical finger tapping pattern
        typing_score = self._calculate_typing_score(hand_landmarks, movement_intensity)
        frame_result["typing_score"] = typing_score

        # Face touching: fingertips close to detected face center
        face_touching_score = 0.0
        if face_landmarks is not None:
            face_touching_score = self._calculate_face_touching_score(hand_landmarks, face_landmarks)
        frame_result["face_touching_score"] = face_touching_score

        # Overall detection confidence
        confidence = self._calculate_confidence(hand_landmarks)
        frame_result["confidence"]    = confidence
        frame_result["hand_detected"] = True

        # Update previous positions AFTER all calculations so fidget/typing
        # can compare current frame vs the actual previous frame
        if hand_positions:
            self.previous_hand_positions = hand_positions

        return frame_result

    # ──────────────────────────────────────────────────────────────
    # PACKET BUILDER
    # ──────────────────────────────────────────────────────────────

    def _generate_standardized_packet(self, current_time: float) -> Optional[Dict]:
        """
        Aggregate the 1-second rolling window into a standardized output packet.

        Output format matches spec:
            state_tags.fidget_level        (float 0-1)
            state_tags.face_touching       (bool)
            state_tags.typing_cadence      (str or None)
            state_tags.posture_slump       (float 0-1)
            probabilities.prob_fidgeting   (float 0-1)
            probabilities.prob_typing      (float 0-1)
            probabilities.prob_face_touching (float 0-1)

        Returns:
            Standardized dict or None if window is empty.
        """
        if not self.movement_intensity_window:
            return None

        avg_movement     = float(np.mean(self.movement_intensity_window))
        avg_fidget       = float(np.mean(self.fidget_score_window))       if self.fidget_score_window       else 0.0
        avg_typing       = float(np.mean(self.typing_score_window))       if self.typing_score_window       else 0.0
        avg_face_touch   = float(np.mean(self.face_touching_window))      if self.face_touching_window      else 0.0
        avg_confidence   = float(np.mean(self.confidence_window))         if self.confidence_window         else 0.0

        # Probabilities: clip to [0, 1]
        prob_fidgeting    = float(np.clip(avg_fidget,     0.0, 1.0))
        prob_typing       = float(np.clip(avg_typing,     0.0, 1.0))
        prob_face_touch   = float(np.clip(avg_face_touch, 0.0, 1.0))

        # ── State tags (REQUIRED by spec) ──

        # fidget_level: 0-1 score (spec asks for float like 0.8)
        fidget_level = prob_fidgeting

        # face_touching: bool
        face_touching_bool = bool(prob_face_touch > self.FACE_THRESHOLD)

        # typing_cadence: None | "slow" | "fast" | "erratic"
        typing_cadence: Optional[str] = None
        if prob_typing > self.TYPING_THRESHOLD:
            if avg_movement > 0.6:
                typing_cadence = "erratic"
            elif avg_movement > 0.15:
                typing_cadence = "fast"
            else:
                typing_cadence = "slow"

        # posture_slump: low movement for extended time approximates slouching
        posture_slump = float(max(0.0, 1.0 - avg_movement * 3.0)) if avg_movement < 0.3 else 0.0

        # FIX: hands_detected is True only if at least one frame in the window had real hands
        hands_detected = self._hands_seen_in_window

        return {
            # Metadata
            "module":          "gesture",
            "timestamp":       current_time,
            "window_duration": self.output_interval,
            "frame_count":     len(self.frames_buffer),

            # ── State tags (REQUIRED by spec) ──
            "state_tags": {
                "fidget_level":   fidget_level,
                "face_touching":  face_touching_bool,
                "typing_cadence": typing_cadence,
                "posture_slump":  posture_slump,
            },

            # ── Raw metrics ──
            "raw_metrics": {
                "movement_intensity":  avg_movement,
                "fidget_score":        avg_fidget,
                "typing_score":        avg_typing,
                "face_proximity_score": avg_face_touch,
            },

            # ── Probabilities (0.0 – 1.0) ──
            "probabilities": {
                "prob_fidgeting":     prob_fidgeting,
                "prob_typing":        prob_typing,
                "prob_face_touching": prob_face_touch,
            },

            "overall_confidence": avg_confidence,
            "hands_detected":     hands_detected,
        }

    # ──────────────────────────────────────────────────────────────
    # METRIC CALCULATIONS
    # ──────────────────────────────────────────────────────────────

    def _calculate_movement_intensity(self, hand_landmarks):
        """
        Calculate overall hand movement intensity (0.0 – 1.0) from
        frame-to-frame velocity of wrists and fingertips.

        Returns:
            (movement_intensity float, hand_positions list)
        """
        try:
            hand_positions = []

            for hand_marks in hand_landmarks:
                if hand_marks is None or len(hand_marks) < 21:
                    continue

                # Convert MediaPipe NormalizedLandmark to (x, y) numpy arrays
                coords = []
                for mark in hand_marks:
                    if hasattr(mark, "x") and hasattr(mark, "y"):
                        coords.append([mark.x, mark.y])
                    elif hasattr(mark, "__len__"):
                        coords.append(mark[:2])
                    else:
                        coords.append([0.0, 0.0])

                if len(coords) < 21:
                    continue

                all_points = np.array(coords)  # shape (21, 2)
                wrist      = all_points[0]
                fingertips = [all_points[i] for i in [4, 8, 12, 16, 20]]

                hand_positions.append({
                    "wrist":      wrist,
                    "fingertips": fingertips,
                    "all_points": all_points,
                })

            if not hand_positions:
                self.previous_hand_positions = None
                return 0.0, hand_positions

            movement_intensity = 0.0

            if (self.previous_hand_positions is not None
                    and len(self.previous_hand_positions) == len(hand_positions)):
                total_distance = 0.0
                point_count    = 0

                for curr, prev in zip(hand_positions, self.previous_hand_positions):
                    # Wrist velocity (weighted 2x — large wrist movement = high activity)
                    wrist_dist = np.linalg.norm(curr["wrist"] - prev["wrist"])
                    total_distance += wrist_dist * 2.0
                    point_count    += 1

                    # Fingertip velocities
                    for cf, pf in zip(curr["fingertips"], prev["fingertips"]):
                        total_distance += np.linalg.norm(cf - pf)
                        point_count    += 1

                avg_dist           = total_distance / point_count if point_count > 0 else 0.0
                # In normalized coords, avg per-frame movement of 0.005 = moderate, 0.015+ = fast
                movement_intensity = min(avg_dist / 0.012, 1.0)

            return movement_intensity, hand_positions

        except Exception as e:
            print(f"[GestureAgent] Error in movement intensity: {e}")
            return 0.0, []

    def _calculate_fidget_score(self, hand_landmarks, hand_positions) -> float:
        """
        Fidget = fingers moving rapidly while wrist is relatively still.
        Score range: 0.0 (no fidget) – 1.0 (strong fidget).
        """
        try:
            if not hand_positions or self.previous_hand_positions is None:
                return 0.0
            if len(hand_positions) != len(self.previous_hand_positions):
                return 0.0

            finger_movements = 0.0
            wrist_movement   = 0.0

            for i, hand_pos in enumerate(hand_positions):
                if i >= len(self.previous_hand_positions):
                    continue
                prev = self.previous_hand_positions[i]

                wrist_dist     = np.linalg.norm(hand_pos["wrist"] - prev["wrist"])
                wrist_movement = max(wrist_movement, wrist_dist)

                for cf, pf in zip(hand_pos["fingertips"], prev["fingertips"]):
                    finger_movements += np.linalg.norm(cf - pf)

            avg_finger = finger_movements / (len(hand_positions) * 5) if hand_positions else 0.0

            if wrist_movement < self.WRIST_FIDGET_THRESHOLD:
                # Fingers active, wrist still = fidgeting
                # 0.004 avg finger movement per frame = clear fidget
                fidget_score = min(avg_finger / 0.004, 1.0)
            else:
                # Wrist moving = gesturing/typing, not pure fidget
                fidget_score = min(avg_finger / 0.008, 0.5)

            return float(np.clip(fidget_score, 0.0, 1.0))

        except Exception as e:
            print(f"[GestureAgent] Error in fidget score: {e}")
            return 0.0

    def _calculate_typing_score(self, hand_landmarks, movement_intensity: float) -> float:
        """
        Typing = repetitive vertical (Y-axis) finger tapping with moderate movement.
        Score range: 0.0 – 1.0.
        """
        try:
            if not hand_landmarks or movement_intensity < 0.01:
                return 0.0

            vertical_movements = []

            for hand_idx, hand_marks in enumerate(hand_landmarks):
                if len(hand_marks) < 21:
                    continue

                coords = []
                for mark in hand_marks:
                    if hasattr(mark, "x") and hasattr(mark, "y"):
                        coords.append([mark.x, mark.y])
                    elif hasattr(mark, "__len__"):
                        coords.append(mark[:2])
                    else:
                        coords.append([0.0, 0.0])

                if len(coords) < 21:
                    continue

                if (self.previous_hand_positions is not None
                        and hand_idx < len(self.previous_hand_positions)):
                    prev_points = self.previous_hand_positions[hand_idx]["all_points"]

                    if len(prev_points) >= 21:
                        # Index (8) and middle (12) fingers are primary typing digits
                        index_v  = abs(coords[8][1]  - prev_points[8,  1])
                        middle_v = abs(coords[12][1] - prev_points[12, 1])

                        vertical_movements.extend([index_v, middle_v])

                        if (index_v > self.TYPING_FINGER_THRESHOLD
                                or middle_v > self.TYPING_FINGER_THRESHOLD):
                            self.tap_timestamps.append(time.time())

            if not vertical_movements:
                return 0.0

            avg_vertical = float(np.mean(vertical_movements))

            # Count taps in last 2 seconds for rhythm detection
            now         = time.time()
            recent_taps = sum(1 for t in self.tap_timestamps if now - t < 2.0)

            # 0.003 vertical movement per frame is a clear tap
            base_score    = min(avg_vertical / self.TYPING_FINGER_THRESHOLD, 1.0)
            rhythm_bonus  = min(recent_taps / 10.0, 0.5)
            typing_score  = base_score * 0.6 + rhythm_bonus

            # Modulate by movement intensity:
            # Very erratic movement (>0.6) reduces typing confidence
            # Moderate movement (0.25–0.6) matches typical typing pace
            if movement_intensity > 0.6:
                typing_score *= 0.7
            elif movement_intensity > 0.1:
                typing_score *= 1.2

            return float(np.clip(typing_score, 0.0, 1.0))

        except Exception as e:
            print(f"[GestureAgent] Error in typing score: {e}")
            return 0.0

    def _calculate_face_touching_score(self, hand_landmarks, face_landmarks) -> float:
        """
        Measure how close any fingertip is to the face center.
        Score = 1.0 when a fingertip is at/near face center; 0.0 when far away.
        Uses nose + chin landmarks to approximate face center.
        """
        try:
            if not hand_landmarks or not face_landmarks:
                return 0.0

            # Key face center landmark indices: nose tip (1), chin (152), nose bridge (168)
            face_center_indices = [1, 4, 152, 168]
            face_points = []

            if hasattr(face_landmarks, "__iter__"):
                for idx in face_center_indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        if hasattr(lm, "x") and hasattr(lm, "y"):
                            face_points.append(np.array([lm.x, lm.y]))

            if not face_points:
                return 0.0

            face_center  = np.mean(face_points, axis=0)
            min_distance = float("inf")

            for hand_marks in hand_landmarks:
                if hand_marks is None or len(hand_marks) < 21:
                    continue

                for tip_idx in [4, 8, 12, 16, 20]:
                    mark = hand_marks[tip_idx]
                    if hasattr(mark, "x") and hasattr(mark, "y"):
                        fingertip = np.array([mark.x, mark.y])
                        dist = np.linalg.norm(fingertip - face_center)
                        min_distance = min(min_distance, dist)

            if min_distance == float("inf"):
                return 0.0

            if min_distance >= self.FACE_TOUCHING_THRESHOLD:
                return 0.0

            # Linear: closer = higher score
            score = 1.0 - (min_distance / self.FACE_TOUCHING_THRESHOLD)
            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            print(f"[GestureAgent] Error in face touching score: {e}")
            return 0.0

    def _calculate_confidence(self, hand_landmarks) -> float:
        """
        Overall detection confidence based on landmark presence/visibility scores.
        Falls back to 0.9 if MediaPipe doesn't provide these attributes.
        """
        try:
            confidences = []
            for hand_marks in hand_landmarks:
                if hand_marks is None or len(hand_marks) == 0:
                    continue

                hand_conf = []
                for pt in hand_marks:
                    if hasattr(pt, "presence") and pt.presence is not None:
                        hand_conf.append(pt.presence)
                    elif hasattr(pt, "visibility") and pt.visibility is not None:
                        hand_conf.append(pt.visibility)
                    else:
                        hand_conf.append(0.9)

                if hand_conf:
                    confidences.append(float(np.mean(hand_conf)))

            return float(np.mean(confidences)) if confidences else 0.0

        except Exception as e:
            print(f"[GestureAgent] Error in confidence: {e}")
            return 0.5

    def _reset_hand_state(self) -> None:
        """Reset velocity tracking when hands disappear from frame."""
        self.previous_hand_positions = None
        self.tap_timestamps.clear()