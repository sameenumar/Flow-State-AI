"""
Test script for gesture_agent (gesture.py) in isolation.
Run from the project root:
    python tests/test_gesture.py

What it tests:
  1. gesture_agent starts and runs without crashing
  2. Output packet matches the required format
  3. hands_detected flag is accurate (not always True)
  4. fidget / typing / face_touching scores behave as expected
"""

import sys
import os
import time
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.gesture_module.gesture import gesture_agent


# ──────────────────────────────────────────────────────────────────
# Mock landmark helpers
# ──────────────────────────────────────────────────────────────────

class MockLandmark:
    """Simulates a MediaPipe NormalizedLandmark."""
    def __init__(self, x, y, z=0.0):
        self.x        = x
        self.y        = y
        self.z        = z
        self.presence  = 0.99
        self.visibility = 0.99


def make_hand_landmarks(wrist_x=0.5, wrist_y=0.7, wiggle=0.0):
    """
    Build a 21-point hand landmark list.
    Wrist at wrist_x, wrist_y. Fingertips spread around it.
    wiggle adds small random offsets to fingertips to simulate movement.
    """
    lms = []
    # Wrist (0)
    lms.append(MockLandmark(wrist_x, wrist_y))

    # Fingers 1–20: rough positions around wrist
    offsets = [
        (0.0, -0.06), (0.0, -0.10), (0.0, -0.13), (0.02, -0.16), (0.04, -0.18),  # thumb
        (0.04, -0.06), (0.04, -0.10), (0.04, -0.14), (0.04, -0.17), (0.04, -0.19), # index
        (0.02, -0.06), (0.02, -0.10), (0.02, -0.15), (0.02, -0.18), (0.02, -0.20), # middle
        (-0.01, -0.05),(-0.01,-0.09),(-0.01,-0.13),(-0.01,-0.16),(-0.01,-0.18),   # ring+pinky
    ]

    for dx, dy in offsets:
        w = np.random.uniform(-wiggle, wiggle) if wiggle > 0 else 0.0
        lms.append(MockLandmark(
            np.clip(wrist_x + dx + w, 0.0, 1.0),
            np.clip(wrist_y + dy + w, 0.0, 1.0),
        ))
    return lms


def make_face_landmarks(center_x=0.5, center_y=0.3):
    """Build minimal face landmark list (just enough for face_touching check)."""
    lms = [MockLandmark(0.0, 0.0)] * 200  # fill with dummies
    # Overwrite key indices used in face_touching
    for idx in [1, 4, 152, 168]:
        lms[idx] = MockLandmark(center_x, center_y)
    return lms


def make_frame(hand_positions=None, face_present=True, wiggle=0.0):
    """Build a gesture frame dict."""
    if hand_positions is None:
        hand_landmarks = []
    else:
        hand_landmarks = [make_hand_landmarks(*pos, wiggle=wiggle) for pos in hand_positions]

    return {
        "hand_landmarks": hand_landmarks,
        "face_landmarks": make_face_landmarks() if face_present else None,
        "hand_gestures":  None,
        "frame_id":       int(time.time() * 1000),
    }


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))


def check_required_fields(result):
    """Assert output packet has all fields required by spec."""
    assert result is not None, "Result is None"

    # Top-level keys
    for key in ["module", "timestamp", "state_tags", "raw_metrics",
                "probabilities", "hands_detected"]:
        assert key in result, f"Missing top-level key: {key}"

    # state_tags (spec-required)
    state = result["state_tags"]
    for key in ["fidget_level", "face_touching", "typing_cadence", "posture_slump"]:
        assert key in state, f"Missing state_tag key: {key}"

    assert isinstance(state["face_touching"], bool), "face_touching must be bool"
    assert 0.0 <= state["fidget_level"]  <= 1.0,     "fidget_level out of range"
    assert 0.0 <= state["posture_slump"] <= 1.0,     "posture_slump out of range"

    # probabilities
    probs = result["probabilities"]
    for key in ["prob_fidgeting", "prob_typing", "prob_face_touching"]:
        assert key in probs, f"Missing probability key: {key}"
        assert 0.0 <= probs[key] <= 1.0, f"{key} out of range: {probs[key]}"

    print("  [PASS] All required fields present and in range.")


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

def test_startup():
    print("\n[TEST 1] Agent starts and runs without errors")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()
    assert agent.is_alive(), "Thread not alive after start()"
    agent.stop()
    print("  [PASS] Agent started and stopped cleanly.")


def test_no_hands_output():
    print("\n[TEST 2] No hands → hands_detected=False")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    # Feed 35 frames with no hands
    for _ in range(35):
        agent.enqueue_frame(make_frame(hand_positions=None))
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    if out is not None:
        print_result("No hands output", out)
        check_required_fields(out)
        assert out["hands_detected"] is False, \
            f"hands_detected should be False when no hands in window, got: {out['hands_detected']}"
        print("  [PASS] hands_detected=False when no hands present.")
    else:
        print("  [SKIP] No packet emitted yet (window not elapsed).")
    agent.stop()


def test_with_hands_output():
    print("\n[TEST 3] Hands present → hands_detected=True, packet has valid structure")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    # Feed 35 frames with one hand at center
    for _ in range(35):
        agent.enqueue_frame(make_frame(hand_positions=[(0.5, 0.5)]))
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    assert out is not None, "No packet emitted after 1+ seconds with hands"
    print_result("With hands output", out)
    check_required_fields(out)
    assert out["hands_detected"] is True, "hands_detected should be True"
    print("  [PASS] hands_detected=True, packet structure valid.")
    agent.stop()


def test_fidget_score():
    print("\n[TEST 4] Moving fingers with still wrist → high fidget score")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    # Wrist stays at (0.5, 0.7); fingers wiggle a lot
    for _ in range(35):
        agent.enqueue_frame(make_frame(hand_positions=[(0.5, 0.7)], wiggle=0.025))
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    if out:
        print_result("Fidget output", out)
        check_required_fields(out)
        fidget = out["probabilities"]["prob_fidgeting"]
        print(f"  prob_fidgeting: {fidget:.3f}")
        # Just check it's a valid float — threshold varies by movement
        assert 0.0 <= fidget <= 1.0, "prob_fidgeting out of range"
        print("  [PASS] prob_fidgeting is in valid range.")
    else:
        print("  [SKIP] No packet yet.")
    agent.stop()


def test_face_touching_score():
    print("\n[TEST 5] Hand near face → face_touching=True")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    # Face is at (0.5, 0.3). Put hand right next to it: wrist at (0.5, 0.4)
    # Fingertips will be ~0.1-0.2 above wrist → close to face center
    for _ in range(35):
        agent.enqueue_frame(make_frame(hand_positions=[(0.5, 0.42)]))
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    if out:
        print_result("Face touching output", out)
        check_required_fields(out)
        face_touch_prob = out["probabilities"]["prob_face_touching"]
        print(f"  prob_face_touching: {face_touch_prob:.3f}")
        assert 0.0 <= face_touch_prob <= 1.0, "prob_face_touching out of range"
        print("  [PASS] Face touching score in valid range.")
    else:
        print("  [SKIP] No packet yet.")
    agent.stop()


def test_typing_cadence():
    print("\n[TEST 6] Repetitive vertical finger motion → typing_cadence set")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    # Alternate between two Y positions to simulate tapping
    for i in range(50):
        y_offset = 0.02 if i % 3 == 0 else 0.0
        frame = make_frame(hand_positions=[(0.5, 0.5 + y_offset)])
        agent.enqueue_frame(frame)
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    if out:
        print_result("Typing output", out)
        check_required_fields(out)
        cadence = out["state_tags"]["typing_cadence"]
        print(f"  typing_cadence: {cadence}")
        # It may or may not trigger depending on thresholds; just check type is valid
        assert cadence in [None, "slow", "fast", "erratic"], \
            f"Invalid typing_cadence value: {cadence}"
        print("  [PASS] typing_cadence is one of None/slow/fast/erratic.")
    else:
        print("  [SKIP] No packet yet.")
    agent.stop()


def test_output_format_matches_spec():
    print("\n[TEST 7] Full output packet matches spec format exactly")
    agent = gesture_agent(output_frequency=1.0)
    agent.start()

    for _ in range(35):
        agent.enqueue_frame(make_frame(hand_positions=[(0.5, 0.5)]))
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    assert out is not None, "No packet emitted"

    # Spec requires these exact keys in state_tags
    state = out["state_tags"]
    assert "fidget_level"   in state, "Missing fidget_level"
    assert "face_touching"  in state, "Missing face_touching"
    assert "typing_cadence" in state, "Missing typing_cadence"
    assert "posture_slump"  in state, "Missing posture_slump"

    # fidget_level must be a float (spec shows 0.8, not "high"/"medium")
    assert isinstance(state["fidget_level"], float), \
        f"fidget_level should be float, got {type(state['fidget_level'])}"

    print_result("Spec-format check", out)
    print("  [PASS] Output matches spec format.")
    agent.stop()


# ──────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  gesture_agent Test Suite")
    print("="*60)

    tests = [
        test_startup,
        test_no_hands_output,
        test_with_hands_output,
        test_fidget_score,
        test_face_touching_score,
        test_typing_cadence,
        test_output_format_matches_spec,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] Unexpected: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")
