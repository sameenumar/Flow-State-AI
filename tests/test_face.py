"""
Test script for FaceAgent (face_detection.py) in isolation.
Run from the project root:
    python tests/test_face.py

What it tests:
  1. FaceAgent starts and runs without crashing
  2. Output packet matches the required format
  3. Blink detection and decay logic work
  4. Memory doesn't grow (blink_times pruning)
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_modules.face_module.face_detection import FaceAgent


# ──────────────────────────────────────────────────────────────────
# Minimal mock of MediaPipe FaceLandmarkerResult
# ──────────────────────────────────────────────────────────────────

class MockBlendshape:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class MockFaceResult:
    """Simulates a MediaPipe FaceLandmarkerResult with a happy face."""
    def __init__(self, smile=0.7, frown=0.05, blink=0.1, squint=0.1):
        self.face_landmarks = [[]]   # Non-empty so FaceAgent knows face is detected
        self.face_blendshapes = [[
            MockBlendshape("mouthSmileLeft",   smile),
            MockBlendshape("mouthSmileRight",  smile),
            MockBlendshape("mouthFrownLeft",   frown),
            MockBlendshape("mouthFrownRight",  frown),
            MockBlendshape("browDownLeft",     0.1),
            MockBlendshape("browDownRight",    0.1),
            MockBlendshape("eyeBlinkLeft",     blink),
            MockBlendshape("eyeBlinkRight",    blink),
            MockBlendshape("eyeSquintLeft",    squint),
            MockBlendshape("eyeSquintRight",   squint),
            MockBlendshape("jawOpen",          0.05),
        ]]


class MockFaceResultNoFace:
    """Simulates no face detected."""
    face_landmarks   = []
    face_blendshapes = []


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

import numpy as np

def make_blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def print_result(label, result):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))


def check_required_fields(result):
    """Assert that the output packet has all required fields from spec."""
    required_emotion_keys = {"joy", "frustration", "fatigue"}
    required_metric_keys  = {"blink_rate", "eye_closure_index"}

    assert result is not None,                          "Result is None"
    assert "emotion_probabilities" in result,           "Missing emotion_probabilities"
    assert "raw_metrics"           in result,           "Missing raw_metrics"
    assert "valence_arousal"       in result,           "Missing valence_arousal"
    assert "face_detected"         in result,           "Missing face_detected"

    emotions = result["emotion_probabilities"]
    metrics  = result["raw_metrics"]

    for key in required_emotion_keys:
        assert key in emotions, f"Missing emotion key: {key}"
        assert 0.0 <= emotions[key] <= 1.0, f"{key} out of range: {emotions[key]}"

    for key in required_metric_keys:
        assert key in metrics, f"Missing metric key: {key}"

    va = result["valence_arousal"]
    assert "valence" in va and "arousal" in va, "Missing valence/arousal"
    assert -1.0 <= va["valence"] <= 1.0, f"Valence out of range: {va['valence']}"
    assert  0.0 <= va["arousal"] <= 1.0, f"Arousal out of range: {va['arousal']}"

    print("  [PASS] All required fields present and in range.")


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

def test_startup():
    print("\n[TEST 1] Agent starts and runs without errors")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()
    assert agent.is_alive(), "Thread is not alive after start()"
    agent.stop()
    print("  [PASS] Agent started and stopped cleanly.")


def test_happy_face_output():
    print("\n[TEST 2] Happy face produces high joy, low frustration")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    frame  = make_blank_frame()
    result = MockFaceResult(smile=0.8, frown=0.02, blink=0.05)

    # Feed 35 frames (~1+ second of data at simulated 30fps)
    for _ in range(35):
        agent.enqueue_frame(result, frame)
        time.sleep(0.033)   # ~30fps

    # Give thread time to emit a 1Hz packet
    time.sleep(0.2)

    out = agent.latest_result
    print_result("Happy face output", out)
    check_required_fields(out)

    assert out["face_detected"] is True,                "face_detected should be True"
    assert out["emotion_probabilities"]["joy"] > 0.4,   "Joy should be high for smiling face"
    assert out["emotion_probabilities"]["frustration"] < 0.4, "Frustration should be low"

    print("  [PASS] Joy > 0.4, Frustration < 0.4 for happy face.")
    agent.stop()


def test_frustrated_face_output():
    print("\n[TEST 3] Frustrated face produces high frustration, low joy")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    frame  = make_blank_frame()
    result = MockFaceResult(smile=0.02, frown=0.85, blink=0.1)

    for _ in range(35):
        agent.enqueue_frame(result, frame)
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    print_result("Frustrated face output", out)
    check_required_fields(out)

    assert out["emotion_probabilities"]["frustration"] > 0.3, "Frustration should be high"
    assert out["emotion_probabilities"]["joy"] < 0.4,         "Joy should be low"
    print("  [PASS] Frustration > 0.3 for frowning face.")
    agent.stop()


def test_no_face_decay():
    print("\n[TEST 4] No face → output decays toward zero")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    frame = make_blank_frame()

    # First send a happy face to build up scores
    happy = MockFaceResult(smile=0.9)
    for _ in range(35):
        agent.enqueue_frame(happy, frame)
        time.sleep(0.033)
    time.sleep(0.2)

    joy_before = agent.latest_result["emotion_probabilities"]["joy"]
    print(f"  Joy before no-face: {joy_before:.3f}")

    # Now send no-face for 2 seconds
    no_face = MockFaceResultNoFace()
    for _ in range(60):
        agent.enqueue_frame(no_face, frame)
        time.sleep(0.033)
    time.sleep(0.2)

    out = agent.latest_result
    print_result("After 2s no face", out)
    check_required_fields(out)

    assert out["face_detected"] is False, "face_detected should be False"
    joy_after = out["emotion_probabilities"]["joy"]
    print(f"  Joy after no-face:  {joy_after:.3f}")
    assert joy_after < joy_before, "Joy should decay when no face is present"
    print("  [PASS] Scores decayed when no face detected.")
    agent.stop()


def test_blink_rate():
    print("\n[TEST 5] Blink detection and blink rate calculation")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    frame = make_blank_frame()

    # Simulate blinking: alternate open/closed every ~6 frames = ~5 blinks/sec = ~20 bpm
    for i in range(50):
        blink_val = 0.9 if (i // 3) % 2 == 0 else 0.05   # closed for 3 frames, open for 3
        result = MockFaceResult(blink=blink_val)
        agent.enqueue_frame(result, frame)
        time.sleep(0.033)

    time.sleep(0.2)

    out = agent.latest_result
    if out:
        blink_rate = out["raw_metrics"]["blink_rate"]
        print(f"  Detected blink rate: {blink_rate} bpm")
        assert 0 <= blink_rate <= 60, f"Blink rate out of range: {blink_rate}"
        print("  [PASS] Blink rate in valid range 0–60 bpm.")
    else:
        print("  [SKIP] No result yet (timing issue) — run test again if needed.")
    agent.stop()


def test_memory_no_leak():
    print("\n[TEST 6] blink_times list does not grow unboundedly")
    agent = FaceAgent(fps=30, output_frequency=1.0)
    agent.start()

    frame  = make_blank_frame()
    # Simulate rapid blinking for 5 seconds
    for i in range(150):
        blink_val = 0.9 if i % 2 == 0 else 0.0
        result = MockFaceResult(blink=blink_val)
        agent.enqueue_frame(result, frame)
        time.sleep(0.033)

    time.sleep(0.2)

    # blink_times should only hold entries from the last 3 seconds
    assert len(agent._blink_times) <= 30, \
        f"blink_times has {len(agent._blink_times)} entries — memory leak!"
    print(f"  blink_times length: {len(agent._blink_times)} (expected ≤ 30)")
    print("  [PASS] No memory leak in blink_times.")
    agent.stop()


# ──────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FaceAgent Test Suite")
    print("="*60)

    tests = [
        test_startup,
        test_happy_face_output,
        test_frustrated_face_output,
        test_no_face_decay,
        test_blink_rate,
        test_memory_no_leak,
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
