from __future__ import annotations
from typing import Optional

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────
# THRESHOLDS (balanced for real responsiveness)
# ─────────────────────────────────────────────
JOY_HIGH = 0.35
FRUSTRATION_HIGH = 0.30

FOCUS_HIGH = 0.55
FOCUS_MED  = 0.30


# ─────────────────────────────────────────────
# SIMPLE MEMORY (THIS FIXES "STUCK STATE")
# ─────────────────────────────────────────────
_prev_emotion = "neutral"
_emotion_boost = 0.0


# ─────────────────────────────────────────────
# WRAPPERS
# ─────────────────────────────────────────────
class _Face:
    def __init__(self, p):
        p = p or {}
        emo = p.get("emotion_probabilities", {}) or {}
        va  = p.get("valence_arousal", {}) or {}

        self.joy = float(emo.get("joy", 0))
        self.frustration = float(emo.get("frustration", 0))
        self.fatigue = float(emo.get("fatigue", 0))

        self.valence = float(va.get("valence", 0))
        self.arousal = float(va.get("arousal", 0.5))


class _Gesture:
    def __init__(self, p):
        p = p or {}
        probs = p.get("probabilities", {}) or {}

        self.typing = float(probs.get("prob_typing", 0))
        self.fidget = float(probs.get("prob_fidgeting", 0))


class _RPPG:
    def __init__(self, p):
        p = p or {}
        self.stress = float(p.get("stress_index", 0))


# ─────────────────────────────────────────────
# CORE FOCUS SCORE (NOW EMOTION-REACTIVE)
# ─────────────────────────────────────────────
def _score_focus(f, g):
    score = 0.0

    # activity
    score += g.typing * 0.35
    score += (1 - g.fidget) * 0.15

    # physical
    score += (1 - f.fatigue) * 0.10

    # emotion (🔥 NOW REAL IMPACT)
    score += f.joy * 0.30
    score -= f.frustration * 0.40
    score += f.valence * 0.20
    score += f.arousal * 0.10

    return _clamp(score)


# ─────────────────────────────────────────────
# EMOTION CLASSIFIER (WITH MEMORY BOOST)
# ─────────────────────────────────────────────
def _classify_emotion(f):
    global _prev_emotion, _emotion_boost

    current = "neutral"

    if f.joy >= JOY_HIGH:
        current = "positive"
    elif f.frustration >= FRUSTRATION_HIGH or f.valence < -0.2:
        current = "negative"

    # 🔥 emotion change detection (THIS FIXES STUCK BEHAVIOR)
    if current != _prev_emotion:
        _emotion_boost = 0.4
    else:
        _emotion_boost *= 0.85

    _prev_emotion = current

    if current == "positive" and _emotion_boost > 0.1:
        return "positive"
    if current == "negative" and _emotion_boost > 0.1:
        return "negative"

    return "neutral"


# ─────────────────────────────────────────────
# COGNITIVE STATE
# ─────────────────────────────────────────────
def _classify_cognitive(f, g, r):
    focus = _score_focus(f, g)

    if focus >= FOCUS_HIGH:
        return "focused"
    elif focus >= FOCUS_MED:
        return "engaged"
    else:
        return "idle"


# ─────────────────────────────────────────────
# ENGAGEMENT
# ─────────────────────────────────────────────
def _classify_engagement(cog):
    if cog == "focused":
        return "high"
    if cog == "engaged":
        return "medium"
    return "low"


# ─────────────────────────────────────────────
# MESSAGE
# ─────────────────────────────────────────────
def _message(cog, emo):
    if cog == "focused":
        return "You're in a strong focus state. Keep going."
    if cog == "engaged":
        return "You're doing well. Stay consistent."
    if emo == "positive":
        return "Good mood detected. Keep it up!"
    if emo == "negative":
        return "You seem off. Take a short break."
    return "Low engagement detected. Start a task."


# ─────────────────────────────────────────────
# MAIN FUSE
# ─────────────────────────────────────────────
def fuse(face_result: Optional[dict],
         gesture_result: Optional[dict],
         rppg_result: Optional[dict]):

    f = _Face(face_result)
    g = _Gesture(gesture_result)
    r = _RPPG(rppg_result)

    cog = _classify_cognitive(f, g, r)
    emo = _classify_emotion(f)
    eng = _classify_engagement(cog)

    return {
        "cognitive_state": cog,
        "emotional_state": emo,
        "stress_level": "low",
        "engagement": eng,
        "confidence": round(_score_focus(f, g), 2),
        "message": _message(cog, emo)
    }