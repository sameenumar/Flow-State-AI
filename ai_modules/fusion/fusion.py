"""
fusion.py

Combines outputs from three agents into a unified state representation.
This module ONLY produces state — it does not generate messages or suggestions.
Messages and suggestions belong in decision.py.

Input format expected from each agent:

  face_result:
    {
      "emotion_probabilities": {"joy": float, "frustration": float, "fatigue": float},
      "valence_arousal":       {"valence": float, "arousal": float},
      "raw_metrics":           {"blink_rate": int, "eye_closure_index": float},
      "face_detected":         bool,
      "confidence":            float  (optional, defaults to 1.0 if face_detected)
    }

  gesture_result:
    {
      "probabilities": {"prob_typing": float, "prob_fidgeting": float, "prob_face_touching": float},
      "state_tags":    {"fidget_level": float, "typing_cadence": str|None, ...},
      "overall_confidence": float,
      "hands_detected": bool
    }

  rppg_result:
    {
      "bpm":            float,
      "hrv_sdnn":       float | None,
      "stress_index":   float,
      "confidence":     float,
      "signal_quality": str
    }

Output format:
    {
      "cognitive_state":   "focused" | "engaged" | "idle",
      "emotional_state":   "positive" | "negative" | "neutral",
      "stress_level":      "high" | "medium" | "low",
      "engagement":        "high" | "medium" | "low",
      "focus_score":       float 0-1,
      "fusion_confidence": float 0-1,
      "timestamp":         float
    }
"""

from __future__ import annotations
import time
from typing import Optional


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────

# Focus score thresholds — what score must be reached for each cognitive label
FOCUS_HIGH = 0.55   # above this → "focused"
FOCUS_MED  = 0.30   # above this → "engaged", below → "idle"

# Emotion classification thresholds
JOY_HIGH          = 0.35   # joy above this → "positive"
FRUSTRATION_HIGH  = 0.30   # frustration above this → "negative"
VALENCE_NEG       = -0.20  # valence below this → "negative"

# Stress thresholds
STRESS_HIGH = 0.65
STRESS_MED  = 0.35


# ─────────────────────────────────────────────────────────────
# INPUT WRAPPERS
# Safe extraction with defaults for every field.
# These normalize the incoming dict formats so the logic
# functions never need to do defensive dict access.
# ─────────────────────────────────────────────────────────────

class _Face:
    """Wraps face_result dict. All fields default to neutral if missing."""
    def __init__(self, p: Optional[dict]):
        p   = p or {}
        emo = p.get("emotion_probabilities") or {}
        va  = p.get("valence_arousal") or {}
        met = p.get("raw_metrics") or {}

        self.joy         = float(emo.get("joy",         0.0))
        self.frustration = float(emo.get("frustration", 0.0))
        self.fatigue     = float(emo.get("fatigue",     0.0))

        # valence: -1 (very negative) to +1 (very positive)
        # arousal: 0 (calm/drowsy) to 1 (excited/agitated)
        self.valence = float(va.get("valence", 0.0))
        self.arousal = float(va.get("arousal", 0.5))

        self.blink_rate       = float(met.get("blink_rate",        0.0))
        self.eye_closure      = float(met.get("eye_closure_index", 0.0))

        # face_detected drives whether we trust any of these values
        self.detected = bool(p.get("face_detected", False))

        # confidence: 1.0 if face detected and agent didn't provide one,
        # 0.0 if no face detected
        self.confidence = float(p.get("confidence", 1.0 if self.detected else 0.0))


class _Gesture:
    """Wraps gesture_result dict."""
    def __init__(self, p: Optional[dict]):
        p     = p or {}
        probs = p.get("probabilities") or {}
        tags  = p.get("state_tags")    or {}

        self.typing       = float(probs.get("prob_typing",       0.0))
        self.fidget       = float(probs.get("prob_fidgeting",    0.0))
        self.face_touch   = float(probs.get("prob_face_touching",0.0))

        self.fidget_level = float(tags.get("fidget_level",  0.0))
        self.posture_slump= float(tags.get("posture_slump", 0.0))

        self.detected   = bool(p.get("hands_detected", False))
        self.confidence = float(p.get("overall_confidence", 0.0))


class _RPPG:
    """Wraps rppg_result dict."""
    def __init__(self, p: Optional[dict]):
        p = p or {}

        self.bpm          = float(p.get("bpm",          0.0))
        self.hrv_sdnn     = p.get("hrv_sdnn")   # float or None
        self.stress       = float(p.get("stress_index", 0.0))
        self.confidence   = float(p.get("confidence",   0.0))
        self.quality      = str(p.get("signal_quality", "poor"))

        # rPPG is only trustworthy when signal quality is fair or good
        # and confidence is above a minimum threshold
        self.reliable = (self.quality in ("good", "fair")) and (self.confidence > 0.3)


# ─────────────────────────────────────────────────────────────
# FUSION CONFIDENCE
# How much to trust the overall fusion output.
# Each agent contributes based on whether it detected its target
# and what its own confidence score is.
# Weighted because rPPG is the most expensive signal to acquire
# (5s warmup) while face is the most reliable in normal conditions.
# ─────────────────────────────────────────────────────────────

def _fusion_confidence(f: _Face, g: _Gesture, r: _RPPG) -> float:
    # Weights represent relative reliability of each modality
    # in normal desktop use conditions
    w_face    = 0.45
    w_gesture = 0.25
    w_rppg    = 0.30

    face_conf    = f.confidence if f.detected    else 0.0
    gesture_conf = g.confidence if g.detected    else 0.0
    rppg_conf    = r.confidence if r.reliable    else 0.0

    score = (face_conf    * w_face +
             gesture_conf * w_gesture +
             rppg_conf    * w_rppg)

    return _clamp(score)


# ─────────────────────────────────────────────────────────────
# FOCUS SCORE
# Measures behavioral engagement and cognitive activity.
# Uses only face emotion and gesture — NOT valence/arousal
# to avoid double-counting (valence is derived from joy/frustration).
# Weights sum to 1.0 on the positive side so the scale is meaningful.
# ─────────────────────────────────────────────────────────────

def _score_focus(f: _Face, g: _Gesture) -> float:
    score = 0.0

    # Behavioral activity signals (0.50 total weight)
    # Typing is the strongest indicator of active cognitive work
    score += g.typing * 0.35
    # Low fidgeting suggests sustained attention (not restlessness)
    score += (1.0 - g.fidget) * 0.15

    # Physical state signals (0.20 total weight)
    # Low fatigue supports focus; high fatigue degrades it
    score += (1.0 - f.fatigue) * 0.15
    # Low posture slump indicates alertness
    score += (1.0 - g.posture_slump) * 0.05

    # Emotional signals (0.30 total weight)
    # Joy is a direct focus enabler (positive affect supports cognition)
    score += f.joy * 0.20
    # Frustration directly disrupts focus — strongest negative weight
    score -= f.frustration * 0.30

    # Total positive maximum: 0.35 + 0.15 + 0.15 + 0.05 + 0.20 = 0.90
    # This gives a realistic ceiling slightly below 1.0 to prevent
    # the focus score from saturating at max unless nearly all signals align.

    return _clamp(score)


# ─────────────────────────────────────────────────────────────
# STRESS CLASSIFICATION
# Uses rPPG as the physiological ground truth.
# Cross-validates with face arousal and fatigue.
# Falls back gracefully when rPPG is unreliable.
# ─────────────────────────────────────────────────────────────

def _classify_stress(f: _Face, r: _RPPG) -> str:
    if r.reliable:
        # rPPG provides direct physiological measurement.
        # Face arousal and fatigue corroborate it.
        # rPPG stress_index already combines BPM deviation and HRV.
        stress_score = (r.stress    * 0.60 +
                        f.arousal   * 0.25 +
                        f.fatigue   * 0.15)
    else:
        # rPPG not available or unreliable — estimate from face only.
        # Less accurate but better than returning a hardcoded value.
        stress_score = (f.arousal   * 0.55 +
                        f.fatigue   * 0.30 +
                        f.frustration * 0.15)

    if stress_score >= STRESS_HIGH:
        return "high"
    elif stress_score >= STRESS_MED:
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────
# COGNITIVE STATE
# Derived from focus score with clear thresholds.
# ─────────────────────────────────────────────────────────────

def _classify_cognitive(f: _Face, g: _Gesture) -> str:
    focus = _score_focus(f, g)

    if focus >= FOCUS_HIGH:
        return "focused"
    elif focus >= FOCUS_MED:
        return "engaged"
    return "idle"


# ─────────────────────────────────────────────────────────────
# EMOTIONAL STATE
# Simple three-way classification. State memory lives in
# FusionEngine instance to avoid global mutable state.
# ─────────────────────────────────────────────────────────────

def _classify_emotion(f: _Face) -> str:
    if f.joy >= JOY_HIGH:
        return "positive"
    elif f.frustration >= FRUSTRATION_HIGH or f.valence < VALENCE_NEG:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────────────────────
# ENGAGEMENT LEVEL
# Derived directly from cognitive state — no additional inputs.
# ─────────────────────────────────────────────────────────────

def _classify_engagement(cog: str) -> str:
    if cog == "focused":
        return "high"
    if cog == "engaged":
        return "medium"
    return "low"


# ─────────────────────────────────────────────────────────────
# FUSION ENGINE
# Class-based so emotion memory is instance state, not globals.
# Instantiate once at startup and call .fuse() on every update.
# ─────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Stateful fusion engine. Maintains emotion memory across calls
    so transient signal noise doesn't cause state to flicker.

    Usage:
        engine = FusionEngine()
        result = engine.fuse(face_result, gesture_result, rppg_result)
    """

    def __init__(self, emotion_decay: float = 0.85):
        """
        Args:
            emotion_decay: How quickly the emotion boost decays per call.
                           0.85 = 15% decay per fusion call.
                           At 1 Hz this means ~6 seconds to fully decay.
        """
        self._prev_emotion   = "neutral"
        self._emotion_boost  = 0.0
        self._emotion_decay  = emotion_decay

    def _apply_emotion_memory(self, raw_emotion: str) -> str:
        if raw_emotion != self._prev_emotion:
            self._emotion_boost = 0.4
        else:
            self._emotion_boost *= self._emotion_decay

        # While boost is active, hold the previous emotion to resist flickering.
        # Only switch once the new emotion has been stable long enough
        # that the boost from the *last* change has decayed away.
        if self._emotion_boost > 0.1:
            return self._prev_emotion  # hold — don't switch yet
        
        self._prev_emotion = raw_emotion
        return raw_emotion

    def fuse(self,
             face_result:    Optional[dict],
             gesture_result: Optional[dict],
             rppg_result:    Optional[dict]) -> dict:
        """
        Combines agent outputs into a unified state representation.

        Returns:
            {
              "cognitive_state":   str,
              "emotional_state":   str,
              "stress_level":      str,
              "engagement":        str,
              "focus_score":       float,
              "fusion_confidence": float,
              "timestamp":         float
            }
        """
        f = _Face(face_result)
        g = _Gesture(gesture_result)
        r = _RPPG(rppg_result)

        cog         = _classify_cognitive(f, g)
        raw_emo     = _classify_emotion(f)
        emo         = self._apply_emotion_memory(raw_emo)
        stress      = _classify_stress(f, r)
        eng         = _classify_engagement(cog)
        focus       = _score_focus(f, g)
        confidence  = _fusion_confidence(f, g, r)

        return {
            "cognitive_state":   cog,
            "emotional_state":   emo,
            "stress_level":      stress,
            "engagement":        eng,
            "focus_score":       round(focus, 3),
            "fusion_confidence": round(confidence, 3),
            "timestamp":         time.time(),
        }


# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE WRAPPER
# For callers that don't need state persistence across calls.
# FusionEngine is the recommended interface for production use.
# ─────────────────────────────────────────────────────────────

_default_engine = FusionEngine()

def fuse(face_result:    Optional[dict] = None,
         gesture_result: Optional[dict] = None,
         rppg_result:    Optional[dict] = None) -> dict:
    """
    Convenience wrapper around FusionEngine.fuse().
    Uses a module-level default engine instance.
    For production use, instantiate FusionEngine directly.
    """
    return _default_engine.fuse(face_result, gesture_result, rppg_result)