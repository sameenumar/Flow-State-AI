"""
decision.py

Reads the structured state dict produced by fusion.py and generates
user-facing messages and actionable suggestions.

This module has NO knowledge of raw agent outputs — it only reads
the normalized state representation from FusionEngine.fuse().

Separation rationale:
  fusion.py  → WHAT state the person is in (measurement)
  decision.py → WHAT to tell or do about it (interpretation + action)

Input format (fusion output):
    {
      "cognitive_state":   "focused" | "engaged" | "idle",
      "emotional_state":   "positive" | "negative" | "neutral",
      "stress_level":      "high" | "medium" | "low",
      "engagement":        "high" | "medium" | "low",
      "focus_score":       float 0-1,
      "fusion_confidence": float 0-1,
      "timestamp":         float
    }

Output format:
    {
      "message":    str,       — primary one-line status shown to user
      "suggestion": str,       — actionable tip or recommendation
      "priority":   str,       — "urgent" | "normal" | "passive"
      "alert":      bool,      — True if immediate user attention needed
    }
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────────────────────
# CONFIDENCE GATE
# Below this threshold, fusion output is too uncertain to act on.
# We report a waiting state instead of potentially wrong decisions.
# ─────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.25


# ─────────────────────────────────────────────────────────────
# MESSAGE + SUGGESTION TABLES
# Keyed by (cognitive_state, emotional_state, stress_level).
# More specific combinations take precedence over general ones.
# ─────────────────────────────────────────────────────────────

# Primary message: describes what's happening
_MESSAGES = {
    # Focused states
    ("focused",  "positive",  "low"):    "Deep focus with positive energy.",
    ("focused",  "positive",  "medium"): "Strong focus — mild stress present.",
    ("focused",  "positive",  "high"):   "Focused but physiological stress is elevated.",
    ("focused",  "neutral",   "low"):    "Clean focus state detected.",
    ("focused",  "neutral",   "medium"): "Good focus. Watch your stress levels.",
    ("focused",  "neutral",   "high"):   "Focused but stress is high — consider a break soon.",
    ("focused",  "negative",  "low"):    "Focused despite negative mood.",
    ("focused",  "negative",  "medium"): "Working through difficulty — stress building.",
    ("focused",  "negative",  "high"):   "High stress and frustration. Sustaining is risky.",

    # Engaged states
    ("engaged",  "positive",  "low"):    "Positive and engaged. Good momentum.",
    ("engaged",  "positive",  "medium"): "Engaged with moderate stress.",
    ("engaged",  "positive",  "high"):   "Good mood but stress is high.",
    ("engaged",  "neutral",   "low"):    "Steady engagement.",
    ("engaged",  "neutral",   "medium"): "Moderate engagement, moderate stress.",
    ("engaged",  "neutral",   "high"):   "Stress is elevated. Consider stepping back.",
    ("engaged",  "negative",  "low"):    "Negative mood but still engaged.",
    ("engaged",  "negative",  "medium"): "Frustration and stress building.",
    ("engaged",  "negative",  "high"):   "Stress and frustration are high. Take a break.",

    # Idle states
    ("idle",     "positive",  "low"):    "Relaxed and calm.",
    ("idle",     "positive",  "medium"): "Good mood but low activity.",
    ("idle",     "positive",  "high"):   "Calm mood but body stress is elevated.",
    ("idle",     "neutral",   "low"):    "Low engagement detected.",
    ("idle",     "neutral",   "medium"): "Disengaged with rising stress.",
    ("idle",     "neutral",   "high"):   "Low engagement and high stress.",
    ("idle",     "negative",  "low"):    "Negative mood, low activity.",
    ("idle",     "negative",  "medium"): "Feeling down and stressed.",
    ("idle",     "negative",  "high"):   "High stress and low engagement. Intervention recommended.",
}

# Suggestion: what the user should do about their current state
_SUGGESTIONS = {
    ("focused",  "positive",  "low"):    "Keep going — conditions are optimal.",
    ("focused",  "positive",  "medium"): "Take a 2-minute breathing pause in 15 minutes.",
    ("focused",  "positive",  "high"):   "Your heart rate is elevated. A short walk would help.",
    ("focused",  "neutral",   "low"):    "Good time to tackle your hardest task.",
    ("focused",  "neutral",   "medium"): "Schedule a break within 20 minutes.",
    ("focused",  "neutral",   "high"):   "Take a 5-minute break before continuing.",
    ("focused",  "negative",  "low"):    "Push through — the work is helping.",
    ("focused",  "negative",  "medium"): "Note what's frustrating you. Address it after this session.",
    ("focused",  "negative",  "high"):   "Stop and reset. Short break now is more productive.",

    ("engaged",  "positive",  "low"):    "Good time to increase task difficulty.",
    ("engaged",  "positive",  "medium"): "Maintain current pace. Monitor stress.",
    ("engaged",  "positive",  "high"):   "Drink water and take a standing break soon.",
    ("engaged",  "neutral",   "low"):    "Try a more challenging task to increase focus.",
    ("engaged",  "neutral",   "medium"): "Pace yourself. Take a 5-minute break soon.",
    ("engaged",  "neutral",   "high"):   "Take a break now. High stress reduces output quality.",
    ("engaged",  "negative",  "low"):    "Identify the source of frustration.",
    ("engaged",  "negative",  "medium"): "Step away for 5 minutes to reset.",
    ("engaged",  "negative",  "high"):   "Take a meaningful break. Your output is at risk.",

    ("idle",     "positive",  "low"):    "Start with a small, easy task to build momentum.",
    ("idle",     "positive",  "medium"): "Light activity or a walk could help.",
    ("idle",     "positive",  "high"):   "Check in with how you're feeling physically.",
    ("idle",     "neutral",   "low"):    "Set a 10-minute focused work timer.",
    ("idle",     "neutral",   "medium"): "Take a break then return with a clear goal.",
    ("idle",     "neutral",   "high"):   "Rest. Do not force productivity right now.",
    ("idle",     "negative",  "low"):    "Short walk or change of environment may help.",
    ("idle",     "negative",  "medium"): "Take a proper break away from the screen.",
    ("idle",     "negative",  "high"):   "Step away from work completely for at least 15 minutes.",
}

# Priority: how urgently the suggestion should be surfaced
_PRIORITY = {
    "low":    "passive",
    "medium": "normal",
    "high":   "urgent",
}

# Alert: whether to visually flag this state to the user immediately
_ALERT_CONDITIONS = {
    ("idle", "negative", "high"),
    ("focused", "negative", "high"),
    ("engaged", "negative", "high"),
}


def decide(fusion_state: Optional[dict]) -> dict:
    """
    Generates a user-facing decision packet from a fusion state dict.

    Args:
        fusion_state: Output dict from FusionEngine.fuse(). May be None
                      during warmup before any fusion result is available.

    Returns:
        {
          "message":    str,
          "suggestion": str,
          "priority":   str,
          "alert":      bool,
        }
    """
    # Default response during warmup or if fusion hasn't run yet
    if fusion_state is None:
        return {
            "message":    "Initializing sensors...",
            "suggestion": "Please remain visible to the camera.",
            "priority":   "passive",
            "alert":      False,
        }

    # Below confidence threshold, don't make decisions
    confidence = fusion_state.get("fusion_confidence", 0.0)
    if confidence < MIN_CONFIDENCE:
        return {
            "message":    "Gathering data...",
            "suggestion": "Ensure your face is clearly visible.",
            "priority":   "passive",
            "alert":      False,
        }

    cog    = fusion_state.get("cognitive_state",  "idle")
    emo    = fusion_state.get("emotional_state",  "neutral")
    stress = fusion_state.get("stress_level",     "low")

    key = (cog, emo, stress)

    message    = _MESSAGES.get(key,    f"State: {cog}, {emo}, stress {stress}.")
    suggestion = _SUGGESTIONS.get(key, "Continue monitoring.")
    priority   = _PRIORITY.get(stress, "passive")
    alert      = key in _ALERT_CONDITIONS

    return {
        "message":    message,
        "suggestion": suggestion,
        "priority":   priority,
        "alert":      alert,
    }