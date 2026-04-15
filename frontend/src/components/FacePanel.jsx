/**
 * FacePanel.jsx
 *
 * Displays emotion probabilities (joy, frustration, fatigue) as animated
 * progress bars, plus blink rate and eye closure index.
 *
 * The emotion bars use CSS width transitions — when the value changes,
 * CSS animates the bar smoothly instead of jumping. That one CSS property
 * (transition: width 0.6s ease) is what makes it feel live and fluid.
 */

import React from 'react';

const EMOTION_CONFIG = [
  { key: 'joy',         label: 'JOY',         color: '#00f5a0' },
  { key: 'frustration', label: 'FRUSTRATION',  color: '#ff4757' },
  { key: 'fatigue',     label: 'FATIGUE',      color: '#f5a623' },
];

function EmotionBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="emotion-row">
      <div className="emotion-meta">
        <span className="emotion-label">{label}</span>
        <span className="emotion-pct" style={{ color }}>{pct}%</span>
      </div>
      <div className="emo-track">
        <div
          className="emo-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}55, ${color})`,
            boxShadow: `0 0 8px ${color}66`,
          }}
        />
      </div>
    </div>
  );
}

function ValenceArousal({ valence, arousal }) {
  // Map valence (-1 to 1) and arousal (0 to 1) to x/y in a 2D canvas
  const x = ((valence + 1) / 2) * 100;
  const y = (1 - arousal) * 100;
  return (
    <div className="va-container">
      <span className="badge-label">VALENCE · AROUSAL</span>
      <div className="va-grid">
        <span className="va-axis-label left">− NEG</span>
        <div className="va-plot">
          <div className="va-crosshair-h" />
          <div className="va-crosshair-v" />
          <div
            className="va-dot"
            style={{ left: `${x}%`, top: `${y}%` }}
          />
        </div>
        <span className="va-axis-label right">POS +</span>
      </div>
      <div className="va-labels-row">
        <span className="va-sub">CALM</span>
        <span className="va-sub">AROUSED</span>
      </div>
    </div>
  );
}

export default function FacePanel({ face }) {
  const f = face || {};
  return (
    <div className="panel face-panel">
      <div className="panel-header">
        <span className="panel-label">FACE · EMOTION</span>
        <span className={`face-dot ${f.face_detected ? 'active' : 'inactive'}`}>
          {f.face_detected ? 'DETECTED' : 'NO FACE'}
        </span>
      </div>

      <div className="emotion-bars">
        {EMOTION_CONFIG.map(({ key, label, color }) => (
          <EmotionBar key={key} label={label} value={f[key]} color={color} />
        ))}
      </div>

      <div className="face-metrics">
        <div className="face-metric">
          <span className="badge-label">BLINK RATE</span>
          <span className="metric-val">{f.blink_rate ?? '—'}<span className="metric-unit"> /min</span></span>
        </div>
        <div className="face-metric">
          <span className="badge-label">EYE CLOSURE</span>
          <span className="metric-val">{f.eye_closure_index != null ? (f.eye_closure_index * 100).toFixed(0) + '%' : '—'}</span>
        </div>
      </div>

      <ValenceArousal valence={f.valence} arousal={f.arousal} />
    </div>
  );
}