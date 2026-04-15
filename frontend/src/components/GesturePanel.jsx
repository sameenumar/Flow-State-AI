/**
 * GesturePanel.jsx
 *
 * Shows gesture recognition output: fidget level, typing cadence,
 * posture slump, face touching, and probability bars.
 */

import React from 'react';

const CADENCE_COLOR = {
  slow:    '#00c9ff',
  fast:    '#00f5a0',
  erratic: '#ff4757',
  null:    '#333355',
};

function ProbBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 100);
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="prob-pct" style={{ color }}>{pct}%</span>
    </div>
  );
}

function CadenceBadge({ cadence }) {
  const color = CADENCE_COLOR[cadence] || CADENCE_COLOR[null];
  return (
    <div className="cadence-wrap">
      <span className="badge-label">TYPING CADENCE</span>
      <span className="cadence-badge" style={{ color, borderColor: color + '55', backgroundColor: color + '11' }}>
        {cadence ? cadence.toUpperCase() : 'NONE'}
      </span>
    </div>
  );
}

function SlumpMeter({ value }) {
  const pct = Math.round((value || 0) * 100);
  const color = pct > 60 ? '#ff4757' : pct > 30 ? '#f5a623' : '#00f5a0';
  return (
    <div className="slump-wrap">
      <div className="slump-header">
        <span className="badge-label">POSTURE SLUMP</span>
        <span style={{ color, fontSize: '0.75rem' }}>{pct}%</span>
      </div>
      <div className="slump-track">
        <div className="slump-fill" style={{ width: `${pct}%`, backgroundColor: color, boxShadow: `0 0 6px ${color}88` }} />
      </div>
    </div>
  );
}

export default function GesturePanel({ gesture }) {
  const g = gesture || {};
  return (
    <div className="panel gesture-panel">
      <div className="panel-header">
        <span className="panel-label">GESTURE · BODY</span>
        <span className={`face-dot ${g.hands_detected ? 'active' : 'inactive'}`}>
          {g.hands_detected ? 'HANDS ON' : 'NO HANDS'}
        </span>
      </div>

      <CadenceBadge cadence={g.typing_cadence} />
      <SlumpMeter value={g.posture_slump} />

      <div className="gesture-flags">
        <div className={`flag-chip ${g.face_touching ? 'flagged' : ''}`}>
          {g.face_touching ? '✋ FACE TOUCH' : 'NO FACE TOUCH'}
        </div>
      </div>

      <div className="prob-section">
        <span className="badge-label" style={{ marginBottom: '8px', display: 'block' }}>PROBABILITIES</span>
        <ProbBar label="TYPING"       value={g.prob_typing}        color="#00f5a0" />
        <ProbBar label="FIDGETING"    value={g.prob_fidgeting}     color="#f5a623" />
        <ProbBar label="FACE TOUCH"   value={g.prob_face_touching} color="#ff4757" />
      </div>
    </div>
  );
}