/**
 * StatePanel.jsx
 *
 * Shows the fusion output: cognitive state, emotional state, stress level,
 * engagement, focus score, and the decision message + suggestion.
 *
 * Props it receives:
 *   fusion   — the fusion block from the payload
 *   decision — the decision block from the payload
 *
 * Components receive data as "props" — think of props as the arguments
 * you pass to a function. The parent (Dashboard) passes data down,
 * this component just renders what it receives.
 */

import React from 'react';

const STATE_COLORS = {
  focused:  '#00f5a0',
  engaged:  '#00c9ff',
  idle:     '#a0a0b0',
  unknown:  '#555',
};

const EMOTION_COLORS = {
  positive: '#00f5a0',
  neutral:  '#f5a623',
  negative: '#ff4757',
  unknown:  '#555',
};

const STRESS_COLORS = {
  low:     '#00f5a0',
  medium:  '#f5a623',
  high:    '#ff4757',
  unknown: '#555',
};

const PRIORITY_COLORS = {
  passive: '#555577',
  normal:  '#f5a623',
  urgent:  '#ff4757',
};

function Badge({ label, value, colorMap }) {
  const color = colorMap[value] || colorMap.unknown || '#555';
  return (
    <div className="badge-item">
      <span className="badge-label">{label}</span>
      <span className="badge-value" style={{ color, borderColor: color + '44', backgroundColor: color + '11' }}>
        {value || '—'}
      </span>
    </div>
  );
}

function FocusBar({ score }) {
  const pct = Math.round((score || 0) * 100);
  const color = pct > 60 ? '#00f5a0' : pct > 35 ? '#f5a623' : '#ff4757';
  return (
    <div className="focus-bar-wrap">
      <div className="focus-bar-header">
        <span className="badge-label">FOCUS SCORE</span>
        <span className="focus-pct" style={{ color }}>{pct}%</span>
      </div>
      <div className="focus-bar-track">
        <div
          className="focus-bar-fill"
          style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}88, ${color})` }}
        />
      </div>
    </div>
  );
}

export default function StatePanel({ fusion, decision }) {
  const f = fusion   || {};
  const d = decision || {};
  const priority = d.priority || 'passive';

  return (
    <div className="panel state-panel">
      <div className="panel-header">
        <span className="panel-label">COGNITIVE STATE</span>
        <span className="conf-chip">
          CONF {Math.round((f.fusion_confidence || 0) * 100)}%
        </span>
      </div>

      <div className="badge-grid">
        <Badge label="STATE"    value={f.cognitive_state} colorMap={STATE_COLORS}   />
        <Badge label="EMOTION"  value={f.emotional_state} colorMap={EMOTION_COLORS} />
        <Badge label="STRESS"   value={f.stress_level}    colorMap={STRESS_COLORS}  />
        <Badge label="ENGAGE"   value={f.engagement}      colorMap={STATE_COLORS}   />
      </div>

      <FocusBar score={f.focus_score} />

      <div className="decision-block" style={{ borderColor: PRIORITY_COLORS[priority] + '66' }}>
        <p className="decision-msg">{d.message || '—'}</p>
        <p className="decision-tip">{d.suggestion || ''}</p>
        {d.alert && <span className="alert-chip">⚠ ALERT</span>}
      </div>
    </div>
  );
}