/**
 * VitalsPanel.jsx
 *
 * Shows rPPG vitals: BPM, HRV, stress index, signal quality,
 * and a live scrolling pulse waveform chart.
 *
 * The waveform is drawn on a <canvas> element using the browser's
 * 2D drawing API. Every time `history` changes (new sample arrives),
 * useEffect re-draws the canvas. This is how you do real-time charts
 * without a charting library — just draw lines on a canvas.
 *
 * Props:
 *   vitals  — vitals block from payload
 *   history — array of pulse_sample floats (last 60 seconds)
 */

import React, { useRef, useEffect } from 'react';

function PulseCanvas({ history }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(0,245,160,0.07)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (H / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Normalize to canvas height
    const min = Math.min(...history);
    const max = Math.max(...history);
    const range = max - min || 1;

    const xStep = W / (history.length - 1);

    // Glow effect — draw line twice, wider then sharp
    [{ width: 4, alpha: 0.2 }, { width: 1.5, alpha: 1 }].forEach(({ width, alpha }) => {
      ctx.beginPath();
      ctx.strokeStyle = `rgba(0,245,160,${alpha})`;
      ctx.lineWidth = width;
      ctx.lineJoin = 'round';
      ctx.lineCap  = 'round';

      history.forEach((val, i) => {
        const x = i * xStep;
        const y = H - ((val - min) / range) * (H * 0.8) - H * 0.1;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // Leading dot
    const lastVal = history[history.length - 1];
    const dotX = W - 1;
    const dotY = H - ((lastVal - min) / range) * (H * 0.8) - H * 0.1;
    ctx.beginPath();
    ctx.arc(dotX, dotY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00f5a0';
    ctx.shadowColor = '#00f5a0';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

  }, [history]);

  return <canvas ref={canvasRef} width={340} height={80} className="pulse-canvas" />;
}

function Stat({ label, value, unit, color }) {
  return (
    <div className="vital-stat">
      <span className="vital-label">{label}</span>
      <span className="vital-value" style={{ color: color || '#e0e0ff' }}>
        {value ?? '—'}
        {value != null && unit && <span className="vital-unit"> {unit}</span>}
      </span>
    </div>
  );
}

const QUALITY_COLOR = { good: '#00f5a0', fair: '#f5a623', poor: '#ff4757', warming_up: '#555577' };

export default function VitalsPanel({ vitals, history }) {
  const v = vitals || {};
  const qColor = QUALITY_COLOR[v.signal_quality] || '#555';
  const stressColor = v.stress_index > 0.65 ? '#ff4757' : v.stress_index > 0.35 ? '#f5a623' : '#00f5a0';

  return (
    <div className="panel vitals-panel">
      <div className="panel-header">
        <span className="panel-label">VITALS · rPPG</span>
        <span className="quality-chip" style={{ color: qColor, borderColor: qColor + '44' }}>
          {(v.signal_quality || 'warming up').toUpperCase()}
        </span>
      </div>

      <div className="vitals-grid">
        <Stat label="HEART RATE"   value={v.bpm}          unit="bpm" color="#00f5a0" />
        <Stat label="HRV SDNN"     value={v.hrv_sdnn}     unit="ms"  color="#00c9ff" />
        <Stat label="STRESS INDEX" value={v.stress_index != null ? (v.stress_index * 100).toFixed(0) + '%' : null} color={stressColor} />
        <Stat label="SIGNAL CONF"  value={v.confidence != null ? (v.confidence * 100).toFixed(0) + '%' : null} color="#a0a0c0" />
      </div>

      <div className="pulse-section">
        <span className="badge-label">PULSE WAVEFORM</span>
        <PulseCanvas history={history} />
      </div>
    </div>
  );
}