/**
 * Dashboard.jsx
 *
 * The single page of the app. It:
 * 1. Calls useFlowData() to get live data from the WebSocket
 * 2. Passes the right slice of data to each panel component
 *
 * Think of this as the "orchestrator" — it knows where data comes from
 * and which component needs what. The panels themselves are dumb;
 * they just render whatever they receive.
 *
 * Layout:
 *   ┌─────────────────┬──────────────────┐
 *   │   VIDEO FEED    │   STATE PANEL    │
 *   │                 │                  │
 *   ├────────┬────────┴──────────────────┤
 *   │ VITALS │   FACE   │   GESTURE      │
 *   └────────┴──────────┴────────────────┘
 */

import React from 'react';
import { useFlowData } from '../hooks/useFlowData';
import VideoFeed    from '../components/VideoFeed';
import StatePanel   from '../components/StatePanel';
import VitalsPanel  from '../components/VitalsPanel';
import FacePanel    from '../components/FacePanel';
import GesturePanel from '../components/GesturePanel';

const STATUS_LABEL = {
  connected:    'LIVE',
  disconnected: 'OFFLINE',
  error:        'ERROR',
};
const STATUS_COLOR = {
  connected:    '#00f5a0',
  disconnected: '#ff4757',
  error:        '#f5a623',
};

export default function Dashboard() {
  const { data, connected, history } = useFlowData();

  const statusColor = STATUS_COLOR[connected] || '#555';
  const lastUpdate  = data?.timestamp
    ? new Date(data.timestamp * 1000).toLocaleTimeString()
    : '—';

  return (
    <div className="dashboard">

      {/* ── Top bar ─────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-left">
          <span className="logo-text">FLOW<span className="logo-accent">STATE</span></span>
          <span className="logo-sub">AI · COGNITIVE MONITOR</span>
        </div>
        <div className="topbar-right">
          <span className="update-time">LAST UPDATE {lastUpdate}</span>
          <span className="status-pill" style={{ color: statusColor, borderColor: statusColor + '55', backgroundColor: statusColor + '11' }}>
            <span className="status-blink" style={{ backgroundColor: statusColor }} />
            {STATUS_LABEL[connected] || 'OFFLINE'}
          </span>
        </div>
      </header>

      {/* ── Main grid ───────────────────────────────────────────── */}
      <main className="grid">

        {/* Row 1 */}
        <div className="grid-video">
          <VideoFeed frame={data?.frame} connected={connected} />
        </div>

        <div className="grid-state">
          <StatePanel fusion={data?.fusion} decision={data?.decision} />
        </div>

        {/* Row 2 */}
        <div className="grid-vitals">
          <VitalsPanel vitals={data?.vitals} history={history} />
        </div>

        <div className="grid-face">
          <FacePanel face={data?.face} />
        </div>

        <div className="grid-gesture">
          <GesturePanel gesture={data?.gesture} />
        </div>

      </main>

      <footer className="dash-footer">
        <span>FLOW-STATE-AI · REAL-TIME COGNITIVE MONITORING</span>
      </footer>
    </div>
  );
}