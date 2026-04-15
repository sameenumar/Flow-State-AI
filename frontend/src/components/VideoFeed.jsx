/**
 * VideoFeed.jsx
 *
 * Displays the live processed frame sent from ai_modules.
 * The frame arrives as a base64 JPEG string inside payload.frame.
 * We just set it as the src of an <img> tag — React handles the rest.
 *
 * Why not use <video>? Because we're not streaming a video — we're
 * receiving individual JPEG frames at 1 Hz over WebSocket and
 * displaying the most recent one. An <img> that updates its src
 * is the correct approach at this frequency.
 */

import React from 'react';

export default function VideoFeed({ frame, connected }) {
  return (
    <div className="video-feed">
      <div className="video-header">
        <span className="panel-label">LIVE FEED</span>
        <span className={`conn-dot ${connected}`} />
      </div>
      <div className="video-wrapper">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Live processed feed"
            className="feed-img"
          />
        ) : (
          <div className="feed-placeholder">
            <div className="pulse-ring" />
            <p>{connected === 'connected' ? 'Waiting for frame...' : 'Connecting to AI module...'}</p>
          </div>
        )}
      </div>
    </div>
  );
}