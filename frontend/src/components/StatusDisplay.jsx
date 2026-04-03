import React from 'react';

const StatusDisplay = ({ analysisResult }) => {
  if (!analysisResult) {
    return <div className="status-display">Waiting for analysis...</div>;
  }

  const { success, flow_state, confidence, face_data, gesture_data, rppg_data } = analysisResult;

  if (!success) {
    return <div className="status-display error">Analysis failed</div>;
  }

  return (
    <div className="status-display">
      <h2>Analysis Results</h2>
      
      <div className="result-section">
        <h3>Flow State</h3>
        <p className="flow-state">{flow_state || 'Unknown'}</p>
        <p className="confidence">Confidence: {(confidence * 100).toFixed(1)}%</p>
      </div>

      {face_data && (
        <div className="result-section">
          <h3>Face Detection</h3>
          <p>Faces Detected: {face_data.faces_detected || 0}</p>
          <p>Confidence: {(face_data.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {gesture_data && (
        <div className="result-section">
          <h3>Gesture Recognition</h3>
          <p>Gesture: {gesture_data.gesture || 'None'}</p>
          <p>Confidence: {(gesture_data.confidence * 100).toFixed(1)}%</p>
        </div>
      )}

      {rppg_data && (
        <div className="result-section">
          <h3>Vital Signs (rPPG)</h3>
          <p>Heart Rate: {rppg_data.heart_rate?.toFixed(1) || 'N/A'} bpm</p>
          <p>SpO2: {rppg_data.spo2?.toFixed(1) || 'N/A'}%</p>
          <p>Confidence: {(rppg_data.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
};

export default StatusDisplay;
