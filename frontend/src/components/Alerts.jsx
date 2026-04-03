import React, { useState, useEffect } from 'react';

const Alerts = ({ analysisResult }) => {
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    if (!analysisResult || !analysisResult.success) {
      return;
    }

    const newAlerts = [];

    // Check flow state
    if (analysisResult.flow_state === 'not_in_flow') {
      newAlerts.push({
        type: 'warning',
        message: 'Not currently in flow state. Check focus level.',
      });
    }

    // Check heart rate
    if (analysisResult.rppg_data) {
      const { heart_rate } = analysisResult.rppg_data;
      if (heart_rate > 120) {
        newAlerts.push({
          type: 'warning',
          message: 'Heart rate is elevated. Take a break if needed.',
        });
      } else if (heart_rate < 60) {
        newAlerts.push({
          type: 'info',
          message: 'Heart rate is low. Ensure you are relaxed.',
        });
      }
    }

    // Check SpO2
    if (analysisResult.rppg_data?.spo2 < 95) {
      newAlerts.push({
        type: 'warning',
        message: 'Low oxygen levels. Ensure proper ventilation.',
      });
    }

    setAlerts(newAlerts);
  }, [analysisResult]);

  if (alerts.length === 0) {
    return null;
  }

  return (
    <div className="alerts-container">
      {alerts.map((alert, index) => (
        <div key={index} className={`alert alert-${alert.type}`}>
          {alert.message}
        </div>
      ))}
    </div>
  );
};

export default Alerts;
