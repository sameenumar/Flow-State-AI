import React, { useState, useCallback } from 'react';
import WebcamFeed from '../components/WebcamFeed';
import StatusDisplay from '../components/StatusDisplay';
import Alerts from '../components/Alerts';
import { analyzeFlowState } from '../services/api';

const Home = () => {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCapture = useCallback(async (imageData) => {
    setIsLoading(true);
    setError(null);

    try {
      // Remove data URI prefix to get base64
      const base64Data = imageData.split(',')[1];
      const result = await analyzeFlowState(base64Data);
      setAnalysisResult(result);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <div className="home-page">
      <h1>Flow-State-AI</h1>
      <p>Detect and analyze your flow state in real-time</p>

      <div className="main-container">
        <div className="webcam-section">
          <WebcamFeed onCapture={handleCapture} />
          {isLoading && <p className="loading">Analyzing...</p>}
        </div>

        <div className="results-section">
          {error && <div className="error-message">{error}</div>}
          <StatusDisplay analysisResult={analysisResult} />
          <Alerts analysisResult={analysisResult} />
        </div>
      </div>
    </div>
  );
};

export default Home;
