import React, { useRef, useEffect, useState } from 'react';

const WebcamFeed = ({ onCapture }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    const startWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsRunning(true);
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
      }
    };

    startWebcam();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      context.drawImage(videoRef.current, 0, 0, 640, 480);
      const imageData = canvasRef.current.toDataURL('image/jpeg');
      onCapture(imageData);
    }
  };

  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        width="640"
        height="480"
        style={{ display: 'block', margin: '0 auto' }}
      />
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{ display: 'none' }}
      />
      <button onClick={captureFrame} disabled={!isRunning}>
        Capture & Analyze
      </button>
    </div>
  );
};

export default WebcamFeed;
