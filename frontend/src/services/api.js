import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Analysis API calls
export const analyzeFlowState = async (imageData) => {
  try {
    const response = await api.post('/api/analyze', {
      image_data: imageData,
      analyze_face: true,
      analyze_gesture: true,
      analyze_rppg: true,
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing flow state:', error);
    throw error;
  }
};

// Health check
export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export default api;
