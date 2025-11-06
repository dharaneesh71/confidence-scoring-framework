/**
 * API service for communicating with the backend
 */
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds for model inference
});

/**
 * Submit a question and get AI answer with confidence score
 */
export const submitQuery = async (question) => {
  try {
    const response = await api.post('/query', { question });
    return response.data;
  } catch (error) {
    throw handleError(error);
  }
};

/**
 * Upload a PDF document to the knowledge base
 */
export const uploadDocument = async (file, username, password) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    // Create a separate axios instance for file upload
    const uploadApi = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120000, // 2 minutes for file upload
    });

    const response = await uploadApi.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      auth: {
        username,
        password,
      },
    });
    return response.data;
  } catch (error) {
    throw handleError(error);
  }
};

/**
 * Get system status
 */
export const getStatus = async () => {
  try {
    const response = await api.get('/status');
    return response.data;
  } catch (error) {
    throw handleError(error);
  }
};

/**
 * Health check
 */
export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw handleError(error);
  }
};

/**
 * Handle API errors consistently
 */
const handleError = (error) => {
  if (error.response) {
    // Server responded with error
    const message = error.response.data?.detail || error.response.data?.error || error.message;
    return new Error(message);
  } else if (error.request) {
    // Request made but no response
    return new Error('No response from server. Please check if the backend is running.');
  } else {
    // Other errors
    return new Error(error.message || 'An unexpected error occurred');
  }
};

export default api;