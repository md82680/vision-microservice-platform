import axios from 'axios';
import { Prediction, TrainingMetrics, ApiResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadImage = async (file: File): Promise<ApiResponse<Prediction>> => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return { data: response.data, status: response.status };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return {
        data: {} as Prediction,
        error: error.response?.data?.message || 'An error occurred',
        status: error.response?.status || 500,
      };
    }
    throw error;
  }
};

export const getTrainingMetrics = async (): Promise<ApiResponse<TrainingMetrics>> => {
  try {
    const response = await api.get('/training-metrics');
    return { data: response.data, status: response.status };
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return {
        data: {} as TrainingMetrics,
        error: error.response?.data?.message || 'An error occurred',
        status: error.response?.status || 500,
      };
    }
    throw error;
  }
};

export const predictImage = async (file: File): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await api.post('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error predicting image:', error);
    throw error;
  }
}; 