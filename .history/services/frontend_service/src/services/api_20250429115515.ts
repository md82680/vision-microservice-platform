import axios from 'axios';
import { Prediction, TrainingMetrics, ApiResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

interface FastAPIError {
  detail?: string | Array<{ type: string; loc: string[]; msg: string; input: any; ctx: any }>;
  message?: string;
}

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
      const apiError = error.response?.data as FastAPIError;
      let errorMessage = 'An error occurred';
      if (apiError?.detail) {
        if (typeof apiError.detail === 'string') {
          errorMessage = apiError.detail;
        } else if (Array.isArray(apiError.detail) && apiError.detail.length > 0) {
          errorMessage = apiError.detail[0].msg;
        }
      } else if (apiError?.message) {
        errorMessage = apiError.message;
      }
      return {
        data: {} as Prediction,
        error: errorMessage,
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
      const apiError = error.response?.data as FastAPIError;
      let errorMessage = 'An error occurred';
      if (apiError?.detail) {
        if (typeof apiError.detail === 'string') {
          errorMessage = apiError.detail;
        } else if (Array.isArray(apiError.detail) && apiError.detail.length > 0) {
          errorMessage = apiError.detail[0].msg;
        }
      } else if (apiError?.message) {
        errorMessage = apiError.message;
      }
      return {
        data: {} as TrainingMetrics,
        error: errorMessage,
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
    const response = await api.post('/predict', formData);
    return response.data;
  } catch (error) {
    console.error('Error predicting image:', error);
    if (axios.isAxiosError(error)) {
      const apiError = error.response?.data as FastAPIError;
      let errorMessage = 'Failed to predict image';
      if (apiError?.detail) {
        if (typeof apiError.detail === 'string') {
          errorMessage = apiError.detail;
        } else if (Array.isArray(apiError.detail) && apiError.detail.length > 0) {
          errorMessage = apiError.detail[0].msg;
        }
      } else if (apiError?.message) {
        errorMessage = apiError.message;
      }
      throw new Error(errorMessage);
    }
    throw error;
  }
}; 