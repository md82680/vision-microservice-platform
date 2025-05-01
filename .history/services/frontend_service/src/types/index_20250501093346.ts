export interface Prediction {
  class?: string;
  predicted_class?: string;
  confidence: number;
  timestamp?: string;
}

export interface TrainingMetrics {
  accuracy: number;
  loss: number;
  trainingTime: string;
  lastUpdated: string;
}

export interface ApiResponse<T> {
  data: T;
  error?: string;
  status: number;
} 