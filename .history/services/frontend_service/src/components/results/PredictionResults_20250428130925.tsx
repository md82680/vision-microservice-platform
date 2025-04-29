import React from 'react';
import { Prediction } from '../../types';
import ConfidenceMeter from './ConfidenceMeter';

interface PredictionResultsProps {
  prediction: Prediction | null;
  isLoading: boolean;
  error: string | null;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({
  prediction,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return (
      <div className="text-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mx-auto"></div>
        <p className="mt-4 text-text">Analyzing image...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-8">
        <div className="text-red-500 mb-4">⚠️ {error}</div>
        <p className="text-text">Please try again with a different image.</p>
      </div>
    );
  }

  if (!prediction) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 max-w-md mx-auto">
      <h3 className="text-xl font-semibold text-text mb-4">Prediction Results</h3>
      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-500">Predicted Class</p>
          <p className="text-lg font-medium text-text">{prediction.class}</p>
        </div>
        <ConfidenceMeter confidence={prediction.confidence} />
        <div className="text-sm text-gray-500">
          Analyzed at: {new Date(prediction.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
};

export default PredictionResults; 