import React from 'react';
import { useQuery } from 'react-query';
import { getTrainingMetrics } from '../services/api';
import { TrainingMetrics } from '../types';
import Header from '../components/common/Header';
import LoadingSpinner from '../components/common/LoadingSpinner';

const TrainingHistory: React.FC = () => {
  const { data, isLoading, error } = useQuery<TrainingMetrics>(
    'trainingMetrics',
    async () => {
      const response = await getTrainingMetrics();
      if (response.error) {
        throw new Error(response.error);
      }
      return response.data;
    },
    {
      refetchInterval: 300000, // Refetch every 5 minutes
    }
  );

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-2xl font-bold text-text mb-6">Training History</h1>
          
          {isLoading ? (
            <LoadingSpinner />
          ) : error ? (
            <div className="bg-white rounded-lg shadow-md p-6 text-center">
              <div className="text-red-500 mb-4">⚠️ Error loading training metrics</div>
              <p className="text-text">Please try again later.</p>
            </div>
          ) : data ? (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold text-text mb-2">Model Accuracy</h3>
                  <div className="w-full bg-secondary rounded-full h-2.5">
                    <div
                      className="bg-primary h-2.5 rounded-full transition-all duration-300"
                      style={{ width: `${data.accuracy * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-right text-sm text-gray-500 mt-1">
                    {Math.round(data.accuracy * 100)}%
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-text mb-2">Model Loss</h3>
                  <div className="w-full bg-secondary rounded-full h-2.5">
                    <div
                      className="bg-primary h-2.5 rounded-full transition-all duration-300"
                      style={{ width: `${(1 - data.loss) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-right text-sm text-gray-500 mt-1">
                    {data.loss.toFixed(4)}
                  </p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-text mb-2">Training Time</h3>
                  <p className="text-text">{data.trainingTime}</p>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-text mb-2">Last Updated</h3>
                  <p className="text-text">
                    {new Date(data.lastUpdated).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </main>
    </div>
  );
};

export default TrainingHistory; 