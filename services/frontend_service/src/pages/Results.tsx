import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Prediction } from '../types';
import PredictionResults from '../components/results/PredictionResults';
import Header from '../components/common/Header';

const Results: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const prediction = location.state?.prediction as Prediction | null;

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h1 className="text-2xl font-bold text-text mb-6">Prediction Results</h1>
            <PredictionResults
              prediction={prediction}
              isLoading={false}
              error={!prediction ? 'No prediction results available' : null}
            />
            <div className="mt-6 text-center">
              <button
                onClick={() => navigate('/')}
                className="bg-primary text-white px-6 py-2 rounded-lg hover:bg-accent transition-colors"
              >
                Try Another Image
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results; 