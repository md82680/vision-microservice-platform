import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ImageUploader from '../components/upload/ImageUploader';
import Header from '../components/common/Header';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = (prediction: any) => {
    // Navigate to results page with the prediction data
    navigate('/results', { state: { prediction: prediction } });
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto text-center">
          <h1 className="text-3xl font-bold text-text mb-4">
            Image Classification System
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            Upload an image to classify it using our trained model. Supported formats: JPEG, JPG, PNG
          </p>
          
          <div className="bg-white rounded-lg shadow-md p-6">
            <ImageUploader 
              onPrediction={handlePrediction}
              onError={handleError}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
            {isLoading && (
              <div className="mt-4">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary mx-auto"></div>
                <p className="mt-2 text-text">Processing your image...</p>
              </div>
            )}
            {error && (
              <div className="mt-4 text-red-500">
                {error}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Home; 