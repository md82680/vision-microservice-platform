import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation } from 'react-query';
import ImageUploader from '../components/upload/ImageUploader';
import { uploadImage } from '../services/api';
import Header from '../components/common/Header';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  const { mutate: handleUpload, isLoading } = useMutation(
    (file: File) => uploadImage(file),
    {
      onSuccess: (response) => {
        if (response.error) {
          setError(response.error);
        } else {
          navigate('/results', { state: { prediction: response.data } });
        }
      },
      onError: () => {
        setError('An error occurred while processing your image. Please try again.');
      },
    }
  );

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
            <ImageUploader onFileSelect={handleUpload} />
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