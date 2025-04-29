import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import FileTypeValidator from './FileTypeValidator';
import { predictImage } from '../../services/api';

interface ImageUploaderProps {
  onPrediction: (prediction: any) => void;
  onError: (error: string) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onPrediction, onError }) => {
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) {
      setError('No file selected');
      onError('No file selected');
      return;
    }

    setError(null);
    setIsLoading(true);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);

    // Send to API
    try {
      console.log('Sending file to API:', file.name, file.type, file.size);
      const prediction = await predictImage(file);
      console.log('Received prediction:', prediction);
      onPrediction(prediction);
    } catch (err) {
      console.error('Error during prediction:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to predict image';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [onPrediction, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png']
    },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024, // 5MB
    multiple: false
  });

  return (
    <div className="w-full max-w-md mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-secondary' : 'border-gray-300 hover:border-primary'}
          ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <input {...getInputProps()} disabled={isLoading} />
        {preview ? (
          <div className="relative">
            <img src={preview} alt="Preview" className="max-h-64 mx-auto" />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setPreview(null);
              }}
              className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1"
              disabled={isLoading}
            >
              ×
            </button>
          </div>
        ) : (
          <div>
            <p className="text-lg">
              {isLoading 
                ? 'Processing image...'
                : isDragActive
                ? 'Drop the image here'
                : 'Drag and drop an image here, or click to select'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supported formats: JPEG, JPG, PNG (max 5MB)
            </p>
          </div>
        )}
      </div>
      {error && (
        <div className="mt-2 text-red-500 text-sm text-center">{error}</div>
      )}
      <FileTypeValidator file={preview ? new File([], 'preview') : null} onError={setError} />
    </div>
  );
};

export default ImageUploader; 