import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import FileTypeValidator from './FileTypeValidator';

interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onFileSelect }) => {
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setError(null);
      onFileSelect(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/jpg': ['.jpg'],
      'image/png': ['.png']
    },
    maxFiles: 1
  });

  return (
    <div className="w-full max-w-md mx-auto">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-secondary' : 'border-gray-300 hover:border-primary'}`}
      >
        <input {...getInputProps()} />
        {preview ? (
          <div className="relative">
            <img src={preview} alt="Preview" className="max-h-64 mx-auto" />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setPreview(null);
                onFileSelect(null as any);
              }}
              className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-1"
            >
              ×
            </button>
          </div>
        ) : (
          <div>
            <p className="text-lg">
              {isDragActive
                ? 'Drop the image here'
                : 'Drag and drop an image here, or click to select'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supported formats: JPEG, JPG, PNG
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