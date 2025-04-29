import React from 'react';

interface FileTypeValidatorProps {
  file: File | null;
  onError: (message: string) => void;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

const FileTypeValidator: React.FC<FileTypeValidatorProps> = ({ file, onError }) => {
  React.useEffect(() => {
    if (file) {
      if (!ALLOWED_TYPES.includes(file.type)) {
        onError('Please upload a valid image file (JPEG, JPG, or PNG)');
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        onError('File size should be less than 5MB');
        return;
      }
    }
  }, [file, onError]);

  return null;
};

export default FileTypeValidator; 