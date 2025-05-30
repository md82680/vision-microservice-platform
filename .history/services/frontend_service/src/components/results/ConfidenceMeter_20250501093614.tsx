import React from 'react';

interface ConfidenceMeterProps {
  confidence: number;
  className?: string;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({ confidence, className = '' }) => {
  // If confidence is already a percentage (0-100), use it directly
  // Otherwise, assume it's a decimal (0-1) and convert to percentage
  const percentage = confidence > 1 ? confidence : Math.round(confidence * 100);

  return (
    <div className={`w-full ${className}`}>
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-text">Confidence</span>
        <span className="text-sm font-medium text-text">{percentage}%</span>
      </div>
      <div className="w-full bg-secondary rounded-full h-2.5">
        <div
          className="bg-primary h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    </div>
  );
};

export default ConfidenceMeter; 