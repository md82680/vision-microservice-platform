import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home.tsx';
import Results from './pages/Results.tsx';
import TrainingHistory from './pages/TrainingHistory.tsx';
import { QueryClient, QueryClientProvider } from 'react-query';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/results" element={<Results />} />
            <Route path="/training-history" element={<TrainingHistory />} />
          </Routes>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App; 