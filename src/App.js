import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AnalysisProvider } from './AnalysisContext';
import Homepage from './homepage';
import Display from './Display';
import Display2 from './Display2';
import Display3 from './Display3';
import './App.css';

function App() {
  return (
    <AnalysisProvider>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<Homepage />} />
            <Route path="/display" element={<Display />} />
            <Route path="/display2" element={<Display2 />} />
            <Route path="/display3" element={<Display3 />} />
          </Routes>
        </div>
      </Router>
    </AnalysisProvider>
  );
}

export default App;