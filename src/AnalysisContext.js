import React, { createContext, useContext, useState } from 'react';

const AnalysisContext = createContext();

export const useAnalysis = () => {
    const context = useContext(AnalysisContext);
    if (!context) {
        throw new Error('useAnalysis must be used within AnalysisProvider');
    }
    return context;
};

export const AnalysisProvider = ({ children }) => {
    const [analysisResults, setAnalysisResults] = useState(null);
    const [analysisType, setAnalysisType] = useState(''); // 'image', 'audio', 'text'

    return (
        <AnalysisContext.Provider value={{
            analysisResults,
            setAnalysisResults,
            analysisType,
            setAnalysisType
        }}>
            {children}
        </AnalysisContext.Provider>
    );
};
