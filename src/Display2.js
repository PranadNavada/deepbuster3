import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Display2.css';
import Particles from './Particles';

const Display2 = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const file = location.state?.file;
    
    const [spectrogramData, setSpectrogramData] = React.useState(null);
    const [heatmapData, setHeatmapData] = React.useState(null);
    const [descriptionData, setDescriptionData] = React.useState([]);
    const [percentage, setPercentage] = React.useState(0);
    const [loading, setLoading] = React.useState(false);

    React.useEffect(() => {
        if (file) {
            analyzeAudio(file);
        }
    }, [file]);

    const analyzeAudio = async (audioFile) => {
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', audioFile);
            
            const response = await fetch('http://localhost:8000/analyze-audio', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            setSpectrogramData(data.spectrogram);
            setHeatmapData(data.heatmap);
            setDescriptionData(data.description || []);
            setPercentage(data.percentage || 0);
        } catch (error) {
            console.error('Error analyzing audio:', error);
            setDescriptionData(['Error: Could not connect to audio analysis server. Make sure port 8000 is running.']);
            setPercentage(0);
        } finally {
            setLoading(false);
        }
    };

    const handleBack = () => {
        navigate('/');
    };

    return (
        <div className="display2-container">
            <Particles
                particleColors={["#ffffff"]}
                particleCount={200}
                particleSpread={10}
                speed={0.1}
                particleBaseSize={100}
                moveParticlesOnHover={false}
                alphaParticles={false}
                disableRotation={false}
                pixelRatio={1}
            />
            
            <div className="logo-container">
                <img src="/logo1.png" alt="DeepBuster Logo" className="logo" />
            </div>

            <button className="back-button" onClick={handleBack}>
                <span className="back-arrow">←</span>
            </button>

            <div className="display2-content">
                {loading && (
                    <div className="loading-overlay">
                        <div className="loading-spinner">Analyzing audio...</div>
                    </div>
                )}
                
                <div className="left-section">
                    <div className="section-box spectrogram-box">
                        <h2 className="section-title">Spectrogram</h2>
                        <div className="content-area">
                            {spectrogramData ? (
                                <img src={spectrogramData} alt="Spectrogram" className="content-image" />
                            ) : (
                                <div className="placeholder-text">Spectrogram will appear here</div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="right-section">
                    <div className="top-right">
                        <div className="section-box heatmap-box">
                            <h2 className="section-title">Heat map</h2>
                            <div className="content-area">
                                {heatmapData ? (
                                    <img src={heatmapData} alt="Heat map" className="content-image" />
                                ) : (
                                    <div className="placeholder-text">Heat map will appear here</div>
                                )}
                            </div>
                        </div>
                        
                        <div className="percentage-box">
                            <svg className="progress-ring" width="140" height="140" viewBox="0 0 160 160">
                                <circle
                                    className="progress-ring-circle-bg"
                                    stroke="rgba(255, 255, 255, 0.1)"
                                    strokeWidth="8"
                                    fill="transparent"
                                    r="60"
                                    cx="80"
                                    cy="80"
                                />
                                <circle
                                    className="progress-ring-circle"
                                    stroke="#ff8c00"
                                    strokeWidth="8"
                                    strokeLinecap="round"
                                    fill="transparent"
                                    r="60"
                                    cx="80"
                                    cy="80"
                                    style={{
                                        strokeDasharray: `${2 * Math.PI * 60}`,
                                        strokeDashoffset: `${2 * Math.PI * 60 * (1 - percentage / 100)}`,
                                        transform: 'rotate(-90deg)',
                                        transformOrigin: '50% 50%',
                                        transition: 'stroke-dashoffset 0.5s ease'
                                    }}
                                />
                                <text
                                    x="80"
                                    y="85"
                                    className="percentage-text"
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                >
                                    {percentage}%
                                </text>
                            </svg>
                        </div>
                    </div>

                    <div className="section-box description-box">
                        <h2 className="section-title">Description</h2>
                        <div className="content-area">
                            {descriptionData && descriptionData.length > 0 ? (
                                <ul className="description-list">
                                    {descriptionData.map((item, index) => (
                                        <li key={index}>{item}</li>
                                    ))}
                                </ul>
                            ) : (
                                <div className="placeholder-text">Description will appear here</div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Display2;
