import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Display.css';
import Particles from './Particles';

const Display = () => {
    const navigate = useNavigate();
    
    // Sample data - replace with actual data from props or state
    const [imageData, setImageData] = React.useState(null);
    const [heatmapData, setHeatmapData] = React.useState(null);
    const [descriptionData, setDescriptionData] = React.useState([
        'Analysis point 1',
        'Analysis point 2',
        'Analysis point 3'
    ]);
    const [percentage, setPercentage] = React.useState(67);

    const handleBack = () => {
        navigate('/');
    };

    return (
        <div className="display-container">
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

            <div className="display-content">
                <div className="left-section">
                    <div className="section-box image-box">
                        <h2 className="section-title">Image</h2>
                        <div className="content-area">
                            {imageData ? (
                                <img src={imageData} alt="Processed" className="display-image" />
                            ) : (
                                <div className="placeholder-text">Image will appear here</div>
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
                                    <img src={heatmapData} alt="Heatmap" className="heatmap-image" />
                                ) : (
                                    <div className="placeholder-text">Heatmap will appear here</div>
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

export default Display;
