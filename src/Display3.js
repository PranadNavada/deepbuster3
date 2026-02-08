import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Display3.css';
import Particles from './Particles';

const Display3 = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const text = location.state?.text || '';

    const [textContent, setTextContent] = React.useState('');
    const [descriptionData, setDescriptionData] = React.useState([]);
    const [percentage, setPercentage] = React.useState(0);
    const [loading, setLoading] = React.useState(false);

    const handleBack = () => {
        navigate('/');
    };

    React.useEffect(() => {
        if (!text) {
            console.log('Display3: No text provided');
            return;
        }

        console.log('Display3 received text:', text.substring(0, 50) + '...');
        setTextContent(text);

        const analyzeText = async () => {
            setLoading(true);

            try {
                const response = await fetch("http://127.0.0.1:8001/analyze", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                
                setTextContent(data.text);

                const results = data.results || {};

                const perplexity = results["Perplexity"] ?? results["perplexity"] ?? results["ppl"] ?? "N/A";
                const burstiness = results["Burstiness"] ?? results["burstiness"] ?? "N/A";
                
                let aiPercentage = results["ai_percentage"];
                
                if (aiPercentage === undefined || aiPercentage === null) {
                    aiPercentage = results["ai_pct"] 
                        ?? results["AI Percentage"]
                        ?? results["AI_Percentage"]
                        ?? results["aiPercentage"]
                        ?? data["ai_percentage"]
                        ?? 0;
                }
                
                aiPercentage = Number(aiPercentage) || 0;

                // Fallback: If still 0, compute from prediction
                if (aiPercentage === 0) {
                    const pred = (data.prediction || "").toLowerCase();
                    if (pred.includes("ai") || pred.includes("generated") || pred.includes("machine")) {
                        aiPercentage = 85;
                    } else if (pred.includes("human")) {
                        aiPercentage = 15;
                    } else {
                        aiPercentage = 50;
                    }
                }

                setDescriptionData([
                    `Prediction: ${data.prediction}`,
                    `Perplexity: ${typeof perplexity === 'number' ? perplexity.toFixed(2) : perplexity}`,
                    `Burstiness: ${typeof burstiness === 'number' ? burstiness.toFixed(2) : burstiness}`,
                ]);

                setPercentage(aiPercentage);

            } catch (error) {
                console.error("Backend error:", error);
                setDescriptionData(["Error analyzing text"]);
            } finally {
                setLoading(false);
            }
        };

        analyzeText();
    }, [text]);

    return (
        <div className="display3-container">
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

            <div className="display3-content">
                <div className="top-section">
                    <div className="section-box text-box">
                        <h2 className="section-title">Text</h2>
                        <div className="content-area text-content">
                            {loading ? (
                                <div className="placeholder-text">Analyzing...</div>
                            ) : textContent ? (
                                <p className="text-display">{textContent}</p>
                            ) : (
                                <div className="placeholder-text">Text content will appear here</div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="bottom-section">
                    <div className="section-box description-box">
                        <h2 className="section-title">Description</h2>
                        <div className="content-area">
                            {descriptionData.length > 0 ? (
                                <ul className="description-list">
                                    {descriptionData.map((item, index) => (
                                        <li key={index}>{item}</li>
                                    ))}
                                </ul>
                            ) : (
                                <div className="placeholder-text">
                                    Analysis will appear here
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="percentage-box">
                        <svg 
                            className="progress-ring" 
                            width="140" 
                            height="140" 
                            viewBox="0 0 160 160"
                            key={percentage}
                        >
                            <circle
                                stroke="rgba(255, 255, 255, 0.1)"
                                strokeWidth="8"
                                fill="transparent"
                                r="60"
                                cx="80"
                                cy="80"
                            />
                            <circle
                                stroke="#ff8c00"
                                strokeWidth="8"
                                strokeLinecap="round"
                                fill="transparent"
                                r="60"
                                cx="80"
                                cy="80"
                                strokeDasharray={`${2 * Math.PI * 60}`}
                                strokeDashoffset={`${2 * Math.PI * 60 * (1 - percentage / 100)}`}
                                style={{
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
                        <p className="ai-generated-label">AI Generated</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Display3;