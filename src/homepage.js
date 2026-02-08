import React, { useState } from 'react';
import './homepage.css';
import TextType from './TextType';
import Particles from './Particles';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from './AnalysisContext';

const Homepage = () => {
    const navigate = useNavigate();
    const { setAnalysisResults, setAnalysisType } = useAnalysis();
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [fileType, setFileType] = useState('');
    const [fileCategory, setFileCategory] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState('');
    const [statusMessage, setStatusMessage] = useState('');
    const [textInput, setTextInput] = useState('');
    const [inputMode, setInputMode] = useState('text'); 
    const [textBoxWidth, setTextBoxWidth] = useState(500);
    const [textBoxHeight, setTextBoxHeight] = useState(56);

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            setTextInput('');
            setInputMode('file');
            
            const extension = file.name.split('.').pop().toUpperCase();
            setFileType(extension);
            
            if (file.type.startsWith('image/')) {
                setSelectedFile(file);
                setFileCategory('image');
                const reader = new FileReader();
                reader.onloadend = () => {
                    setPreviewUrl(reader.result);
                };
                reader.readAsDataURL(file);
            } else if (file.type.startsWith('audio/') || extension === 'WAV' || extension === 'MP3') {
                setSelectedFile(file);
                setFileCategory('audio');
                setPreviewUrl(null); 
            } else if (file.type === 'text/plain' || extension === 'TXT') {
                // Read text file content
                const reader = new FileReader();
                reader.onload = (e) => {
                    const content = e.target.result;
                    navigate('/display3', { state: { text: content } });
                };
                reader.readAsText(file);
            } else {
                alert('Unsupported file type. Please upload an image, audio (WAV/MP3), or text file.');
                event.target.value = '';
            }
        }
    };

    const handleDeleteFile = (e) => {
        e.stopPropagation();
        setSelectedFile(null);
        setPreviewUrl(null);
        setFileType('');
        setFileCategory('');
        setInputMode('text');
        setTextBoxWidth(500);
        setTextBoxHeight(56);
        document.getElementById('fileInput').value = '';
    };

    const triggerFileInput = () => {
        document.getElementById('fileInput').click();
    };

    const handleTextChange = (e) => {
        const text = e.target.value;
        setTextInput(text);
        
        // Dynamic width expansion (x-axis first)
        const minWidth = 500;
        const maxWidth = 800;
        const charWidth = 8; // approximate character width
        const calculatedWidth = Math.min(maxWidth, Math.max(minWidth, text.length * charWidth + 80));
        setTextBoxWidth(calculatedWidth);
        
        // Dynamic height expansion (y-axis after width maxed)
        const minHeight = 56;
        const maxHeight = 200;
        if (calculatedWidth >= maxWidth) {
            // Calculate height based on text overflow
            const lineCount = Math.ceil((text.length * charWidth) / (maxWidth - 80));
            const calculatedHeight = Math.min(maxHeight, Math.max(minHeight, lineCount * 30));
            setTextBoxHeight(calculatedHeight);
        } else {
            setTextBoxHeight(minHeight);
        }
        
        if (text && selectedFile) {
            setSelectedFile(null);
            setPreviewUrl(null);
            setFileType('');
            setFileCategory('');
            document.getElementById('fileInput').value = '';
        }
        
        setInputMode(text ? 'text' : 'text');
    };

    const handleSubmit = () => {
        console.log('handleSubmit called', { selectedFile, textInput: textInput.trim() });
        
        if (!selectedFile && !textInput.trim()) {
            setUploadStatus('error');
            setStatusMessage('Please select a file or enter text');
            setTimeout(() => {
                setUploadStatus('');
                setStatusMessage('');
            }, 3000);
            return;
        }

        // Handle text input
        if (textInput.trim() && !selectedFile) {
            console.log('Navigating to display3 with text');
            navigate('/display3', { state: { text: textInput } });
            return;
        }

        // Handle file upload
        if (selectedFile) {
            console.log('Navigating to display with file:', selectedFile.name, 'category:', fileCategory);
            
            if (fileCategory === 'image') {
                navigate('/display', { state: { file: selectedFile } });
            } else if (fileCategory === 'audio') {
                navigate('/display2', { state: { file: selectedFile } });
            }
        }
    };

    return (
        <div className="homepage-container">
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
            {!selectedFile && (
                <h1 className="welcome-text">
                    <TextType 
                        text="Welcome to DeepBuster"
                        as="span"
                        typingSpeed={100}
                        loop={false}
                        showCursor={true}
                        cursorCharacter="_"
                    />
                </h1>
            )}
            
            {selectedFile && fileCategory === 'image' && previewUrl && (
                <div className="image-preview">
                    <div className="image-container">
                        <button className="delete-button" onClick={handleDeleteFile}>
                            ×
                        </button>
                        <img src={previewUrl} alt="Preview" />
                        <div className="file-type-label">{fileType}</div>
                    </div>
                </div>
            )}
            
            {selectedFile && fileCategory === 'audio' && (
                <div className="audio-preview">
                    <div className="audio-container">
                        <button className="delete-button" onClick={handleDeleteFile}>
                            ×
                        </button>
                        <div className="audio-icon">🎵</div>
                        <div className="audio-filename">{selectedFile.name}</div>
                        <div className="file-type-label">{fileType}</div>
                    </div>
                </div>
            )}

            <div className={`upload-bar ${previewUrl ? 'has-preview' : ''}`}>
                <input
                    type="file"
                    id="fileInput"
                    accept="image/*,.wav,audio/wav,.mp3,audio/mpeg,.txt,text/plain"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                />
                <button 
                    className="plus-button" 
                    onClick={triggerFileInput}
                    disabled={uploading}
                    title="Upload file"
                >
                    <span className="plus-icon">+</span>
                </button>
                <textarea
                    className="text-input-box"
                    placeholder="Enter text here..."
                    value={textInput}
                    onChange={handleTextChange}
                    disabled={uploading || selectedFile !== null}
                    rows="1"
                    style={{ 
                        width: `${textBoxWidth}px`, 
                        height: `${textBoxHeight}px`,
                        transition: 'width 0.3s ease, height 0.3s ease'
                    }}
                />
                <button 
                    className="arrow-button" 
                    onClick={handleSubmit}
                    disabled={(!selectedFile && !textInput.trim()) || uploading}
                    style={{ opacity: (!selectedFile && !textInput.trim()) || uploading ? 0.5 : 1 }}
                >
                    <span className="arrow-icon">{uploading ? '⏳' : '→'}</span>
                </button>
            </div>
            
            {statusMessage && (
                <div className={`upload-status ${uploadStatus}`}>
                    {statusMessage}
                </div>
            )}
        </div>
    );
};

export default Homepage;