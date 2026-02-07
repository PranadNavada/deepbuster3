import React, { useState } from 'react';
import './homepage.css';
import TextType from './TextType';
import { uploadFileToFirebase } from './firebaseService';
import Particles from './Particles';

const Homepage = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [fileType, setFileType] = useState('');
    const [fileCategory, setFileCategory] = useState('');
    const [uploading, setUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState('');

    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
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
            } else if (file.type.startsWith('audio/') || extension === 'WAV') {
                setSelectedFile(file);
                setFileCategory('audio');
                setPreviewUrl(null); 
            }
        }
    };

    const handleDeleteFile = (e) => {
        e.stopPropagation();
        setSelectedFile(null);
        setPreviewUrl(null);
        setFileType('');
        setFileCategory('');
        document.getElementById('fileInput').value = '';
    };

    const triggerFileInput = () => {
        document.getElementById('fileInput').click();
    };

    const handleSubmit = async () => {
        if (!selectedFile) {
            alert('Please select a file first!');
            return;
        }

        if (fileCategory !== 'image') {
            alert('Only image files can be submitted at this time.');
            return;
        }

        setUploading(true);
        setUploadStatus('Uploading...');

        try {
            const result = await uploadFileToFirebase(selectedFile, fileCategory, fileType);
            console.log('Upload result:', result);
            setUploadStatus('Upload successful!');
            setTimeout(() => {
                setUploadStatus('');
            }, 3000);
            
        } catch (error) {
            console.error('Upload failed:', error);
            setUploadStatus('Upload failed. Please try again.');
            setTimeout(() => {
                setUploadStatus('');
            }, 3000);
        } finally {
            setUploading(false);
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
                    accept="image/*,.wav,audio/wav"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                />
                <button className="upload-main-button" onClick={triggerFileInput}>
                    <div className="plus-circle">
                        <span className="plus-icon">+</span>
                    </div>
                    <span className="upload-text">Add a file</span>
                </button>
                <button 
                    className="arrow-button" 
                    onClick={handleSubmit}
                    disabled={!selectedFile || uploading}
                    style={{ opacity: !selectedFile || uploading ? 0.5 : 1 }}
                >
                    <span className="arrow-icon">{uploading ? '⏳' : '→'}</span>
                </button>
            </div>
            
            {uploadStatus && (
                <div className="upload-status">{uploadStatus}</div>
            )}
        </div>
    );
};

export default Homepage;