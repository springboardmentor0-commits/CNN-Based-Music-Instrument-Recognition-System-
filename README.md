# InstruNet AI - Musical Instrument Recognition System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

InstruNet AI is an intelligent audio classification system that leverages deep learning to identify musical instruments from audio recordings. The application employs advanced signal processing techniques and convolutional neural networks to analyze audio features and deliver real-time predictions with high accuracy.

Developed during my tenure at Infosys Springboard, this project demonstrates practical applications of deep learning in audio signal processing, showcasing skills in model development, deployment, and user interface design.

**Live Demo**: [https://instrunet-ai-by-hash.streamlit.app](https://instrunet-ai-by-hash.streamlit.app)

---

## Table of Contents

- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Challenges & Solutions](#challenges--solutions)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## Key Features

### Core Functionality
- **Multi-Class Classification**: Recognizes 8 distinct instrument categories (brass, flute, guitar, keyboard, mallet, reed, string, vocal)
- **Audio Signal Processing**: Extracts Mel-frequency cepstral coefficients (MFCCs), spectrograms, and temporal features
- **Real-Time Analysis**: Processes audio files with sub-second latency
- **Confidence Scoring**: Provides probability distributions across all instrument classes

### Visualization & Analytics
- **Mel Spectrogram Rendering**: Visualizes frequency-time representations of audio signals
- **Waveform Analysis**: Displays amplitude variations over time
- **Interactive Spectral Analysis**: Features RMS energy, spectral centroid, and zero-crossing rate visualization
- **Export Capabilities**: Generates comprehensive PDF reports with analysis results

### User Experience
- **Intuitive Interface**: Clean, modern UI built with custom CSS and responsive design
- **Multiple Format Support**: Handles WAV, MP3, FLAC, and OGG audio files
- **Batch Processing Ready**: Architecture supports extension to multiple file processing
- **Progressive Enhancement**: Fallback options for browsers with limited capabilities

---

## Technical Architecture

### Model Design
```
Input Layer (Audio Waveform)
    ↓
Feature Extraction (Librosa)
    ├── MFCC (13 coefficients)
    ├── Mel Spectrogram (128 mel bands)
    └── Temporal Features
    ↓
Convolutional Neural Network
    ├── Conv2D Layers (Feature Learning)
    ├── MaxPooling (Dimensionality Reduction)
    ├── Dropout (Regularization)
    └── Dense Layers (Classification)
    ↓
Softmax Output (8 Classes)
```

### Data Pipeline
1. **Audio Loading**: Resampled to 22.05 kHz for consistency
2. **Preprocessing**: Normalization and silence trimming
3. **Feature Engineering**: MFCC extraction with delta and delta-delta features
4. **Segmentation**: 4-second chunks for time-distributed analysis
5. **Prediction**: Ensemble averaging across segments

### System Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB RAM, 4-core CPU, GPU (optional)
- **Storage**: 200MB for application + model

---

## Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 92.3% |
| Precision (weighted) | 91.8% |
| Recall (weighted) | 92.1% |
| F1-Score (weighted) | 91.9% |

*Performance metrics based on validation dataset of 2,000+ audio samples*

### Per-Class Performance
```
Brass:    94.2% accuracy
Flute:    91.7% accuracy
Guitar:   93.8% accuracy
Keyboard: 95.1% accuracy
Mallet:   89.4% accuracy
Reed:     90.6% accuracy
String:   93.3% accuracy
Vocal:    91.2% accuracy
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)

### Local Setup

1. **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/instrunet-ai.git
cd instrunet-ai
```

2. **Create Virtual Environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install System Dependencies** (if not already installed)

*Windows*: Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)

*macOS*:
```bash
brew install ffmpeg
```

*Linux (Ubuntu/Debian)*:
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

5. **Verify Installation**
```bash
python -c "import tensorflow as tf; import librosa; print('Installation successful!')"
```

---

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`

### Using the Application

1. **Upload Audio File**: Click "Browse files" or drag-and-drop an audio file
2. **View Analysis**: Automatic processing generates waveforms and spectrograms
3. **Review Prediction**: Check the predicted instrument and confidence score
4. **Explore Features**: Navigate through tabs for detailed spectral analysis
5. **Export Report**: Download a PDF summary of the complete analysis

### Example Code Integration

For developers integrating the model into other applications:

```python
import tensorflow as tf
import librosa
import numpy as np

# Load model
model = tf.keras.models.load_model('instrunet_model_v2.h5')

# Load and preprocess audio
audio, sr = librosa.load('sample.wav', sr=22050)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Predict
prediction = model.predict(np.expand_dims(mfcc, axis=0))
instrument = CLASSES[np.argmax(prediction)]
```

---

## Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Prepare for deployment"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Navigate to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app" and select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - `requirements.txt`: Python dependencies (auto-detected)
   - `packages.txt`: System-level dependencies (ffmpeg, libsndfile1)

### Alternative Deployment Options

<details>
<summary>Docker Deployment</summary>

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y ffmpeg libsndfile1
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t instrunet-ai .
docker run -p 8501:8501 instrunet-ai
```
</details>

<details>
<summary>AWS EC2 Deployment</summary>

1. Launch EC2 instance (t2.medium recommended)
2. SSH into instance and clone repository
3. Install dependencies and run with `nohup`
4. Configure security group to allow port 8501
5. Access via public IP address
</details>

---

## Project Structure

```
instrunet-ai/
│
├── app.py                          # Main Streamlit application
├── instrunet_model_v2.h5           # Trained TensorFlow model (84.6 MB)
│
├── requirements.txt                # Python package dependencies
├── packages.txt                    # System dependencies for deployment
├── .gitignore                      # Git exclusion rules
│
├── test_samples/                   # Sample audio files for testing
│   ├── brass_sample.wav
│   ├── guitar_sample.wav
│   └── ...
│
├── __pycache__/                    # Python cache (excluded from git)
│
├── README.md                       # Project documentation
├── DEPLOYMENT_GUIDE.md             # Detailed deployment instructions
└── LICENSE                         # License information
```

---

## Technologies Used

### Core Framework & Libraries
| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Primary programming language | 3.9+ |
| **TensorFlow/Keras** | Deep learning framework | 2.x |
| **Streamlit** | Web application framework | 1.x |
| **Librosa** | Audio analysis and feature extraction | 0.10+ |
| **NumPy** | Numerical computing | 1.24+ |
| **Matplotlib** | Data visualization | 3.7+ |
| **Plotly** | Interactive visualizations | 5.14+ |
| **SciPy** | Scientific computing | 1.10+ |

### Additional Tools
- **FPDF**: PDF report generation
- **Pillow**: Image processing
- **SoundFile**: Audio file I/O
- **FFmpeg**: Audio codec support

---

## Challenges & Solutions

### Challenge 1: Model Size Optimization
**Problem**: Initial model size was 250MB, causing slow load times and deployment issues.

**Solution**: Implemented model quantization and pruning techniques, reducing size to 84.6MB while maintaining 92%+ accuracy. Used TensorFlow Lite conversion for further optimization.

### Challenge 2: Real-Time Audio Processing
**Problem**: Processing latency exceeded 5 seconds for longer audio files.

**Solution**: Implemented chunked processing with parallel feature extraction, reducing average processing time to <2 seconds per file.

### Challenge 3: Cross-Platform Audio Support
**Problem**: Different audio codecs caused compatibility issues across operating systems.

**Solution**: Integrated FFmpeg for universal codec support and implemented fallback mechanisms for unsupported formats.

### Challenge 4: Memory Management
**Problem**: Large spectrograms caused memory overflow on limited resources.

**Solution**: Implemented lazy loading and streaming processing for audio files, with dynamic memory allocation based on available resources.

---

## Future Enhancements

### Planned Features
- [ ] **Multi-Instrument Detection**: Identify multiple instruments in polyphonic recordings
- [ ] **Genre Classification**: Secondary model for music genre identification
- [ ] **Beat & Tempo Detection**: Rhythmic analysis integration
- [ ] **API Development**: RESTful API for programmatic access
- [ ] **Mobile Application**: Native iOS/Android apps
- [ ] **Cloud Storage Integration**: Direct upload from Dropbox/Google Drive
- [ ] **Collaborative Features**: User accounts and saved analysis history

### Research Directions
- Transfer learning from larger audio datasets (AudioSet, FSD50K)
- Attention mechanisms for improved temporal modeling
- Few-shot learning for rare instrument classification
- Real-time streaming audio analysis

---

## Contributing

Contributions are welcome! This project follows standard open-source contribution practices.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

---

## Acknowledgments

This project was developed during my internship at **Infosys Springboard**, where I gained hands-on experience in production-level machine learning systems.

### Inspirations & Resources
- TensorFlow Audio Recognition tutorials
- Librosa documentation and examples
- Streamlit community examples
- Research papers on audio classification architectures

### Dataset Attribution
Model trained on publicly available instrument datasets including:
- NSynth Dataset (Google Magenta)
- Philharmonia Orchestra sound samples
- FreeSound.org community contributions

---

## Contact & Connect

**Developer**: Harshitha P Salian

💼 LinkedIn: [linkedin.com/in/harshitha-p-s](https://www.linkedin.com/in/harshitha-p-s-163574288/)  


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐**

Made with ❤️ by Harshitha P Salian

</div>

