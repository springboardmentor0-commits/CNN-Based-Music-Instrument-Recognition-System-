# 🎵 InstruNet AI  
## CNN-Based Music Instrument Recognition System

---

## 📌 Project Overview

InstruNet AI is a deep learning-based system designed to automatically detect musical instruments in an audio track using Convolutional Neural Networks (CNNs) applied to Mel-Spectrogram representations.

This project implements a complete end-to-end machine learning pipeline including preprocessing, CNN training, evaluation, visualization, and automated report generation (JSON & PDF).

---

## 🎯 Objectives

- Convert audio signals into Mel-Spectrogram images
- Train a CNN model for instrument classification
- Perform confidence-based prediction
- Generate segment-wise instrument activity timeline
- Export structured JSON & PDF analysis reports

---

## 🛠️ Technology Stack

- Python 3
- Librosa (Audio Processing)
- TensorFlow / Keras (CNN Model)
- NumPy
- Matplotlib & Seaborn (Visualization)
- Scikit-learn (Evaluation Metrics)
- ReportLab (PDF Generation)
- JSON (Export Reports)

---

# 📅 Milestone-Wise Development

---

# 🚀 Milestone 1 (Week 1–2): Data Collection & Preprocessing

### Dataset Preparation
- Loaded acoustic dataset (`X_raw_subset.npy`, `y_labels_subset.npy`)
- Total samples: 6813

### Spectrogram Generation
- Converted audio signals into 128 Mel-band spectrograms
- Converted to log scale (dB)
- Cropped to 128 × 128
- Normalized between 0 and 1

Final CNN Input Shape:
```
(6706, 128, 128, 1)
```

### Label Encoding
- One-hot encoding applied
- Instrument families:
  - Bass
  - Brass
  - Flute
  - Guitar
  - Keyboard
  - Mallet
  - Organ
  - Reed
  - String
  - Synth
  - Vocal

✅ Milestone Result: Dataset prepared in image format for CNN training.

---

# 🧠 Milestone 2 (Week 3–4): CNN Model Development

### Model Architecture

- Conv2D (32 filters)
- MaxPooling
- Conv2D (64 filters)
- MaxPooling
- Conv2D (128 filters)
- MaxPooling
- Flatten
- Dense (128 neurons)
- Dropout (0.5)
- Output Layer (Sigmoid activation)

Total Parameters: 3.3 Million

### Training Configuration

- Optimizer: Adam
- Loss: Binary Crossentropy
- Batch Size: 32
- EarlyStopping enabled
- Train/Validation Split: 80/20

✅ Milestone Result: Successfully trained baseline CNN model.

---

# 📊 Milestone 3 (Week 5–6): Model Evaluation & Optimization

### Final Validation Accuracy
```
86.8%
```

### Performance Metrics

- Precision ≈ 0.85
- Recall ≈ 0.85
- F1-Score ≈ 0.84
- Accuracy ≈ 0.85

### Evaluation Tools

- Classification Report
- Confusion Matrix Heatmap
- Training Accuracy & Loss Graphs

Improvements Applied:
- Dropout regularization
- Early stopping
- Validation monitoring

✅ Milestone Result: Optimized CNN with strong generalization performance.

---

# 🌐 Milestone 4 (Week 7–8): Deployment & Visualization

## Real-Time Audio Prediction

- Load audio file
- Convert to Mel-Spectrogram
- Normalize
- Predict instrument probabilities
- Apply confidence threshold

Example Output:
```
brass : 100.0%
```

---

## Visualization Features

### Waveform Plot  
Amplitude vs Time visualization

### Mel Spectrogram  
Time-frequency representation

### Confidence Bar Chart  
Instrument probability levels

### Segment-wise Timeline Analysis  
Audio divided into 1-second segments  
Heatmap showing instrument activity over time

Example Timeline Shape:
```
(4, 8)
```

---

# 📁 Report Generation

## JSON Report

Generated file:
```
instrument_analysis_report.json
```

Includes:
- Audio file name
- Analysis timestamp
- Overall prediction
- Segment-wise instrument timeline
- Model performance summary

---

## PDF Report

Generated file:
```
instrument_analysis_report.pdf
```

Includes:
- Structured instrument analysis
- Timeline table
- Confidence summary

---

# 📷 Screenshots

### 🎼 Mel Spectrogram
![Mel Spectrogram](output_0_1.png)

### 📊 Training Accuracy & Loss
![Training Performance](output_10_1.png)

### 🔥 Confusion Matrix
![Confusion Matrix](output_16_0.png)

### ⏱️ Segment-wise Timeline
![Timeline Analysis](output_21_0.png)

---

# 📂 Project Structure

```
├── milestone_2_and_3.ipynb
├── output_*.png
├── instrunet_final.keras
├── instrument_analysis_report.json
├── instrument_analysis_report.pdf
└── README.md
```

---

# 🔧 Installation

```
pip install numpy matplotlib librosa tensorflow scikit-learn seaborn reportlab
```

---

# ▶️ Usage

### Train Model
```
python model_training.py
```

### Run Prediction
```
python inference.py
```

---

# 🧠 Learning Outcomes

- Applied CNNs to audio spectrogram data
- Implemented multi-class instrument classification
- Built full ML pipeline from preprocessing to deployment
- Generated automated JSON & PDF reports
- Developed visualization-driven model analysis

---

# 🚀 Project Status

✅ End-to-End Functional  
✅ CNN Model Trained & Optimized  
✅ Real-Time Audio Prediction  
✅ Visualization Integrated  
✅ JSON & PDF Report Generation  
✅ Milestone-Wise Completed  

---

# 👨‍💻 Author

Sai Deva Harsha

---

⭐ If you found this project interesting, feel free to star the repository!
