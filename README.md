# InstruNet AI - CNN-Based Music Instrument Recognition System

Automated musical instrument recognition system using CNN and mel-spectrograms. Identifies 11 instrument types from audio with **80.45% accuracy**.

**Key Features:**
- 🎸 11 Instrument Classes (Bass, Brass, Flute, Guitar, Keyboard, Mallet, Organ, Reed, String, Synth Lead, Vocal)
- 📊 82.45% Validation Accuracy
- 🚀 Streamlit Web Interface
- 📄 JSON/PDF Reports
- 📦 Batch Processing

---

## 🎯 Problem Statement

Manual instrument identification is time-consuming, subjective, and not scalable. **InstruNet AI** automates this using deep learning (MobileNetV2) on mel-spectrograms from the NSynth dataset.

---

## 🚀 Quick Start

### Run Training
```bash
# Open milestonetasks.ipynb
# Run Cell 14 (main training code)
# Training time: ~15 minutes
```

### Launch Web App
```bash
streamlit run app.py
# Access at http://localhost:8501
```

---

## 🏗️ Architecture


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ rescaling (Rescaling)           │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_224            │ (None, 7, 7, 1280)     │     2,257,984 │
│ (Functional)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 1280)           │         5,120 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │       655,872 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 256)            │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 11)             │         2,827 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

**Model Details:**
- Base: MobileNetV2 (ImageNet pre-trained, frozen)
- Custom Head: Dense(512) → Dense(256) → Dense(11)
- Total Parameters: 3.05M (793K trainable)

---

## 📅 8-Week Milestones

| Week | Task | Status | Key Output |
|------|------|--------|------------|
| 1-2 | Data Collection & Preprocessing | ✅ | 2200 spectrograms (200/class) |
| 3-4 | CNN Model Development | ✅ | 82.45% accuracy|
| 5-6 | Evaluation & Tuning | ✅ | F1: 0.80, Confusion matrix |
| 7-8 | Deployment & Visualization | ✅ | Streamlit app, reports |

---

## 🔬 Algorithm

```python
1. Load audio (Librosa, sr=22050 Hz)
2. Generate mel-spectrogram (128 bins, 224×224 pixels)
3. Pass through MobileNetV2 CNN
4. Classify into 11 instrument classes
5. Return prediction + confidence + visualizations
```

**Training:** Adam optimizer (lr=0.002), 25 epochs, EarlyStopping, ReduceLROnPlateau

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **82.45%** |
| Training Accuracy | 94.2% |
| F1-Score | 0.80 |
| Best Classes | Vocal (100%), Organ (92%), Flute (91%) |
| Challenging | Guitar (57%), Bass (60%), Keyboard (60%) |
| Inference Time | ~1-2s/audio |
| Model Size | 10.2 MB |

---

## 📂 Project Structure

```
📁 Music Instrument Recognition System/
├── 📓 milestonetasks.ipynb    # Main training notebook (Cell 14)
├── 📓 app.py                   # Streamlit web app
├── 📁 models/                  # Trained models (.keras)
├── 📁 spectrograms/            # Training data (2200 images)
├── 📁 nsynth_small/audio/      # Audio dataset (11 classes)
└── 📁 reports/                 # Generated JSON/PDF reports
```

---

## 🛠️ Tech Stack

- **ML:** TensorFlow 2.20, MobileNetV2
- **Audio:** Librosa 0.11
- **Visualization:** Matplotlib, Plotly
- **UI:** Streamlit
- **Data:** NSynth (Google Magenta)

---

## 📖 Usage

### Training (Notebook)
```python
# Cell 14 in milestonetasks.ipynb
model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=[...])
# Result: 80.45% accuracy
```

### Inference (Python)
```python
from app import predict_instrument
result = predict_instrument("audio.wav", model, class_labels)
# Returns: {predicted_instrument, confidence, top3_predictions}
```

### Web Interface
Upload audio → View waveform/spectrogram → Get predictions → Download report

---

## ✅ Achievements

- ✅ 8-week milestone plan completed
- ✅ 80.45% validation accuracy
- ✅ Production-ready web app
- ✅ Comprehensive documentation
- ✅ Report generation (JSON/PDF)
- ✅ Batch processing capability

---

**Project:** InstruNet AI  
**Version:** 1.0  
**Status:** ✅ Complete (All 8-Week Milestones Achieved)  
**Accuracy:** 80.45% Validation  
**Date:** February 19, 2026