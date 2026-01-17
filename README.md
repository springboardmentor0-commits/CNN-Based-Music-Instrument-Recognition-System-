# CNN-Based Musical Instrument Recognition System

## Overview
This project aims to build a Convolutional Neural Network (CNN)–based system for automatically recognizing musical instruments from audio signals.  
Audio samples are transformed into **log-mel spectrograms**, which are treated as image-like inputs for deep learning models.

The project is being developed incrementally following a milestone-based approach, starting with data preprocessing and validation.

---

## Dataset
**NSynth Dataset (Acoustic Subset)** – Google Magenta

- High-quality, monophonic instrument recordings
- Uniform pitch and velocity conditions
- Audio signals converted into fixed-size log-mel spectrograms
- **8 acoustic instrument classes** used for classification

Due to storage constraints, the full dataset and processed NumPy files are **not included** in this repository.

---

## Current Project Structure

CNN_music_instrument/
│
├── data/
│ ├── sample_spectograms/ # Sample mel-spectrograms for visual inspection
│
├── notebooks/
│ └── cnn_music_instrument.ipynb # Data preprocessing notebook
│
├── scripts/
│ ├── preprocessing.py # Audio preprocessing pipeline
│ ├── acoustic_filter.py # Instrument filtering logic
│ └── balance_train_acoustic.py # Dataset balancing utilities
│
├── metadata/
│ └── class_mapping.json # Instrument label mapping
│
├── .gitignore # Dataset and cache exclusions
└── README.md


---

## Milestone Summary

### ✅ Milestone 1 – Data Preprocessing (Completed)

- Audio standardization (sample rate normalization, mono conversion)
- Fixed-duration audio segmentation
- Log-mel spectrogram extraction
- Dataset stored in NumPy format for efficient training
- Class-to-index mapping generation
- **Visual sanity checks using representative spectrogram samples**

Sample mel-spectrogram images are provided in the `data/sample_spectograms/` directory for verification.

---

## Notes
- Full `.npy` datasets and raw audio files are excluded using `.gitignore`
- The preprocessing pipeline is fully reproducible using the provided scripts and notebook
- Subsequent milestones will focus on CNN model training, evaluation, and deployment

---

## Current Status
✔ Data preprocessing completed  
✔ Dataset validated through visual inspection  
⏳ CNN model training to be implemented in Milestone 2
