# Synthetic Audio Dataset Generation

This repository contains a Python-based system for generating a **synthetic audio dataset** that simulates multiple musical instruments using mathematical signal synthesis.  
The dataset is designed for **machine learning, deep learning, and digital signal processing experiments**, especially in audio classification and feature extraction tasks.

---

## 🎯 Project Objective

The goal of this project is to create a **clean, controlled audio dataset** where different instrument classes are generated using predefined frequency and harmonic structures.  
Since the data is synthetic, it removes real-world noise and recording inconsistencies, making it ideal for **prototyping and academic experimentation**.

---

## 🎵 Instrument Classes

The dataset simulates three musical instruments, each with a distinct waveform pattern:

### 🎶 Flute
- Generated using a pure sine wave
- Frequency: **1000 Hz**
- Represents smooth and clean tonal behavior

### 🎻 Violin
- Generated using multiple harmonics
- Base frequency: **440 Hz**
- Includes **5 harmonics** to simulate rich string vibrations

### 🎺 Trumpet
- Generated using strong harmonic overtones
- Base frequency: **600 Hz**
- Includes **3 harmonics** for a bright sound profile

---

## 📁 Dataset Structure

```
synthetic_audio/
├── flute/
│   ├── flute_0.wav
│   ├── flute_1.wav
│   └── ...
├── violin/
│   ├── violin_0.wav
│   ├── violin_1.wav
│   └── ...
└── trumpet/
    ├── trumpet_0.wav
    ├── trumpet_1.wav
    └── ...
```


- Each instrument folder contains **100 audio samples**
- File format: **.wav**

---

## ⚙️ Audio Configuration

| Parameter        | Value        |
|------------------|-------------|
| Sampling Rate    | 22050 Hz    |
| Duration         | 3 seconds   |
| Channels         | Mono        |
| Samples/Class    | 100         |
| Total Samples    | 300         |

---

## 🧠 How It Works (Workflow Overview)

1. Define time duration and sampling rate
2. Generate waveform using sine functions
3. Apply harmonic summation for complex instruments
4. Save generated signals as `.wav` files
5. Organize output into class-wise directories

---

## 🚀 Applications & Use Cases

- Audio classification models
- CNN-based sound recognition
- Mel-spectrogram and MFCC extraction
- DSP and signal processing learning
- Academic demonstrations and internships
- Interview-ready project showcase

---

## ▶️ How to Generate the Dataset

1. Ensure Python is installed
2. Install required libraries (`numpy`, `soundfile`)
3. Run the `datasetGeneration.py` script
4. The dataset will be created inside the `synthetic_audio/` directory

---

## 📝 Key Highlights

- Fully **synthetic dataset** (no real recordings)
- Consistent audio length and format
- Clean signal generation using mathematics
- Ideal for controlled ML experiments
- Lightweight and easy to reproduce

---

## 🔮 Future Enhancements

- Add background noise variation
- Randomize pitch and amplitude
- Support additional instruments
- Generate mel-spectrogram datasets
- Integrate directly with CNN training pipelines

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 🙌 Author

Developed as part of an academic / internship-based AI and audio processing project.
