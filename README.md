# 🎵 InstruNet AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow.svg)

**Music Instrument Recognition & Audio Quality Assessment**

InstruNet AI is a deep learning web application that analyzes audio files (`.wav` or `.mp3`) to predict the musical instruments playing and assess the overall acoustic health of the recording. 

### 🚀 **[Try the Live Web App Here!](https://prasi001-instrunet-ai.hf.space)**

---

## 🌟 Key Features
* **Instrument Detection:** Uses a trained Convolutional Neural Network (CNN) to predict instruments from 10 distinct classes (Guitar, Brass, Strings, Vocal, etc.).
* **Quality Assessment:** Analyzes spectral flatness and harmonic-to-noise ratio (HNR) to grade the audio condition as `HEALTHY`, `AGED`, or `BROKEN`.
* **Visual Diagnostics:** Generates real-time interactive Mel-spectrograms and waveform plots using `librosa`.
* **Exportable Reports:** Users can instantly download their analysis results as a JSON file or a formatted PDF.

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow / Keras (CNN trained on the NSynth Dataset)
* **Audio Processing:** Librosa
* **Frontend UI:** Streamlit
* **Deployment:** Hugging Face Spaces

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/springboardmentor0-commits/CNN-Based-Music-Instrument-Recognition-System-.git](https://github.com/springboardmentor0-commits/CNN-Based-Music-Instrument-Recognition-System-.git)
2.Install the required dependencies:
   pip install -r requirements.txt

3.Run the streamlit app:
   streamlit run app.py
