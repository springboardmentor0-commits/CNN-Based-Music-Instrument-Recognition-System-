
# 🎵 InstruNet AI  
### Multi-Instrument Audio Intelligence Platform  

---

## 🌐 Live Demo

🔗 **Live Application:**  
https://cnn-based-musical-instrument-detect.vercel.app/ 



---

## 🚩 Problem Statement

Modern audio analysis systems struggle with:

- Multi-instrument detection in mixed audio
- Lack of interpretability in AI predictions
- No structured analytics dashboards
- No exportable business reports
- Limited scalability for production systems

There is a need for a production-ready AI platform capable of:

- Multi-label instrument classification
- Visual analytics and explainability
- Exportable JSON & PDF reports
- Premium business workflows
- Scalable deep learning architecture

---

## 🎯 Purpose of the Project

InstruNet AI was built to create a deployable, real-world AI system that:

- Detects multiple musical instruments from raw audio
- Uses CNN-based deep learning on Mel-Spectrogram features
- Provides an interactive analytics dashboard
- Generates professional PDF and JSON reports
- Implements premium access logic for business scenarios

---

## 📌 Project Overview

InstruNet AI processes uploaded WAV files through the following pipeline:

1. Audio → Mel-Spectrogram transformation  
2. CNN-based multi-label prediction  
3. Probability aggregation  
4. Dashboard visualization  
5. Report generation  

The deployed demo includes:

- 🔐 Sign-in / Login system
- 🎧 Audio upload & processing
- 🧠 CNN confidence scoring
- 📊 Interactive analytics dashboard
- 📄 Downloadable PDF reports
- 📦 JSON export
- ⭐ Premium watermark-free mode

---

## 🧠 Tech Stack

### AI & Backend
- Python
- TensorFlow / Keras
- Librosa

### Visualization
- Matplotlib
- Mel-Spectrogram analysis
- Timeline probability graphs
- Pie charts
- Frequency band distribution

### Frontend & Deployment
- Python
- Streamlit

### Reporting
- JSON export
- Professional PDF generation

---

## ✨ Core Features

### 🔐 Authentication
- Secure login / sign-in
- Premium feature toggle
- Watermark removal for premium users

### 🎧 Audio Intelligence
- WAV file upload
- Multi-label instrument detection
- CNN confidence scoring

### 📊 Analytics Dashboard
- Instrument probability timeline
- Audio waveform visualization
- Mel-Spectrogram display
- Frequency band distribution analysis
- Interactive pie chart
- Prediction analysis summary
- Dominant instrument highlight

### 📄 Reporting System
- JSON export
- Professional PDF download
- Premium watermark-free reporting

---

## 🎼 Supported Instruments

- 🎶 Flute  
- 🎸 Guitar  
- 🎹 Piano  
- 🎻 Violin  

*(Scalable architecture for additional instruments)*

---

## 🧾 Model Card

| Attribute | Details |
|------------|----------|
| Model Type | Convolutional Neural Network (CNN) |
| Input | 128 × 128 Mel-Spectrogram |
| Output | Multi-label probabilities |
| Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Evaluation Accuracy | ~85% |
| Training Data | Custom labeled WAV dataset |

---

## 🏗 CNN Architecture

```text
Input (128x128x1)
↓
Conv2D (32 filters) + ReLU
↓
MaxPooling
↓
Conv2D (64 filters) + ReLU
↓
MaxPooling
↓
Conv2D (128 filters) + ReLU
↓
Flatten
↓
Dense (128) + ReLU
↓
Dropout
↓
Dense (4) + Sigmoid
```

**Why Sigmoid?**  
Because this is a multi-label classification problem where multiple instruments can exist simultaneously.

---

## 📊 Dashboard Features

- Instrument Probability Timeline  
- CNN Confidence Indicator  
- Audio Waveform Visualization  
- Mel-Spectrogram Representation  
- Frequency Band Energy Distribution  
- Interactive Pie Chart  
- Dominant Instrument Highlight  
- Premium Export Controls  

---

## 📦 Example JSON Output

```json
{
  "report": "InstruNet AI Analysis Report",
  "timestamp": "2/22/2026, 11:28:54 AM",
  "instrument": "Guitar",
  "confidence": 90.31613293826148,
  "health": "Healthy",
  "condition": "Excellent resonance with clear harmonic profile.",
  "intensity": 85.42309722468103,
  "developer": "Sai Nikith"
}
```

---

## 💼 Business Use Cases

- 🎼 Music production studios  
- 🎵 Audio catalog tagging systems  
- 📊 Streaming platform analytics  
- 🎓 AI-powered music learning platforms  
- 🔍 Audio forensics  
- 🤖 Smart content moderation systems  

---

## 🚀 Installation (Local Development)

```bash
git clone https://github.com/CNN-Based-Music-Instrument-Recognition-System.git
cd CNN-Based-Music-Instrument-Recognition-System
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛣 Roadmap

- Add 10+ instrument classes
- Expand dataset diversity
- Real-time microphone detection
- Mobile app integration

---

## 🤝 Contribution

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 👨‍💻 Author

**Sai Nikith**  
AI Engineer | Audio Intelligence Developer  

- 🔗 GitHub: [sainikith07](https://github.com/sainikith07)  
- 🔗 LinkedIn: [Sai Nikith Kaleru](https://www.linkedin.com/in/sai-nikith-kaleru/)  
- 📧 Email: sainikith04@gmail.com   

---

## 💬 Support

For collaborations, business inquiries, or improvements:

Connect via LinkedIn.

---

## ▶ How To Run

1. Login to the platform  
2. Upload WAV audio file  
3. View detection results  
4. Analyze dashboard insights  
5. Download JSON or PDF report  
6. Upgrade to premium for watermark-free export  

---

## ⭐ Project Highlights

✔ Multi-label instrument detection  
✔ Real-time probability visualization  
✔ Professional reporting system  
✔ Premium business logic integration  
✔ Scalable CNN architecture  
✔ Production-ready UI  

---

> InstruNet AI demonstrates how deep learning can be transformed from a research prototype into a business-ready intelligent audio analytics platform.

