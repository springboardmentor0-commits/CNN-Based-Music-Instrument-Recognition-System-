# 🎵 InstruNet AI – CNN-based Musical Instrument Recognition  
### Milestone 2 & Milestone 3 Documentation

---

## 📌 Project Overview

**InstruNet AI** is a deep learning system designed to identify musical instruments present in an audio track by converting audio signals into **mel-spectrogram representations** and using **Convolutional Neural Networks (CNNs)** for classification.

The project focuses on **polyphonic instrument recognition** and aims to robustly classify instruments that share similar acoustic characteristics (e.g., guitar vs mallet, reed vs brass).

---

## 🎯 Objectives

### **Milestone 2 – CNN Model Development**
- Design a baseline CNN architecture for instrument classification  
- Train the model using mel-spectrogram features  
- Achieve a stable baseline accuracy  

### **Milestone 3 – Model Evaluation & Tuning**
- Analyze model weaknesses using validation metrics and confusion matrices  
- Improve classification performance through architectural and training optimizations  
- Reduce inter-class confusion  
- Establish an optimized CNN baseline for future extensions  

---

## 📂 Dataset Description

- **Dataset:** NSynth (Acoustic subset)  
- **Preprocessing Steps:**
  - Audio resampled to **22,050 Hz**
  - Fixed duration: **3 seconds**
  - Converted to **128×128 mel-spectrograms**
- **Instrument Classes Used (8):**
  - Brass  
  - Flute  
  - Guitar  
  - Keyboard  
  - Mallet  
  - Reed  
  - String  
  - Vocal  

> Bass, Organ, and Synth Lead were excluded due to insufficient samples and class imbalance.

---

## 🧠 Milestone 2 – Baseline CNN Model

### 🔹 CNN Architecture

Input (128 × 128 × 1)
↓
Conv2D (32 filters) + ReLU
↓
Batch Normalization
↓
MaxPooling
↓
Conv2D (64 filters) + ReLU
↓
Batch Normalization
↓
MaxPooling
↓
Conv2D (128 filters) + ReLU
↓
Batch Normalization
↓
MaxPooling
↓
Global Average Pooling
↓
Dense (128) + ReLU
↓
Dropout (0.3)
↓
Dense (8 classes) + Softmax


### 🔹 Training Configuration
- Optimizer: **Adam**
- Learning Rate: **0.001**
- Loss Function: **Sparse Categorical Cross-Entropy**
- Batch Size: **32**
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint

### 🔹 Baseline Performance
- **Validation Accuracy:** ~82–83%
- **Test Accuracy:** ~82–83%

### 🔹 Observed Issues
- Guitar ↔ Mallet confusion  
- Reed ↔ Brass confusion  
- Bias toward dominant instrument classes  

---

## 🔍 Post-Milestone 2 Analysis

### Key Challenges Identified
1. **Class Imbalance**
   - Uneven distribution of instrument samples
   - Bias toward frequently occurring instruments  

2. **Limited Feature Hierarchy**
   - Shallow representations insufficient for subtle timbral differences  

3. **Early Overfitting**
   - Training accuracy increased faster than validation accuracy  

---

## ⚙️ Milestone 3 – Model Optimization & Tuning

### ✅ 1. Deeper CNN Architecture
- Added additional convolutional depth
- Enables learning of higher-level spectral and timbral patterns
- Improves discrimination between similar instruments

---

### ✅ 2. Learning Rate Optimization
- Reduced learning rate for slower and more stable convergence
- Helps prevent overshooting optimal minima
- Improves validation generalization

---

### ✅ 3. Class Weighting
- Applied class weights during training
- Penalizes misclassification of minority classes
- Improves recall for underrepresented instruments

> Note: Higher loss values are expected when class weighting is applied, as the loss function becomes stricter.

---

### ❌ Attention-based CNN (Experimental)
- Attention mechanism tested after CNN blocks
- Did not provide consistent improvement
- Likely due to dataset size and increased model complexity
- Excluded from final optimized architecture

---

## 📊 Performance Comparison

| Metric | Milestone 2 | Milestone 3 |
|------|------------|------------|
| Validation Accuracy | ~82–83% | **~85%** |
| Test Accuracy | 85.23% | **87.1%** |
| Minority Class Recall | Lower | **Improved** |
| Class Confusion | Higher | **Reduced** |
| Generalization | Moderate | **Better** |

---

## 📈 Key Observations

- Overall accuracy improved without severe overfitting  
- Minority instruments showed improved recall  
- Inter-class confusion reduced  
- Validation loss increased due to class weighting (expected behavior)  

---

## 🏁 Conclusion

Milestone 3 successfully strengthened the baseline CNN model by focusing on **generalization, robustness, and fairness**, rather than optimizing accuracy alone.

The optimized CNN model provides a strong foundation for advanced extensions such as:
- CRNN-based temporal modeling  
- SpecAugment data augmentation  
- Instrument intensity estimation over time  
- Real-world polyphonic music analysis  

---

## 🚀 Next Steps
- Fine-grained hyperparameter tuning  
- Temporal modeling for instrument activity detection  
- Extension to multi-instrument intensity tracking  

---
