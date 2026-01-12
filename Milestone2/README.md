# Milestone 2 – CNN Model Training

## Objective
Train a baseline Convolutional Neural Network (CNN) to classify musical instruments using mel-spectrogram features.

## Input
- Preprocessed spectrogram arrays generated during Milestone 1
- Acoustic instrument samples from the NSynth dataset

## Model Architecture
- Two Conv2D + MaxPooling blocks
- Dense layer with Dropout
- Softmax output layer

## Training Setup
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 20
- Hardware: Google Colab (T4 GPU)

## Results
- Training Accuracy: ~85%
- Validation Accuracy: ~78%

## Evaluation
- Accuracy and loss curves plotted
- Confusion matrix generated for class-wise performance analysis

## Notes
- Trained model saved in `.keras` format (not uploaded due to size)
