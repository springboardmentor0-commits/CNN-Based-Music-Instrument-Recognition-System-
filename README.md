# InstruMusic – Instrument Classification (Milestone 1)

This project builds a machine learning pipeline for classifying musical instruments
using Mel-spectrogram representations of audio data.

## Milestone 1 – Data Preparation & Preprocessing

### Dataset
- NSynth Acoustic subset
- 8 acoustic instrument classes
- 700 samples per class (balanced)

### Pipeline
1. Sub-sampled and curated a balanced acoustic dataset locally
2. Converted audio to fixed-length Mel-spectrograms (128×128)
3. Stored features and labels as NumPy arrays
4. Verified preprocessing in Google Colab
5. Visualized class balance and sample spectrograms

### Classes Used
- Brass
- Flute
- Guitar
- Keyboard
- Mallet
- Reed
- String
- Vocal

### Repository Structure
- `scripts/` – preprocessing and dataset curation scripts
- `notebooks/` – Colab notebook for verification and visualization
- `samples/` – example Mel-spectrogram images

Milestone 1 completed: Data preprocessing and verification done.


> Note: Large datasets and NumPy files are excluded from this repository.
