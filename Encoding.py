# Audio preprocessing and feature extraction
# Converts each audio file into a log-scaled Mel-spectrogram.
# Ensures fixed input dimensions for CNN training.

import librosa
import numpy as np
import os
from skimage.transform import resize
from tqdm import tqdm

DATA_DIR = "synthetic_audio"
SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
IMG_SIZE = (128, 128)

X = []
y = []

labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

print("Label map:", label_map)

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in tqdm(os.listdir(folder), desc=label):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file)

        audio, _ = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = resize(mel_db, IMG_SIZE)

        X.append(mel_db)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
