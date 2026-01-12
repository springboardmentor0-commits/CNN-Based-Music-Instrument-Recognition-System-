import os
import numpy as np
import librosa
import json
from tqdm import tqdm

# ================= PATHS =================
BASE_DIR = r"C:\Users\tyagi\Downloads\nsynth-train.jsonwav"
WORK_DIR = os.path.join(BASE_DIR, "Work")
OUT_DIR = os.path.join(BASE_DIR, "Processed")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= PARAMETERS =================
SR = 22050                 # Sampling rate
DURATION = 3.0             # seconds
SAMPLES = int(SR * DURATION)

N_MELS = 128
IMG_SIZE = 128

N_FFT = 2048
HOP_LENGTH = 512

# ================= LABELS =================
classes = sorted(
    d for d in os.listdir(WORK_DIR)
    if os.path.isdir(os.path.join(WORK_DIR, d))
)

label_map = {cls: idx for idx, cls in enumerate(classes)}

X = []
y = []

# ================= PROCESS LOOP =================
for cls in classes:
    class_dir = os.path.join(WORK_DIR, cls)

    for file in tqdm(os.listdir(class_dir), desc=f"Processing {cls}"):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(class_dir, file)

        # ---- Load audio ----
        audio, _ = librosa.load(path, sr=SR)

        # ---- Pad / trim ----
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        else:
            audio = audio[:SAMPLES]

        # ---- Mel spectrogram ----
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # ---- Convert to dB ----
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # ---- Fix time dimension ----
        mel_db = librosa.util.fix_length(
            mel_db, size=IMG_SIZE, axis=1
        )

        # ---- Normalize ----
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # ---- Add channel dimension (for CNNs) ----
        mel_db = np.expand_dims(mel_db, axis=0)  # (1, 128, 128)

        X.append(mel_db)
        y.append(label_map[cls])

# ================= SAVE =================
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\n✅ Preprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label map:", label_map)
