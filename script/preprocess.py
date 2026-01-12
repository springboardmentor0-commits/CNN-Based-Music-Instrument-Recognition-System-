import os
import numpy as np
import librosa
import json
from tqdm import tqdm

# ================= PATHS =================
BASE_DIR = "/content"
WORK_DIR = os.path.join(BASE_DIR, "Work")
OUT_DIR = os.path.join(BASE_DIR, "Processed")

os.makedirs(OUT_DIR, exist_ok=True)

# ================= PARAMETERS =================
SR = 22050
DURATION = 3.0
SAMPLES = int(SR * DURATION)
N_MELS = 128
IMG_SIZE = 128

# ================= LABELS =================
classes = sorted(os.listdir(WORK_DIR))
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

        # Load audio
        audio, _ = librosa.load(path, sr=SR)

        # Pad / trim to fixed length
        if len(audio) < SAMPLES:
            audio = np.pad(audio, (0, SAMPLES - len(audio)))
        else:
            audio = audio[:SAMPLES]

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_mels=N_MELS
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize to 128x128
        mel_db = librosa.util.fix_length(mel_db, size=IMG_SIZE, axis=1)

        X.append(mel_db)
        y.append(label_map[cls])

# ================= SAVE =================
X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "y.npy"), y)

with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=4)

print("\nPreprocessing complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)

# ================= SAVE TO GOOGLE DRIVE =================
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    
    drive_dest = '/content/drive/MyDrive/NSynth_Processed'
    os.makedirs(drive_dest, exist_ok=True)
    
    # Copy files to Drive
    shutil.copy(os.path.join(OUT_DIR, "X.npy"), drive_dest)
    shutil.copy(os.path.join(OUT_DIR, "y.npy"), drive_dest)
    shutil.copy(os.path.join(OUT_DIR, "label_map.json"), drive_dest)
    
    print(f"\n✓ Files saved to Google Drive: {drive_dest}")
except Exception as e:
    print(f"\n⚠ Could not save to Drive: {e}")
    print("Files are still available in /content/Processed")
