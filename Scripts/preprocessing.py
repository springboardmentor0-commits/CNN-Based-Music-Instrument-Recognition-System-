import os
import json
import numpy as np
import librosa
import cv2
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR = "filtered_nsynth"
OUTPUT_DIR = "npy_dataset"

SPLITS = {
    "train": "train_acoustic_balanced",
    "valid": "valid_acoustic",
    "test":  "test_acoustic"
}

SR = 22050
DURATION = 3.0
N_MELS = 128
IMG_SIZE = 128

# 🔒 Fixed class mapping
CLASS_MAPPING = {
    "brass": 0,
    "flute": 1,
    "guitar": 2,
    "keyboard": 3,
    "mallet": 4,
    "reed": 5,
    "string": 6,
    "vocal": 7
}
# ==========================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def audio_to_mel(wav_path):
    # Load audio
    y, sr = librosa.load(wav_path, sr=SR, duration=DURATION, mono=True)

    # Pad if shorter
    expected_len = int(SR * DURATION)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )

    # Log scale
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to 128x128
    mel_resized = cv2.resize(mel_db, (IMG_SIZE, IMG_SIZE))

    return mel_resized


def process_split(split_name, folder_name):
    print(f"\n🔄 Processing {split_name.upper()} split")

    split_dir = os.path.join(BASE_DIR, folder_name)
    audio_dir = os.path.join(split_dir, "audio")
    labels_path = os.path.join(split_dir, "labels.json")

    # 🔹 output subfolder per split
    out_split_dir = os.path.join(OUTPUT_DIR, split_name)
    ensure_dir(out_split_dir)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    X = []
    y = []

    for sample_id, info in tqdm(labels.items()):
        class_name = info["instrument_family_str"]

        if class_name not in CLASS_MAPPING:
            continue

        wav_path = os.path.join(audio_dir, sample_id + ".wav")
        if not os.path.exists(wav_path):
            continue

        mel_img = audio_to_mel(wav_path)

        X.append(mel_img)
        y.append(CLASS_MAPPING[class_name])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Save npy files inside split folder
    np.save(os.path.join(out_split_dir, "X.npy"), X)
    np.save(os.path.join(out_split_dir, "y.npy"), y)

    # Save full metadata for traceability
    with open(os.path.join(out_split_dir, "labels_full.json"), "w") as f:
        json.dump(labels, f, indent=2)

    print(f"✅ {split_name} saved:")
    print(f"   X.shape = {X.shape}")
    print(f"   y.shape = {y.shape}")


def main():
    ensure_dir(OUTPUT_DIR)

    # Save class mapping once (global)
    with open(os.path.join(OUTPUT_DIR, "class_mapping.json"), "w") as f:
        json.dump(CLASS_MAPPING, f, indent=2)

    for split, folder in SPLITS.items():
        process_split(split, folder)

    print("\n🎉 All splits processed successfully.")
    print("📦 Dataset is now CNN-ready.")


if __name__ == "__main__":
    main()
