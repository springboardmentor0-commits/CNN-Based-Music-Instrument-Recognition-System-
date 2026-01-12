import os
import json
import shutil
from collections import defaultdict

BASE_DIR = "/content"
AUDIO_DIR = os.path.join(BASE_DIR, "nsynth-acoustic", "audio")
METADATA_PATH = os.path.join(BASE_DIR, "nsynth-train", "examples.json")
WORK_DIR = os.path.join(BASE_DIR, "Work")

FILES_PER_CLASS = 700

# Only valid acoustic instrument families
ALLOWED_CLASSES = {
    "guitar",
    "keyboard",
    "flute",
    "brass",
    "string",
    "reed",
    "mallet",
    "vocal"
}

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

wav_files = {
    fname.replace(".wav", ""): fname
    for fname in os.listdir(AUDIO_DIR)
    if fname.endswith(".wav")
}

print(f"Found {len(wav_files)} wav files")

os.makedirs(WORK_DIR, exist_ok=True)
counts = defaultdict(int)

for sample_id, info in metadata.items():
    label = info["instrument_family_str"]

    if label not in ALLOWED_CLASSES:
        continue

    if counts[label] >= FILES_PER_CLASS:
        continue

    if sample_id not in wav_files:
        continue

    src = os.path.join(AUDIO_DIR, wav_files[sample_id])
    dest_dir = os.path.join(WORK_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)

    shutil.copy(src, dest_dir)
    counts[label] += 1

print("\nFinal subset created:")
for k, v in counts.items():
    print(f"{k}: {v}")
