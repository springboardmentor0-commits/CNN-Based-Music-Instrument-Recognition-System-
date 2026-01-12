import json
import os
import shutil

# Paths (Colab)
NSYNTH_DIR = "/content/nsynth-train"
AUDIO_DIR = os.path.join(NSYNTH_DIR, "audio")
JSON_PATH = os.path.join(NSYNTH_DIR, "examples.json")

OUTPUT_BASE = "/content/nsynth-acoustic"
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_BASE, "audio")

# Create output folder
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# Load metadata
with open(JSON_PATH, "r") as f:
    data = json.load(f)

count = 0

for file_id, meta in data.items():
    # 0 = acoustic
    if meta["instrument_source"] == 0:
        src = os.path.join(AUDIO_DIR, file_id + ".wav")
        dst = os.path.join(OUTPUT_AUDIO_DIR, file_id + ".wav")

        if os.path.exists(src):
            shutil.copy(src, dst)
            count += 1

print("Total acoustic files copied:", count)
