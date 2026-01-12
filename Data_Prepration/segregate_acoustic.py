import json
import os
import shutil

# Paths
NSYNTH_DIR = "nsynth-train"
AUDIO_DIR = os.path.join(NSYNTH_DIR, "audio")
JSON_PATH = os.path.join(NSYNTH_DIR, "examples.json")

OUTPUT_BASE = "nsynth-acoustic"

# Instrument family mapping
INSTRUMENT_FAMILIES = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string"
}

# Create output folders
for name in INSTRUMENT_FAMILIES.values():
    os.makedirs(os.path.join(OUTPUT_BASE, name), exist_ok=True)

# Load metadata
with open(JSON_PATH, "r") as f:
    data = json.load(f)

count = 0

for file_id, meta in data.items():
    # 0 = acoustic
    if meta["instrument_source"] == 0:
        family_id = meta["instrument_family"]
        family_name = INSTRUMENT_FAMILIES.get(family_id)

        if family_name:
            src = os.path.join(AUDIO_DIR, file_id + ".wav")
            dst = os.path.join(OUTPUT_BASE, family_name, file_id + ".wav")

            if os.path.exists(src):
                shutil.copy(src, dst)
                count += 1

print("Total acoustic files copied:", count)
