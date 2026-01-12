import os
import json
import shutil
from collections import defaultdict

BASE_DIR =  r"C:\Users\tyagi\Downloads\nsynth-train.jsonwav"
ACOUSTIC_DIR = os.path.join(BASE_DIR, "nsynth-acoustic")
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

os.makedirs(WORK_DIR, exist_ok=True)
counts = defaultdict(int)

for instrument in ALLOWED_CLASSES:
    src_dir = os.path.join(ACOUSTIC_DIR, instrument)

    if not os.path.exists(src_dir):
        print(f"Skipping {instrument} (folder not found)")
        continue

    dest_dir = os.path.join(WORK_DIR, instrument)
    os.makedirs(dest_dir, exist_ok=True)

    for file in os.listdir(src_dir):
        if not file.endswith(".wav"):
            continue

        if counts[instrument] >= FILES_PER_CLASS:
            break

        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(dest_dir, file)
        )
        counts[instrument] += 1

print("\nFinal subset created:")
for k, v in counts.items():
    print(f"{k}: {v}")