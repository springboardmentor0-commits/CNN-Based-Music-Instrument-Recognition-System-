import os
import json
import shutil
import random
from collections import defaultdict

# ---------------- CONFIG ----------------
BASE_DIR = "filtered_nsynth"
INPUT_DIR = os.path.join(BASE_DIR, "train_acoustic")
OUTPUT_DIR = os.path.join(BASE_DIR, "train_acoustic_balanced")

AUDIO_IN = os.path.join(INPUT_DIR, "audio")
AUDIO_OUT = os.path.join(OUTPUT_DIR, "audio")

LABELS_IN = os.path.join(INPUT_DIR, "labels.json")
LABELS_OUT = os.path.join(OUTPUT_DIR, "labels.json")

MAX_SAMPLES_PER_CLASS = 1500   # 👈 you can change (1000 / 1500 / 2000)
RANDOM_SEED = 42
# ---------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    random.seed(RANDOM_SEED)

    ensure_dir(AUDIO_OUT)

    # Load labels
    with open(LABELS_IN, "r") as f:
        labels = json.load(f)

    # Group samples by class
    class_to_ids = defaultdict(list)
    for sample_id, info in labels.items():
        class_name = info["instrument_family_str"]
        class_to_ids[class_name].append(sample_id)

    print("\n📊 CLASS COUNTS BEFORE BALANCING")
    for cls, ids in sorted(class_to_ids.items()):
        print(f"{cls:12s} : {len(ids)}")

    # Balance classes
    balanced_labels = {}

    for cls, ids in class_to_ids.items():
        if len(ids) > MAX_SAMPLES_PER_CLASS:
            selected_ids = random.sample(ids, MAX_SAMPLES_PER_CLASS)
        else:
            selected_ids = ids  # keep all

        for sid in selected_ids:
            balanced_labels[sid] = labels[sid]

            src_wav = os.path.join(AUDIO_IN, sid + ".wav")
            dst_wav = os.path.join(AUDIO_OUT, sid + ".wav")

            if os.path.exists(src_wav):
                shutil.copy2(src_wav, dst_wav)

    # Save new labels
    with open(LABELS_OUT, "w") as f:
        json.dump(balanced_labels, f, indent=2)

    # Final counts
    final_counts = defaultdict(int)
    for info in balanced_labels.values():
        final_counts[info["instrument_family_str"]] += 1

    print("\n✅ CLASS COUNTS AFTER BALANCING")
    for cls, cnt in sorted(final_counts.items()):
        print(f"{cls:12s} : {cnt}")

    print("\n🎉 Train set balancing complete.")
    print(f"Total samples after balancing: {len(balanced_labels)}")


if __name__ == "__main__":
    main()
