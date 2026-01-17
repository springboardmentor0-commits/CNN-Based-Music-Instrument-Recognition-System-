import os
import json
import shutil
from tqdm import tqdm
from collections import defaultdict

# ---------------- CONFIG ----------------
RAW_ROOT = "nsynth-train-all"
AUDIO_DIR = os.path.join(RAW_ROOT, "audio")

SPLITS = {
    "train": "examples-train-original.json",
    "valid": "examples-valid-original.json",
    "test":  "examples-test-original.json"
}
 
OUTPUT_ROOT = "filtered_nsynth"

ACOUSTIC_SOURCE_ID = 0

# ❌ Classes to exclude completely
SKIP_FAMILY_IDS = {
    0,  # bass
    6,  # organ
    9   # synth_lead
}

INSTRUMENT_FAMILIES = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth_lead",
    10: "vocal"
}
# ---------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_split(split_name, json_file):
    print(f"\n================ {split_name.upper()} SPLIT ================")

    input_json_path = os.path.join(RAW_ROOT, json_file)
    output_audio_dir = os.path.join(
        OUTPUT_ROOT, f"{split_name}_acoustic", "audio"
    )
    output_json_path = os.path.join(
        OUTPUT_ROOT, f"{split_name}_acoustic", "labels.json"
    )

    ensure_dir(output_audio_dir)

    with open(input_json_path, "r") as f:
        metadata = json.load(f)

    total_samples = len(metadata)

    family_counts_before = defaultdict(int)
    family_counts_after = defaultdict(int)

    filtered_metadata = {}

    # ---------- FIRST PASS: COUNT BEFORE ----------
    for info in metadata.values():
        family_counts_before[info["instrument_family"]] += 1

    # ---------- SECOND PASS: FILTER + COPY ----------
    for note_id, info in tqdm(metadata.items(), desc=f"Copying {split_name}"):
        source = info["instrument_source"]
        family = info["instrument_family"]

        if (
            source == ACOUSTIC_SOURCE_ID
            and family not in SKIP_FAMILY_IDS
        ):
            wav_src = os.path.join(AUDIO_DIR, f"{note_id}.wav")
            wav_dst = os.path.join(output_audio_dir, f"{note_id}.wav")

            if os.path.exists(wav_src):
                shutil.copy2(wav_src, wav_dst)
                filtered_metadata[note_id] = info
                family_counts_after[family] += 1

    with open(output_json_path, "w") as f:
        json.dump(filtered_metadata, f, indent=2)

    # ---------- REPORT ----------
    print(f"\n📊 {split_name.upper()} SPLIT SUMMARY")
    print(f"Total samples (raw)     : {total_samples}")
    print(f"Samples kept (final)    : {len(filtered_metadata)}")

    print("\nInstrument distribution BEFORE filtering:")
    for fid, cnt in sorted(family_counts_before.items()):
        print(f"  {INSTRUMENT_FAMILIES.get(fid, fid):12s} : {cnt}")

    print("\nInstrument distribution AFTER filtering:")
    for fid, cnt in sorted(family_counts_after.items()):
        print(f"  {INSTRUMENT_FAMILIES.get(fid, fid):12s} : {cnt}")


def main():
    ensure_dir(OUTPUT_ROOT)

    for split, json_name in SPLITS.items():
        process_split(split, json_name)

    print("\n✅ Acoustic dataset filtering complete (bass, organ, synth_lead removed).")


if __name__ == "__main__":
    main()
