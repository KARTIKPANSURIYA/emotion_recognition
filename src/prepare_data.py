#!/usr/bin/env python3
import os
import shutil

RAVDESS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "data", "ravdess",
                                            "audio_speech_actors_01-24"))
TESS_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "data", "tess"))
OUTPUT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", "data", "all"))

EMOTIONS = ["neutral", "calm", "happy", "sad",
            "angry", "fear", "disgust", "surprise"]

RAVDESS_MAP = {
    "01": "neutral", "02": "calm",   "03": "happy",   "04": "sad",
    "05": "angry",   "06": "fear",   "07": "disgust", "08": "surprise"
}


def prepare_output_dirs():
    for emo in EMOTIONS:
        os.makedirs(os.path.join(OUTPUT_ROOT, emo), exist_ok=True)
    print(f"Created `{OUTPUT_ROOT}/{{emotion}}` directories.")


def process_ravdess():
    if not os.path.isdir(RAVDESS_ROOT):
        raise FileNotFoundError(f"Missing RAVDESS folder: {RAVDESS_ROOT}")

    counts = {e: 0 for e in EMOTIONS}
    for actor in sorted(os.listdir(RAVDESS_ROOT)):
        actor_dir = os.path.join(RAVDESS_ROOT, actor)
        if not os.path.isdir(actor_dir):
            continue
        actor_id = actor.split("_")[-1]
        for fname in sorted(os.listdir(actor_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            parts = fname.split("-")
            emo = RAVDESS_MAP.get(parts[2])
            if emo not in EMOTIONS:
                continue
            src = os.path.join(actor_dir, fname)
            dst = os.path.join(OUTPUT_ROOT, emo, f"rav_{actor_id}_{fname}")
            # symlink is near-instantaneous
            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass
            counts[emo] += 1

    print("RAVDESS symlinks created:", counts)


def process_tess():
    if not os.path.isdir(TESS_ROOT):
        raise FileNotFoundError(f"Missing TESS folder: {TESS_ROOT}")

    counts = {e: 0 for e in EMOTIONS}
    for root, dirs, files in os.walk(TESS_ROOT):
        base = os.path.basename(root)
        if base.startswith(("OAF_", "YAF_")):
            dirs[:] = []

            _, emo_part = base.split("_", 1)
            emo_key = emo_part.lower()
            emo = "surprise" if emo_key == "pleasant_surprise" else emo_key
            if emo not in EMOTIONS:
                continue

            for fname in sorted(files):
                if not fname.lower().endswith(".wav"):
                    continue
                src = os.path.join(root, fname)
                dst = os.path.join(OUTPUT_ROOT, emo, f"tess_{base}_{fname}")
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                counts[emo] += 1

    print("TESS symlinks created:", counts)


if __name__ == "__main__":
    print("▶️  Starting fast data preparation (symlinks)…")
    prepare_output_dirs()
    process_ravdess()
    process_tess()
    print("\n✅ Done. Totals in data/all/:")
    for emo in EMOTIONS:
        cnt = len(os.listdir(os.path.join(OUTPUT_ROOT, emo)))
        print(f"  • {emo:9s}: {cnt}")
