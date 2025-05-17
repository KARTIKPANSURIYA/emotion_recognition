#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import multiprocessing as mp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

DATA_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "data", "all"))
OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "outputs"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)

EMOTIONS = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith(".")
)

SR = 16000
yamnet_model = None

def init_worker():
    """Load YAMNet model once per process."""
    global yamnet_model
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def process_file(task):
    """Embed one file; returns (embedding, label) or None on error."""
    emo, fname = task
    fp = os.path.join(DATA_ROOT, emo, fname)
    try:
        wav, _ = librosa.load(fp, sr=SR, mono=True)
        _, embeddings, _ = yamnet_model(wav)
        emb = np.mean(embeddings.numpy(), axis=0)
        return emb, emo
    except Exception as e:
        print(f"[ERROR] {emo}/{fname}: {e}")
        return None

def main():
    tasks = [
        (emo, f)
        for emo in EMOTIONS
        for f in os.listdir(os.path.join(DATA_ROOT, emo))
        if f.lower().endswith(".wav")
    ]
    N = len(tasks)
    print(f"→ Embedding {N} files across {len(EMOTIONS)} emotions with 4 workers…")

    with mp.Pool(processes=4, initializer=init_worker) as pool:
        results = []
        for res in pool.imap_unordered(process_file, tasks):
            if res:
                results.append(res)
            if len(results) % 200 == 0:
                print(f"  • {len(results)}/{N} files embedded")

    X = np.vstack([emb for emb, _ in results])
    y = np.array([label for _, label in results])

    np.save(os.path.join(OUTPUT_ROOT, "X.npy"), X)
    np.save(os.path.join(OUTPUT_ROOT, "y.npy"), y)
    print(f"\n✅ Saved embeddings → `outputs/X.npy` (shape {X.shape})")
    print(f"✅ Saved labels     → `outputs/y.npy` (shape {y.shape})")

if __name__ == "__main__":
    main()
