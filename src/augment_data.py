import os
import librosa
import soundfile as sf

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", "data", "all"))
TARGETS   = ["calm", "surprise"]
SR        = 16000

STRETCH_RATES = [0.9, 1.1]
PITCH_STEPS   = [-2, 2]

for emo in TARGETS:
    folder = os.path.join(DATA_ROOT, emo)
    orig_files = [f for f in os.listdir(folder)
                  if f.lower().endswith(".wav") and not f.startswith("aug_")]

    added = 0
    for fname in orig_files:
        src_path = os.path.join(folder, fname)
        y, _ = librosa.load(src_path, sr=SR)
        base, _ = os.path.splitext(fname)

        for rate in STRETCH_RATES:
            y_s = librosa.effects.time_stretch(y, rate=rate)
            out_fname = f"aug_{base}_stretch{int(rate*100)}.wav"
            sf.write(os.path.join(folder, out_fname), y_s, SR)
            added += 1

        for step in PITCH_STEPS:
            y_p = librosa.effects.pitch_shift(y, sr=SR, n_steps=step)
            out_fname = f"aug_{base}_pitch{step:+d}.wav"
            sf.write(os.path.join(folder, out_fname), y_p, SR)
            added += 1

    print(f"  • {emo}: added {added} new files")
