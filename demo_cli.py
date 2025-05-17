
import os, io, sys, queue, argparse
import numpy as np, soundfile as sf, sounddevice as sd, joblib
import tensorflow as tf, tensorflow_hub as hub

os.environ["CUDA_VISIBLE_DEVICES"] = ""
tf.get_logger().setLevel("ERROR")

YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
OUT    = os.path.join(os.path.dirname(__file__), "outputs")
LE     = joblib.load(f"{OUT}/label_encoder.pkl")
SCALER = joblib.load(f"{OUT}/scaler.pkl")
W_RF, W_HGB, W_XGB, W_SVC, W_MLP = joblib.load(f"{OUT}/ensemble_weights.pkl")
MODELS = {name: joblib.load(f"{OUT}/{name}.pkl") for name in ["rf","hgb","xgb","svc","mlp"]}

def extract_embedding(wav, sr=16000):
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, sr, 16000)
    _, emb_frames, _ = YAMNET(wav)
    return np.mean(emb_frames.numpy(), axis=0)[None,:]

def predict_emotion(emb):
    p = [(w*MODELS[n].predict_proba(SCALER.transform(emb) if n=="mlp" else emb))
         for n,w in zip(MODELS, [W_RF,W_HGB,W_XGB,W_SVC,W_MLP])]
    P = sum(p) / sum([W_RF,W_HGB,W_XGB,W_SVC,W_MLP])
    idx = int(P.argmax(axis=1)[0])
    return LE.inverse_transform([idx])[0], float(P[0,idx])

def record_audio(duration=10, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        if status: print(status, file=sys.stderr)
        q.put(indata.copy())
    print(f"⏺️  Recording {duration}s of audio (sampling at {sr} Hz)…")
    frames=[]
    with sd.InputStream(channels=1, samplerate=sr, callback=cb):
        for _ in range(int(sr/1024*duration)): frames.append(q.get())
    return np.concatenate(frames).flatten(), sr

def main():
    p = argparse.ArgumentParser(description="CLI Speech Emotion Recognition Demo")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("-f","--file",  help="path to WAV file")
    grp.add_argument("-r","--record",action="store_true", help="record 10 s from mic")
    args = p.parse_args()

    if args.file:
        wav, sr = sf.read(args.file)
    else:
        wav, sr = record_audio()

    emb = extract_embedding(wav, sr)
    emo, conf = predict_emotion(emb)
    print(f"\n✅  Predicted emotion: {emo.upper():<8} (confidence {conf*100:5.1f} %)\n")

if __name__ == "__main__":
    main()
