#!/usr/bin/env python3
import os
import glob
import soundfile as sf
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import evaluate
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
import joblib

# ── 1) Gather files & labels ─────────────────────────────────────────────────────
EMOTIONS = sorted(os.listdir("data/all"))
files, labels = [], []
for emo in EMOTIONS:
    for fn in glob.glob(f"data/all/{emo}/*.wav"):
        files.append(fn)
        labels.append(emo)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
os.makedirs("outputs", exist_ok=True)
joblib.dump(le, "outputs/label_encoder.pkl")

# Train/test split
train_files, test_files, train_labels, test_labels = train_test_split(
    files, y, test_size=0.2, stratify=y, random_state=42
)
train_ds = Dataset.from_dict({"path": train_files, "label": train_labels})
test_ds  = Dataset.from_dict({"path": test_files,  "label": test_labels})
dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

# ── 2) Load extractor & model ────────────────────────────────────────────────────
model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_),
    label2id={str(i): i for i in range(len(le.classes_))},
    id2label={i: emo for i, emo in enumerate(le.classes_)},
)

# ── 3) Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(batch):
    speech, sr = sf.read(batch["path"])
    if sr != feature_extractor.sampling_rate:
        speech = librosa.resample(
            speech, orig_sr=sr,
            target_sr=feature_extractor.sampling_rate
        )
    batch["input_values"] = feature_extractor(
        speech, sampling_rate=feature_extractor.sampling_rate
    ).input_values[0]
    return batch

dataset_dict = dataset_dict.map(preprocess, remove_columns=["path"], batched=False)

# ── 4) TrainingArguments ─────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="outputs/w2v2-emotion",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=1e-4,
    logging_dir="outputs/logs",
    logging_steps=50,
    save_steps=100
)

# ── 5) Metrics ───────────────────────────────────────────────────────────────────
metric = evaluate.load("accuracy")
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=p.label_ids)

# ── 6) Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# ── 7) Train ─────────────────────────────────────────────────────────────────────
trainer.train()

# ── 8) Final Evaluation ──────────────────────────────────────────────────────────
results = trainer.evaluate()
print(f"Test accuracy: {results['eval_accuracy']:.4f}")
