# Emotion Recognition from Speech

A practical speech emotion recognition pipeline built on **YAMNet embeddings** and a
**weighted soft-voting ensemble** of classical ML models.

This repository includes:
- end-to-end data preparation from RAVDESS + TESS,
- data augmentation for minority classes,
- embedding extraction with TensorFlow Hub YAMNet,
- ensemble training/evaluation,
- and a CLI demo for live microphone or WAV-file inference.

## Project Structure

```text
emotion_recognition/
├── data/
│   ├── ravdess/
│   ├── tess/
│   └── all/                      # merged per-emotion wav files
├── outputs/                      # saved embeddings, encoders, models, metrics
├── src/
│   ├── prepare_data.py
│   ├── augment_data.py
│   ├── extract_embeddings.py
│   ├── train_classifier.py
│   ├── evaluate_metrics.py
│   ├── train_model.py            # older RF-only path
│   └── evaluate.py               # older Keras eval path
├── scripts/
│   └── finetune_wav2vec2.py      # optional transformer fine-tuning path
├── demo_cli.py                   # inference demo
├── requirements.txt
└── readme.md
```

## Environment Setup

> Recommended: Python 3.10+ in a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

This project expects speech audio under:
- `data/ravdess/audio_speech_actors_01-24`
- `data/tess` (with folders like `OAF_*` and `YAF_*`)

`src/prepare_data.py` creates per-emotion folders in `data/all/` and symlinks files
from source datasets.

## Training Pipeline (Recommended)

Run these steps from the repository root:

1. **Prepare merged dataset folders (symlinks)**
   ```bash
   python src/prepare_data.py
   ```

2. **Augment classes (`calm`, `surprise`)**
   ```bash
   python src/augment_data.py
   ```

3. **Extract YAMNet embeddings (`outputs/X.npy`, `outputs/y.npy`)**
   ```bash
   python src/extract_embeddings.py
   ```

4. **Train weighted ensemble and save artifacts**
   ```bash
   python src/train_classifier.py
   ```

5. **Print reports and confusion matrix**
   ```bash
   python src/evaluate_metrics.py
   ```

### Saved Artifacts

After training, `outputs/` contains:
- `X.npy`, `y.npy` (features/labels),
- `label_encoder.pkl`,
- `rf.pkl`, `hgb.pkl`, `xgb.pkl`, `svc.pkl`, `mlp.pkl`,
- `scaler.pkl` (for MLP input),
- `ensemble_weights.pkl`, `ensemble_probs.npy`,
- `ensemble_confusion_matrix.png`.

## Demo Inference

### Classify a WAV file
```bash
python demo_cli.py --file path/to/file.wav
```

### Record 10 seconds from microphone and classify
```bash
python demo_cli.py --record
```

The CLI prints the predicted emotion label and confidence score.

## Optional: Wav2Vec2 Fine-Tuning Path

An alternative pipeline is available at `scripts/finetune_wav2vec2.py` using
Hugging Face Transformers (`facebook/wav2vec2-base`) for sequence classification.

```bash
python scripts/finetune_wav2vec2.py
```

> This script requires additional dependencies beyond `requirements.txt`
> (for example `transformers`, `datasets`, and `evaluate`).

## Notes

- `demo_cli.py` expects trained artifacts in `outputs/` (encoder, scaler, models,
  and ensemble weights).
- Audio is processed at 16 kHz for embedding extraction/inference.
- Existing `src/train_model.py` and `src/evaluate.py` appear to be older paths kept
  for reference.

## License

No explicit license file is currently included in this repository.
Add one (e.g., MIT/Apache-2.0) if you plan to distribute or reuse this project.
