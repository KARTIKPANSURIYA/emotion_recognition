import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "outputs")

X      = np.load(os.path.join(OUT, "X.npy"))
labels = np.load(os.path.join(OUT, "y.npy"))

le     = joblib.load(os.path.join(OUT, "label_encoder.pkl"))
scaler = joblib.load(os.path.join(OUT, "scaler.pkl"))

Y_enc = le.transform(labels)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, Y_enc, test_size=0.2, stratify=Y_enc, random_state=42
)
X_te_s = scaler.transform(X_te)

rf   = joblib.load(os.path.join(OUT, "rf.pkl"))
hgb  = joblib.load(os.path.join(OUT, "hgb.pkl"))
xgb  = joblib.load(os.path.join(OUT, "xgb.pkl"))
svc  = joblib.load(os.path.join(OUT, "svc.pkl"))
mlp  = joblib.load(os.path.join(OUT, "mlp.pkl"))
probs = np.load(os.path.join(OUT, "ensemble_probs.npy"))

print("\n=== Classification Reports ===\n")
for name, (model, X_input) in [
    ("RandomForest"         , (rf,  X_te)),
    ("HistGradientBoosting" , (hgb, X_te)),
    ("XGBoost"              , (xgb, X_te)),
    ("SVM (RBF)"            , (svc, X_te)),
    ("MLP"                  , (mlp, X_te_s)),
]:
    y_pred = model.predict(X_input)
    print(f"--- {name} ---")
    print(classification_report(
        y_te, y_pred, target_names=le.classes_
    ))

y_pred_ens = np.argmax(probs, axis=1)
cm = confusion_matrix(y_te, y_pred_ens)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Weighted Ensemble Confusion Matrix")
plt.tight_layout()

out_png = os.path.join(OUT, "ensemble_confusion_matrix.png")
plt.savefig(out_png)
print(f"\n✅ Saved ensemble confusion matrix to `{out_png}`")
