#!/usr/bin/env python3
import os
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

X = np.load(os.path.join(OUT, "X.npy"))
y = np.load(os.path.join(OUT, "y.npy"))

le    = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, os.path.join(OUT, "label_encoder.pkl"))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_te_s = scaler.transform(X_te)
joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))

print("Training RandomForest…")
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

print("Training HistGradientBoosting…")
hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42)
hgb.fit(X_tr, y_tr)

print("Training XGBoost…")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric="mlogloss",
    verbosity=0, random_state=42
)
xgb.fit(X_tr, y_tr)

print("Training SVM…")
svc = SVC(kernel="rbf", C=10, gamma="scale",
          probability=True, random_state=42)
svc.fit(X_tr, y_tr)

print("Training MLP…")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    early_stopping=True,
    n_iter_no_change=10,
    max_iter=200,
    random_state=42
)
mlp.fit(X_tr_s, y_tr)

joblib.dump(rf,  os.path.join(OUT, "rf.pkl"))
joblib.dump(hgb, os.path.join(OUT, "hgb.pkl"))
joblib.dump(xgb, os.path.join(OUT, "xgb.pkl"))
joblib.dump(svc, os.path.join(OUT, "svc.pkl"))
joblib.dump(mlp, os.path.join(OUT, "mlp.pkl"))

print("\nIndividual test accuracies:")
for name, (model, data, true) in [
    ("RF ", (rf,  X_te,   y_te)),
    ("HGB", (hgb, X_te,   y_te)),
    ("XGB", (xgb, X_te,   y_te)),
    ("SVC", (svc, X_te,   y_te)),
    ("MLP", (mlp, X_te_s, y_te)),
]:
    acc = accuracy_score(true, model.predict(data))
    print(f" • {name:>3}: {acc * 100:5.2f}%")

print("\nBuilding weighted soft‑voting ensemble…")
p_rf  = rf.predict_proba(X_te)
p_hgb = hgb.predict_proba(X_te)
p_xgb = xgb.predict_proba(X_te)
p_svc = svc.predict_proba(X_te)
p_mlp = mlp.predict_proba(X_te_s)

# TUNED WEIGHTS
w_rf, w_hgb, w_xgb, w_svc, w_mlp = 1.0, 1.1, 1.2, 1.3, 1.5
W = w_rf + w_hgb + w_xgb + w_svc + w_mlp

ensemble_probs = (
    w_rf  * p_rf  +
    w_hgb * p_hgb +
    w_xgb * p_xgb +
    w_svc * p_svc +
    w_mlp * p_mlp
) / W

y_pred_ens = np.argmax(ensemble_probs, axis=1)
acc_ens = accuracy_score(y_te, y_pred_ens)
print(f"→ Ensemble accuracy: {acc_ens * 100:5.2f}%")

np.save(os.path.join(OUT, "ensemble_probs.npy"), ensemble_probs)
joblib.dump([w_rf, w_hgb, w_xgb, w_svc, w_mlp],
            os.path.join(OUT, "ensemble_weights.pkl"))

print(f"\n✅ Artifacts saved under `{OUT}`")
