#!/usr/bin/env python3
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "outputs"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)

X = np.load(os.path.join(OUTPUT_ROOT, "X.npy"))  # (4040,1024)
y = np.load(os.path.join(OUTPUT_ROOT, "y.npy"))  # (4040,)

le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, os.path.join(OUTPUT_ROOT, "label_encoder.pkl"))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 5],
}
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

print("Starting GridSearchCV for RandomForest...")
grid.fit(X_tr, y_tr)
best = grid.best_estimator_
print("Best params:", grid.best_params_)
print(f"CV Accuracy: {grid.best_score_:.3f}")

joblib.dump(best, os.path.join(OUTPUT_ROOT, "rf_model.pkl"))

test_acc = best.score(X_te, y_te)
print(f"Test accuracy: {test_acc*100:.2f}%")
