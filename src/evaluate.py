import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib

OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "outputs"))

X = np.load(os.path.join(OUTPUT_ROOT, "X.npy"))
y = np.load(os.path.join(OUTPUT_ROOT, "y.npy"))
le = joblib.load(os.path.join(OUTPUT_ROOT, "label_encoder.pkl"))

_, X_te, _, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_te_enc = le.transform(y_te)

model = load_model(os.path.join(OUTPUT_ROOT, "model.h5"))
history = np.load(os.path.join(OUTPUT_ROOT, "history.npy"))

plt.plot(history, marker='o')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_ROOT, "val_accuracy.png"))
print("Saved val_accuracy.png")

y_pred = model.predict(X_te).argmax(axis=1)
cm = confusion_matrix(y_te_enc, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_ROOT, "conf_matrix.png"))
print("Saved conf_matrix.png")

from sklearn.metrics import classification_report
print("\n" + classification_report(y_te_enc, y_pred, target_names=le.classes_))
