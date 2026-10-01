import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

folder = Path(__file__).resolve().parent
labels = {0: "Open", 1: "Fist", 2: "Point"}

# قراءة البيانات
with (folder / "gestures.csv").open(newline="") as file:
    rows = list(csv.DictReader(file))

# ترتيب الجلسات حسب ظهورها في الملف
sessions = list(dict.fromkeys(row["session"] for row in rows))

if len(sessions) != 2:
    raise ValueError(
        f"Expected 2 sessions, found {len(sessions)}. "
        "Share this message before continuing."
    )

feature_names = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y")
]


def load_session(session):
    selected = [r for r in rows if r["session"] == session]

    x = np.array(
        [[float(r[name]) for name in feature_names] for r in selected],
        dtype=np.float32,
    )
    y = np.array([int(r["label"]) for r in selected])

    if not np.isfinite(x).all():
        raise ValueError("Invalid coordinates found.")

    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError("Each session must contain all 3 gestures.")

    return x, y


x_train, y_train = load_session(sessions[0])
x_test, y_test = load_session(sessions[1])

print("Training samples:", len(y_train))
print("Test samples:", len(y_test))

for label, name in labels.items():
    print(
        f"{name}: train={np.sum(y_train == label)}, "
        f"test={np.sum(y_test == label)}"
    )

# التدريب يستخدم الجلسة الأولى فقط
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    solver="lbfgs",
    alpha=0.01,
    max_iter=1000,
    random_state=42,
)

model.fit(x_train, y_train)

# اختبار النموذج على الجلسة الثانية
predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.1%}")
print("\nConfusion matrix: Open, Fist, Point")
print("Rows = actual | Columns = predicted")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# حفظ النموذج ومعلومات تجهيز البيانات
model_path = folder / "gesture_model.joblib"
joblib.dump(
    {
        "model": model,
        "labels": labels,
        "normalization": "pixel_xy_wrist_maxabs",
        "train_session": sessions[0],
        "test_session": sessions[1],
    },
    model_path,
)

print(f"\nSaved: {model_path.name}")