"""Train the neural network from classification.ipynb and save what app.py needs.

Run from this folder:  python train.py
It writes model.keras, scaler.pkl and report.json next to this file.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

folder = Path(__file__).parent

# Seeds Python, NumPy and TensorFlow so every run gives the same model
keras.utils.set_random_seed(3)

dataset = load_breast_cancer()
X, Y = dataset.data, dataset.target  # target: 0 = malignant, 1 = benign
row_ids = np.arange(len(X))

# Same split as the notebook. row_ids lets the app find the test records again.
X_train, X_test, Y_train, Y_test, train_ids, test_ids = train_test_split(
    X, Y, row_ids, test_size=0.2, random_state=2
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)  # transform only: the test data must not change the scaler

model = keras.Sequential([
    keras.Input(shape=(30,)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Validation loss keeps falling until about epoch 50, then levels off
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=50, verbose=0)

loss, accuracy = model.evaluate(X_test_std, Y_test, verbose=0)
Y_pred = model.predict(X_test_std, verbose=0).argmax(axis=1)
matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])

model.save(folder / "model.keras")
joblib.dump(scaler, folder / "scaler.pkl")

report = {
    "test_accuracy": round(float(accuracy), 4),
    "test_loss": round(float(loss), 4),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "test_ids": test_ids.tolist(),
    # rows = actual, columns = predicted, order = [malignant, benign]
    "confusion_matrix": matrix.tolist(),
    "history": {key: [round(float(v), 4) for v in values] for key, values in history.history.items()},
}
(folder / "report.json").write_text(json.dumps(report, indent=1) + "\n")

print(f"\nTest accuracy: {accuracy:.4f}   Test loss: {loss:.4f}")
print("Confusion matrix (rows = actual, cols = predicted, [malignant, benign]):")
print(matrix)
