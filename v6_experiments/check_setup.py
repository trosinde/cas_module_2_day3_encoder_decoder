"""Startprüfung auf dem neuen Rechner: Dateien, GPU, Daten und ob das v5-Ergebnis reproduzierbar ist."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras

NB = Path(__file__).resolve().parent.parent
NEEDED = [
    "models/autoencoder_v3.keras",
    "models/autoencoder_v5.keras",
    "models/encoder_v5.keras",
    "autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb",
    "v5_experiments/runs/baselines.json",
    "v5_experiments/runs/R0.weights.h5",
    "v5_experiments/runs/R1.weights.h5",
    "v5_experiments/runs/R3.weights.h5",
    "v5_experiments/article/template.html",
    "v5_experiments/article/data.json",
]
missing = [p for p in NEEDED if not (NB / p).exists()]
print("Fehlende Dateien:", missing or "keine")
print("TensorFlow", tf.__version__, "Keras", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus or "keine, Training läuft auf der CPU")

(_, _), (x_test, _) = keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
print("CIFAR-10 geladen:", x_test.shape)
keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
print("VGG16-Gewichte geladen")

model = keras.models.load_model(NB / "models/autoencoder_v5.keras", compile=False)
recon = model.predict(x_test, batch_size=500, verbose=0)
mse = float(np.mean((x_test - recon) ** 2))
print(f"v5-MSE auf dem Testset: {mse:.5f} (erwartet 0,00216)")
ok = not missing and bool(gpus) and abs(mse - 0.00216) < 0.0001
print("ERGEBNIS:", "OK" if ok else "PROBLEM, siehe oben")
raise SystemExit(0 if ok else 1)
