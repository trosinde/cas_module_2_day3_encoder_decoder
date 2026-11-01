"""Cluster-Positionen der zehn Beispielbilder für die Erklärseite. Läuft auf der CPU.

Rechnet die Einbettungen und t-SNE wie v5_experiments/gather.py und vergleicht das Ergebnis mit den
gespeicherten 2.000 Punkten. Schreibt article/sample_points.json: je Version Punkte und Beispielpositionen.
"""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.manifold import TSNE

import exp

HERE = Path(__file__).resolve().parent
NB = exp.NOTEBOOKS
RUNS_V5 = NB / "v5_experiments" / "runs"
_, x_test = exp.load_data()
ex = x_test[:5000]
idx = np.random.RandomState(0).permutation(5000)[:2000]
stored = json.loads((HERE / "article" / "data.json").read_text())["versions"]


class Sampling(layers.Layer):
    def call(self, inputs):
        m, lv = inputs
        return m + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(m))


def load(name):
    return models.load_model(NB / "models" / name, compile=False,
                             custom_objects={"Sampling": Sampling}, safe_mode=False)


def v1():
    m = load("autoencoder.keras")
    return keras.Model(m.input, m.get_layer("encoder_output").output).predict(ex, batch_size=500, verbose=0)


def v2():
    return load("encoder.keras").predict(ex, batch_size=500, verbose=0)[2]


def v3():
    return load("encoder_v3.keras").predict(ex, batch_size=500, verbose=0)


def v4():
    m = exp.build_v4(256)
    m.load_weights(RUNS_V5 / "R0.weights.h5")
    dense = [l for l in m.layers if isinstance(l, layers.Dense)][0]
    return keras.Model(m.input, dense.output).predict(ex, batch_size=500, verbose=0)


def v5():
    return load("encoder_v5.keras").predict(ex, batch_size=500, verbose=0).reshape(5000, -1)


result = {}
for name, fn in (("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4), ("v5", v5)):
    keras.utils.set_random_seed(0)
    emb = fn()
    pts = emb if emb.shape[1] == 2 else TSNE(2, init="pca", perplexity=30, random_state=42).fit_transform(emb)
    lo, hi = pts[idx].min(0), pts[idx].max(0)
    scale = lambda p: np.round((p - lo) / (hi - lo) * 999).astype(int)
    new = scale(pts[idx])
    sample = scale(pts[:10])
    old = np.array(stored[name]["points"])
    deviation = float(np.abs(new - old).max())
    result[name] = {
        "points": new.tolist(),
        "sample": np.clip(sample, 0, 999).tolist(),
        "max_deviation": deviation,
        "outside": int(((sample < 0) | (sample > 999)).any(axis=1).sum()),
    }
    print(name, "größte Abweichung zu den alten Punkten:", deviation, "außerhalb:", result[name]["outside"], flush=True)
    (HERE / "article" / "sample_points.json").write_text(json.dumps(result))
