"""Bilder, Kennzahlen und Cluster von v6 für die Erklärseite. Läuft auf der CPU.

v6 ist das Treue-Modell (B1), v6g das Natürlichkeits-Modell (B3). Gleiches Verfahren wie bei v1 bis v5.
"""
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import exp

ART = Path(__file__).resolve().parent / "article"
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x = x_test.astype("float32") / 255.0
y = y_test.ravel()
ex, lab = x[:5000], y[:5000]
idx = np.random.RandomState(0).permutation(5000)[:2000]
data = json.loads((ART / "data.json").read_text())


def strip(name, imgs):
    arr = np.clip(np.concatenate(list(imgs[:10]), axis=1) * 255, 0, 255).astype("uint8")
    Image.fromarray(arr).save(ART / "strips" / f"{name}.png")


for key, tag in (("v6", "B1"), ("v6g", "B3")):
    model = exp.build_v5(8, [64, 128, 256])
    model.load_weights(exp.OUT / f"{tag}.weights.h5")
    model.compile(jit_compile=False)
    recon = model.predict(x, batch_size=500, verbose=0)
    metrics = exp.evaluate(recon, x)
    metrics.pop("spectrum")
    strip(key, recon)
    strip(key + "_diff", np.clip(np.abs(x - recon) * 4, 0, 1))
    enc = keras.Model(model.input, model.get_layer("latent").output)
    enc.compile(jit_compile=False)
    emb = enc.predict(ex, batch_size=500, verbose=0).reshape(5000, -1)
    a, b, c, d = train_test_split(emb, lab, test_size=0.2, random_state=42, stratify=lab)
    metrics["knn"] = float(KNeighborsClassifier(5).fit(a, c).score(b, d))
    metrics["dim"] = int(emb.shape[1])
    pts = TSNE(2, init="pca", perplexity=30, random_state=42).fit_transform(emb)
    lo, hi = pts[idx].min(0), pts[idx].max(0)
    scale = lambda p: np.round((p - lo) / (hi - lo) * 999).astype(int)
    metrics["points"] = scale(pts[idx]).tolist()
    metrics["sample"] = np.clip(scale(pts[:10]), 0, 999).tolist()
    metrics["perceptual"] = json.loads((exp.OUT / f"{tag}_perceptual.json").read_text())["test"]
    data["versions"][key] = metrics
    (ART / "data.json").write_text(json.dumps(data))
    print(key, tag, {k: v for k, v in metrics.items() if k not in ("points", "sample")}, flush=True)

for name in ("original", "B1", "B3"):
    path = exp.OUT / f"sr_output_{name}.npy"
    if path.exists():
        strip(f"sr_{name}", np.load(path))
        print("SR-Streifen", name, flush=True)
