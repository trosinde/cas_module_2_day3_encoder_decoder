"""Übernimmt den echten v4-Lauf (32 Zahlen, Gewicht 1e-5, 25 Epochen) in Modelle und Erklärseite.

Quelle: ~/dev/.nb_run_tmp. Der Nachlauf R0 (256 Zahlen) bleibt als eigener Vergleichswert erhalten.
Aufruf: real_v4.py <Release-Ordner>. Läuft auf der CPU.
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
import tensorflow as tf
from keras import models
from keras.applications import vgg16
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import exp

HERE = Path(__file__).resolve().parent
ART = HERE / "article"
LOCAL_MODELS = exp.NOTEBOOKS / "models"
SOURCE = Path.home() / "dev" / ".nb_run_tmp" / "models"
RELEASE = Path(sys.argv[1]) / "models"

auto = models.load_model(SOURCE / "autoencoder_v4.keras", compile=False)
enc = models.load_model(SOURCE / "encoder_v4.keras", compile=False)
dec = models.load_model(SOURCE / "decoder_v4.keras", compile=False)
for target in (LOCAL_MODELS, RELEASE):
    auto.save(target / "autoencoder_v4_new.keras")
    enc.save(target / "encoder_v4.keras")
    dec.save(target / "decoder_v4.keras")
if (RELEASE / "autoencoder_v4.keras").exists() and not (RELEASE / "autoencoder_r0.keras").exists():
    (RELEASE / "autoencoder_v4.keras").rename(RELEASE / "autoencoder_r0.keras")
for target in (LOCAL_MODELS, RELEASE):
    (target / "autoencoder_v4_new.keras").rename(target / "autoencoder_v4.keras")

(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x = x_test.astype("float32") / 255.0
y = y_test.ravel()
ex, lab = x[:5000], y[:5000]
recon = auto.predict(x, batch_size=500, verbose=0)
metrics = exp.evaluate(recon, x)
metrics.pop("spectrum")

emb = enc.predict(ex, batch_size=500, verbose=0)
a, b, c, d = train_test_split(emb, lab, test_size=0.2, random_state=42, stratify=lab)
metrics["knn"] = float(KNeighborsClassifier(5).fit(a, c).score(b, d))
metrics["dim"] = int(emb.shape[1])
idx = np.random.RandomState(0).permutation(5000)[:2000]
pts = TSNE(2, init="pca", perplexity=30, random_state=42).fit_transform(emb)
lo, hi = pts[idx].min(0), pts[idx].max(0)
scale = lambda p: np.round((p - lo) / (hi - lo) * 999).astype(int)
metrics["points"] = scale(pts[idx]).tolist()
metrics["sample"] = np.clip(scale(pts[:10]), 0, 999).tolist()

data = json.loads((ART / "data.json").read_text())
data["versions"]["v4"] = metrics
(ART / "data.json").write_text(json.dumps(data))


def strip(name, imgs):
    arr = np.clip(np.concatenate(list(imgs[:10]), axis=1) * 255, 0, 255).astype("uint8")
    Image.fromarray(arr).save(ART / "strips" / f"{name}.png")


strip("v4", recon)
strip("v4_diff", np.clip(np.abs(x - recon) * 4, 0, 1))


def crop(name, i=7):
    return (np.asarray(Image.open(ART / "strips" / f"{name}.png").convert("RGB"), dtype="float32") / 255.0)[:, i * 32:(i + 1) * 32]


imgs = {n: crop(n) for n in ("original", "v4", "v5")}
names = ["block1_conv2", "block2_conv2", "block3_conv3"]
vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
ext = models.Model(vgg.input, [vgg.get_layer(n).output for n in names])
feats = {k: [f.numpy()[0] for f in ext(vgg16.preprocess_input(v[None] * 255.0))] for k, v in imgs.items()}
out = {"layers": []}
for li, n in enumerate(names):
    o = feats["original"][li]
    ch = int(o.var(axis=(0, 1)).argmax())
    low, high = np.percentile(o[..., ch], 1), np.percentile(o[..., ch], 99)
    for k in imgs:
        g = np.clip((feats[k][li][..., ch] - low) / (high - low + 1e-6), 0, 1)
        Image.fromarray((g * 255).astype("uint8")).save(ART / "strips" / f"feat{li}_{k}.png")
    out["layers"].append({"name": n, "size": o.shape[0], "channels": o.shape[2], "channel": ch,
                          "mse": {k: float(np.mean((feats[k][li] - o) ** 2)) for k in ("v4", "v5")}})
t = tf.constant(imgs["original"][None])
for k in ("v4", "v5"):
    p = tf.constant(imgs[k][None])
    fm = float(np.mean([l["mse"][k] for l in out["layers"]]))
    out[k] = {"l1": float(tf.reduce_mean(tf.abs(t - p))), "mse": float(tf.reduce_mean((t - p) ** 2)),
              "ssim7": float(tf.image.ssim(t, p, 1.0, filter_size=7)[0]), "vgg_raw": fm, "vgg_weighted": fm * 1e-5}
(ART / "feat.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in metrics.items() if k not in ("points", "sample")})
print("Frosch", out["v4"], out["v5"])
