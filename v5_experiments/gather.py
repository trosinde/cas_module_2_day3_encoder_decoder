"""Rechnet Rekonstruktionen, Kennzahlen und Cluster aller Versionen einheitlich für die Erklärseite."""
import json, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, keras, tensorflow as tf
from PIL import Image
from keras import models, layers
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import exp

HERE = Path(__file__).parent
NB = exp.NOTEBOOKS
ART = HERE / "article"; (ART / "strips").mkdir(parents=True, exist_ok=True)
(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0; y = y_test.ravel()
ex = x_test[:5000]; lab = y[:5000]

def strip(name, imgs):
    a = np.clip(np.concatenate(list(imgs[:10]), axis=1) * 255, 0, 255).astype("uint8")
    Image.fromarray(a).save(ART / "strips" / f"{name}.png")

class Sampling(layers.Layer):
    def call(self, inputs):
        m, lv = inputs
        return m + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(m))

CUSTOM = {"Sampling": Sampling}
def load(p):
    return models.load_model(NB / "models" / p, compile=False, custom_objects=CUSTOM, safe_mode=False)

def v1():
    m = load("autoencoder.keras")
    return m.predict(x_test, batch_size=500, verbose=0), keras.Model(m.input, m.get_layer("encoder_output").output).predict(ex, batch_size=500, verbose=0)
def v2():
    e, d = load("encoder.keras"), load("decoder.keras")
    z = e.predict(x_test, batch_size=500, verbose=0)[2]
    return d.predict(z, batch_size=500, verbose=0), z[:5000]
def v3():
    return load("autoencoder_v3.keras").predict(x_test, batch_size=500, verbose=0), load("encoder_v3.keras").predict(ex, batch_size=500, verbose=0)
def v4():
    m = exp.build_v4(256); m.load_weights(exp.OUT / "R0.weights.h5")
    dense = [l for l in m.layers if isinstance(l, layers.Dense)][0]
    return m.predict(x_test, batch_size=500, verbose=0), keras.Model(m.input, dense.output).predict(ex, batch_size=500, verbose=0)
def v5():
    return load("autoencoder_v5.keras").predict(x_test, batch_size=500, verbose=0), load("encoder_v5.keras").predict(ex, batch_size=500, verbose=0).reshape(5000, -1)
def v5_256():
    m = exp.build_v5(4, [32, 64, 128]); m.load_weights(exp.OUT / "R1.weights.h5")
    return m.predict(x_test, batch_size=500, verbose=0), keras.Model(m.input, m.get_layer("latent").output).predict(ex, batch_size=500, verbose=0).reshape(5000, -1)

strip("original", x_test)
data = {"labels": None, "versions": {}}
idx = np.random.RandomState(0).permutation(5000)[:2000]
data["labels"] = lab[idx].tolist()
for name, fn in [("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4), ("v5_256", v5_256), ("v5", v5)]:
    try:
        recon, emb = fn()
    except Exception as err:
        print("FEHLER", name, repr(err)[:400], flush=True); continue
    m = exp.evaluate(recon, x_test); m.pop("spectrum")
    strip(name, recon)
    strip(name + "_diff", np.clip(np.abs(x_test - recon) * 4, 0, 1))
    a, b, c, d = train_test_split(emb, lab, test_size=0.2, random_state=42, stratify=lab)
    m["knn"] = float(KNeighborsClassifier(5).fit(a, c).score(b, d))
    m["dim"] = int(emb.shape[1])
    pts = emb if emb.shape[1] == 2 else TSNE(2, init="pca", perplexity=30, random_state=42).fit_transform(emb)
    pts = pts[idx]; lo, hi = pts.min(0), pts.max(0)
    m["points"] = np.round((pts - lo) / (hi - lo) * 999).astype(int).tolist()
    data["versions"][name] = m
    print(name, {k: v for k, v in m.items() if k != "points"}, flush=True)
(ART / "data.json").write_text(json.dumps(data))
