"""Echte Loss-Werte und VGG-Feature-Bilder am Frosch (Testbild 8) für den Anhang."""
import json, os
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, tensorflow as tf
from PIL import Image
from keras.applications import vgg16
from keras import models
A = Path(__file__).parent / "article"
def crop(n, i=7):
    return (np.asarray(Image.open(A/"strips"/f"{n}.png").convert("RGB"), dtype="float32")/255.0)[:, i*32:(i+1)*32]
imgs = {n: crop(n) for n in ("original", "v4", "v5")}
names = ["block1_conv2", "block2_conv2", "block3_conv3"]
vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
ext = models.Model(vgg.input, [vgg.get_layer(n).output for n in names])
feats = {k: [f.numpy()[0] for f in ext(vgg16.preprocess_input(v[None]*255.0))] for k, v in imgs.items()}
out = {"layers": []}
for li, n in enumerate(names):
    o = feats["original"][li]; ch = int(o.var(axis=(0, 1)).argmax())
    lo, hi = np.percentile(o[..., ch], 1), np.percentile(o[..., ch], 99)
    for k in imgs:
        g = np.clip((feats[k][li][..., ch]-lo)/(hi-lo+1e-6), 0, 1)
        Image.fromarray((g*255).astype("uint8")).save(A/"strips"/f"feat{li}_{k}.png")
    out["layers"].append({"name": n, "size": o.shape[0], "channels": o.shape[2], "channel": ch,
        "mse": {k: float(np.mean((feats[k][li]-o)**2)) for k in ("v4", "v5")}})
t = tf.constant(imgs["original"][None])
for k in ("v4", "v5"):
    p = tf.constant(imgs[k][None])
    fm = float(np.mean([l["mse"][k] for l in out["layers"]]))
    out[k] = {"l1": float(tf.reduce_mean(tf.abs(t-p))), "mse": float(tf.reduce_mean((t-p)**2)),
              "ssim7": float(tf.image.ssim(t, p, 1.0, filter_size=7)[0]), "vgg_raw": fm, "vgg_weighted": fm*1e-6}
(A/"feat.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
