"""KID und normierter VGG-Abstand für fertige Läufe. Läuft auf der CPU, also neben dem Training.

Aufruf: perceptual.py B0 B1 ...   Referenzen: perceptual.py --references
Der VGG-Abstand ist bewusst nicht LPIPS: gleiche Idee, aber ohne dessen gelernte Gewichte.
"""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras.applications import inception_v3

import exp

CACHE = Path.home() / ".cache" / "ae_v6"
BATCH = 16  # 64 belegte rund 10 GB RAM und führte zusammen mit anderen Läufen zum OOM-Kill
LAYERS = ["block1_conv2", "block2_conv2", "block3_conv3"]
KID_SUBSETS = 100
KID_SUBSET_SIZE = 1000

inception = inception_v3.InceptionV3(weights="imagenet", include_top=False, pooling="avg",
                                     input_shape=(299, 299, 3))
normed = exp.make_normed_vgg(64, LAYERS)


def batches(images):
    return (images[i : i + BATCH] for i in range(0, len(images), BATCH))


def inception_features(images):
    feats = []
    for b in batches(images):
        resized = tf.image.resize(b, (299, 299), method="bilinear")
        feats.append(inception(inception_v3.preprocess_input(resized * 255.0), training=False).numpy())
    return np.concatenate(feats)


def real_features(name, images):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"inception_{name}.npy"
    if path.exists():
        return np.load(path)
    feats = inception_features(images)
    np.save(path, feats)
    return feats


def kid(a, b, seed=0):
    """Polynomialer MMD-Schätzer wie im KID-Paper. Mittel und Streuung über Teilmengen."""
    rng = np.random.RandomState(seed)
    dim = a.shape[1]
    scores = []
    for _ in range(KID_SUBSETS):
        x = a[rng.choice(len(a), KID_SUBSET_SIZE, replace=False)]
        y = b[rng.choice(len(b), KID_SUBSET_SIZE, replace=False)]
        kxx = (x @ x.T / dim + 1) ** 3
        kyy = (y @ y.T / dim + 1) ** 3
        kxy = (x @ y.T / dim + 1) ** 3
        m = KID_SUBSET_SIZE
        scores.append(
            (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
            + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
            - 2 * kxy.mean()
        )
    return float(np.mean(scores)), float(np.std(scores))


def vgg_distance(real, recon):
    total = 0.0
    for start in range(0, len(real), BATCH):
        t = normed(real[start : start + BATCH])
        p = normed(recon[start : start + BATCH])
        per_layer = [tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.square(a - b), axis=-1), axis=[1, 2]))
                     for a, b in zip(t, p)]
        total += float(tf.add_n(per_layer)) / len(per_layer)
    return total / len(real)


def score(real, recon, real_feats):
    k, s = kid(inception_features(recon), real_feats)
    return {"kid": k, "kid_std": s, "vgg_distance": vgg_distance(real, recon)}


def load_splits():
    x_train, x_test = exp.load_data()
    return x_train, x_train[45000:], x_test


def rebuild(tag):
    metrics = json.loads((exp.OUT / f"{tag}_metrics.json").read_text())
    model = exp.build_v5(8, metrics["widths"])
    model.load_weights(exp.OUT / f"{tag}.weights.h5")
    return model


def run_tag(tag, x_val, x_test, val_feats, test_feats):
    model = rebuild(tag)
    result = {
        "val": score(x_val, model.predict(x_val, batch_size=500, verbose=0), val_feats),
        "test": score(x_test, model.predict(x_test, batch_size=500, verbose=0), test_feats),
    }
    (exp.OUT / f"{tag}_perceptual.json").write_text(json.dumps(result, indent=1))
    print(tag, result, flush=True)


def run_references(x_train, x_test, test_feats):
    """v5, PCA 512 und als Untergrenze echte Trainingsbilder gegen echte Testbilder."""
    from sklearn.decomposition import PCA

    result = {}
    v5 = keras.models.load_model(exp.NOTEBOOKS / "models" / "autoencoder_v5.keras", compile=False)
    result["v5"] = score(x_test, v5.predict(x_test, batch_size=500, verbose=0), test_feats)
    pca = PCA(512, svd_solver="randomized", random_state=42).fit(x_train[:45000].reshape(45000, -1))
    flat = pca.inverse_transform(pca.transform(x_test.reshape(len(x_test), -1)))
    pca_recon = np.clip(flat, 0, 1).reshape(x_test.shape).astype("float32")
    result["pca512"] = score(x_test, pca_recon, test_feats)
    k, s = kid(real_features("train10k", x_train[:10000]), test_feats)
    result["real_train"] = {"kid": k, "kid_std": s}
    (exp.OUT / "references_perceptual.json").write_text(json.dumps(result, indent=1))
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="*")
    p.add_argument("--references", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out:
        exp.OUT = Path(args.out)
    x_train, x_val, x_test = load_splits()
    test_feats = real_features("test", x_test)
    val_feats = real_features("val", x_val)
    if args.references:
        run_references(x_train, x_test, test_feats)
    for tag in args.tags:
        run_tag(tag, x_val, x_test, val_feats, test_feats)
