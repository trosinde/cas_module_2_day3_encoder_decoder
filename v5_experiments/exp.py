"""v5-Einzelvergleiche R0-R3 und Basislinien. Schreibt nur in den Scratchpad-Ordner."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, callbacks
from keras.applications import vgg16
from scipy.ndimage import convolve

OUT = Path(__file__).resolve().parent / "runs"
NOTEBOOKS = Path(__file__).resolve().parent.parent
BATCH_SIZE = 100
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def load_data():
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    return x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0


# ---------- Architekturen ----------
def build_v4(embedding_dim=256):
    """Encoder und Decoder exakt wie im v4-Notebook."""
    inp = layers.Input((32, 32, 3))
    x = inp
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    shape = x.shape[1:]
    x = layers.Flatten()(x)
    z = layers.Dense(embedding_dim)(x)
    x = layers.Dense(int(np.prod(shape)))(z)
    x = layers.Reshape(shape)(x)
    for f in (256, 128, 64, 32):
        x = layers.Conv2DTranspose(f, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    out = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return models.Model(inp, out, name="v4")


def res_block(x, filters):
    skip = x if x.shape[-1] == filters else layers.Conv2D(filters, 1)(x)
    y = layers.GroupNormalization(groups=8)(x)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = layers.GroupNormalization(groups=8)(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    return layers.Add()([skip, y])


def build_v5(latent_channels=4, widths=(32, 64, 128)):
    """Räumlicher Engpass 8x8xC, Upsampling + Conv, keine Skips über den Engpass."""
    inp = layers.Input((32, 32, 3))
    x = layers.Conv2D(widths[0], 3, padding="same")(inp)
    x = res_block(x, widths[0])
    x = layers.Conv2D(widths[1], 3, strides=2, padding="same")(x)
    x = res_block(x, widths[1])
    x = layers.Conv2D(widths[2], 3, strides=2, padding="same")(x)
    x = res_block(x, widths[2])
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    z = layers.Conv2D(latent_channels, 1, name="latent")(x)

    x = layers.Conv2D(widths[2], 3, padding="same")(z)
    x = res_block(x, widths[2])
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Conv2D(widths[1], 3, padding="same")(x)
    x = res_block(x, widths[1])
    x = layers.UpSampling2D(2, interpolation="bilinear")(x)
    x = layers.Conv2D(widths[0], 3, padding="same")(x)
    x = res_block(x, widths[0])
    x = layers.GroupNormalization(groups=8)(x)
    x = layers.Activation("relu")(x)
    out = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return models.Model(inp, out, name="v5")


# ---------- Losses ----------
def vgg_extractor(size, layer_names):
    vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    vgg.trainable = False
    return models.Model(vgg.input, [vgg.get_layer(n).output for n in layer_names])


def to_vgg(images):
    return vgg16.preprocess_input(images * 255.0)


def make_v4_loss(weight=1e-6):
    extractor = vgg_extractor(32, ["block1_conv2", "block2_conv2", "block3_conv3"])

    def terms(y_true, y_pred):
        pixel = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
        t = extractor(to_vgg(y_true), training=False)
        p = extractor(to_vgg(y_pred), training=False)
        feat = sum(tf.reduce_mean(tf.square(a - b), axis=[1, 2, 3]) for a, b in zip(t, p)) / len(t)
        return {"pixel": pixel, "weighted_feature": weight * feat}

    return terms


def l1_ssim_terms(y_true, y_pred, ssim_weight):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7)
    return {"l1": l1, "weighted_ssim": ssim_weight * (1.0 - ssim)}


def make_r1_loss(ssim_weight=0.2):
    return lambda a, b: l1_ssim_terms(a, b, ssim_weight)


def make_r2_loss(ssim_weight, vgg_weight, tv_weight, vgg_size, layer_names):
    extractor = vgg_extractor(vgg_size, layer_names)

    def normed(images):
        resized = tf.image.resize(images, (vgg_size, vgg_size), method="bilinear")
        feats = extractor(to_vgg(resized), training=False)
        return [f / (tf.norm(f, axis=-1, keepdims=True) + 1e-8) for f in feats]

    def terms(y_true, y_pred):
        out = l1_ssim_terms(y_true, y_pred, ssim_weight)
        t = normed(y_true)
        p = normed(y_pred)
        feat = sum(
            tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(a) - b), axis=-1), axis=[1, 2])
            for a, b in zip(t, p)
        ) / len(t)
        out["weighted_vgg"] = vgg_weight * feat
        dx = tf.reduce_mean(tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]), axis=[1, 2, 3])
        dy = tf.reduce_mean(tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]), axis=[1, 2, 3])
        out["weighted_tv"] = tv_weight * (dx + dy)
        return out

    return terms


def total(terms_fn):
    def loss(y_true, y_pred):
        return tf.add_n(list(terms_fn(y_true, y_pred).values()))

    return loss


# ---------- Messung ----------
def radial_spectrum(images):
    luma = images @ LUMA
    luma = luma - luma.mean(axis=(1, 2), keepdims=True)
    power = np.abs(np.fft.fftshift(np.fft.fft2(luma), axes=(1, 2))) ** 2
    power = power.mean(axis=0)
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - 16, xx - 16).astype(int)
    return (np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel()))[:17]


def sharpness(images):
    return float(
        np.mean([convolve(i @ LUMA, LAPLACIAN, mode="reflect").var() for i in images])
    )


def checker_energy(images):
    """Mittlere Leistung des Luma-Kanals bei der höchsten Frequenz (Nyquist)."""
    luma = images @ LUMA
    power = np.abs(np.fft.fft2(luma)) ** 2
    return float(power[:, 16, 16].mean())


def evaluate(recon, x_test, batch=500):
    mse = float(np.mean((x_test - recon) ** 2))
    ssim = []
    for i in range(0, len(x_test), batch):
        ssim.append(tf.image.ssim(x_test[i : i + batch], recon[i : i + batch], max_val=1.0).numpy())
    return {
        "mse": mse,
        "psnr": float(-10 * np.log10(mse)),
        "ssim": float(np.concatenate(ssim).mean()),
        "sharpness": sharpness(recon[:5000]),
        "nyquist_power": checker_energy(recon),
        "spectrum": radial_spectrum(recon).tolist(),
    }


def save_grid(path, x_test, recon, n=10, zoom=4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5))
    for i in range(n):
        for row, img in enumerate(
            (x_test[i], recon[i], np.clip(np.abs(x_test[i] - recon[i]) * 4, 0, 1))
        ):
            axes[row, i].imshow(np.kron(img, np.ones((zoom, zoom, 1))), interpolation="nearest")
            axes[row, i].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------- Läufe ----------
def configure(run, args):
    if run == "R0":
        return build_v4(256), make_v4_loss(1e-6)
    if run == "R1":
        return build_v5(4, args.widths), make_r1_loss(args.ssim_weight)
    if run == "R2":
        return build_v5(4, args.widths), make_r2_loss(
            args.ssim_weight, args.vgg_weight, args.tv_weight, args.vgg_size, args.vgg_layers
        )
    if run == "R3":
        if args.r3_loss == "R2":
            terms = make_r2_loss(
                args.ssim_weight, args.vgg_weight, args.tv_weight, args.vgg_size, args.vgg_layers
            )
        else:
            terms = make_r1_loss(args.ssim_weight)
        return build_v5(8, args.widths), terms
    if run == "A1":  # alter Decoder, neuer Loss: trennt Decoder und Loss
        return build_v4(256), make_r1_loss(args.ssim_weight)
    if run == "A2":  # alter Decoder, nur MSE: zeigt den VGG-Effekt und den reinen Engpass
        return build_v4(256), lambda t, p: {"pixel": tf.reduce_mean(tf.square(t - p), axis=[1, 2, 3])}
    raise ValueError(run)


class TermLogger(callbacks.Callback):
    """Schreibt pro Epoche die einzelnen Loss-Terme auf einem festen Validierungs-Batch."""

    def __init__(self, terms_fn, batch, path):
        super().__init__()
        self.terms_fn, self.batch, self.path = terms_fn, batch, path
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(self.batch, training=False)
        terms = {k: float(tf.reduce_mean(v)) for k, v in self.terms_fn(self.batch, pred).items()}
        self.rows.append({"epoch": epoch, "val_loss": logs.get("val_loss"), "seconds": time.time(), **terms})
        self.path.write_text(json.dumps(self.rows, indent=1))
        print(f"epoch {epoch} {terms} val_loss={logs.get('val_loss'):.5f}", flush=True)


def run_training(run, args):
    OUT.mkdir(exist_ok=True)
    x_train, x_test = load_data()
    if run == "R0":
        # v4-Umfang; nur die Validierung kommt aus dem Trainingsset, damit das Testset sauber bleibt.
        x_fit, x_val = x_train[:10000], x_train[45000:47000]
    else:
        x_fit, x_val = x_train[: args.train_samples], x_train[45000:]

    model, terms_fn = configure(run, args)
    if args.init_from:
        model.load_weights(OUT / f"{args.init_from}.weights.h5")
    steps_per_epoch = len(x_fit) // BATCH_SIZE
    total_steps = steps_per_epoch * args.epochs

    if args.timing:
        model.compile(optimizer="adam", loss=total(terms_fn), jit_compile=False)
        batch = x_fit[:BATCH_SIZE]
        model.train_on_batch(batch, batch)  # Graph bauen
        start = time.time()
        for _ in range(args.timing):
            model.train_on_batch(batch, batch)
        step = (time.time() - start) / args.timing
        print(f"TIMING {run}: {step:.3f} s/step, {model.count_params()} params", flush=True)
        return

    if run == "R0":
        optimizer = keras.optimizers.Adam()
        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        train = tf.data.Dataset.from_tensor_slices(x_fit).shuffle(len(x_fit)).batch(BATCH_SIZE)
        train = train.map(lambda b: (b, b)).prefetch(tf.data.AUTOTUNE)
    else:
        schedule = keras.optimizers.schedules.CosineDecay(args.lr, total_steps, alpha=0.01)
        optimizer = keras.optimizers.Adam(schedule)
        cbs = []
        train = tf.data.Dataset.from_tensor_slices(x_fit).shuffle(len(x_fit)).batch(BATCH_SIZE)
        train = train.map(
            lambda b: (lambda f: (f, f))(tf.image.random_flip_left_right(b)),
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

    cbs.append(TermLogger(terms_fn, x_val[:BATCH_SIZE], OUT / f"{run}_terms.json"))
    model.compile(optimizer=optimizer, loss=total(terms_fn), jit_compile=False)
    start = time.time()
    model.fit(train, epochs=args.epochs, validation_data=(x_val, x_val),
              batch_size=BATCH_SIZE, callbacks=cbs, verbose=2)
    seconds = time.time() - start
    recon = model.predict(x_test, batch_size=500, verbose=0)
    metrics = evaluate(recon, x_test)
    metrics.update(run=run, seconds=seconds, params=model.count_params(),
                   train_samples=len(x_fit), epochs=args.epochs)
    (OUT / f"{run}_metrics.json").write_text(json.dumps(metrics, indent=1))
    np.save(OUT / f"{run}_recon_first100.npy", recon[:100])
    model.save_weights(OUT / f"{run}.weights.h5")
    save_grid(OUT / f"{run}_grid.png", x_test, recon)
    print({k: v for k, v in metrics.items() if k != "spectrum"}, flush=True)


def run_baselines():
    from sklearn.decomposition import PCA

    OUT.mkdir(exist_ok=True)
    x_train, x_test = load_data()
    result = {"original": {
        "sharpness": sharpness(x_test[:5000]),
        "nyquist_power": checker_energy(x_test),
        "spectrum": radial_spectrum(x_test).tolist(),
    }}
    v3 = models.load_model(NOTEBOOKS / "models" / "autoencoder_v3.keras", compile=False)
    recon = v3.predict(x_test, batch_size=500, verbose=0)
    result["v3"] = evaluate(recon, x_test)
    save_grid(OUT / "v3_grid.png", x_test, recon)
    flat_train = x_train[:45000].reshape(45000, -1)
    flat_test = x_test.reshape(len(x_test), -1)
    for dim in (256, 512):
        pca = PCA(dim, svd_solver="randomized", random_state=42).fit(flat_train)
        recon = np.clip(pca.inverse_transform(pca.transform(flat_test)), 0, 1)
        recon = recon.reshape(x_test.shape).astype("float32")
        result[f"pca{dim}"] = evaluate(recon, x_test)
        save_grid(OUT / f"pca{dim}_grid.png", x_test, recon)
    (OUT / "baselines.json").write_text(json.dumps(result, indent=1))
    for k, v in result.items():
        print(k, {a: b for a, b in v.items() if a != "spectrum"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", choices=["R0", "R1", "R2", "R3", "A1", "A2", "baselines"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--train-samples", type=int, default=45000)
    p.add_argument("--timing", type=int, default=0)
    p.add_argument("--init-from", default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--widths", type=int, nargs=3, default=[32, 64, 128])
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--vgg-weight", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=0.01)
    p.add_argument("--vgg-size", type=int, default=64)
    p.add_argument("--vgg-layers", nargs="+", default=["block1_conv2", "block2_conv2", "block3_conv3"])
    p.add_argument("--r3-loss", choices=["R1", "R2"], default="R1")
    args = p.parse_args()
    if args.run == "baselines":
        run_baselines()
    else:
        run_training(args.run, args)
