"""Baut das v5-Notebook aus v4. Aufruf: build_nb.py [--smoke] <Zielname>"""
import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent
v4 = {c["id"]: c for c in json.load(open(NB / "autoencoder_on_CIFAR-10_v4_perceptual_loss.ipynb"))["cells"]}
meta = json.load(open(NB / "autoencoder_on_CIFAR-10_v4_perceptual_loss.ipynb"))["metadata"]
smoke = "--smoke" in sys.argv
target = sys.argv[-1]

cells = []


def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def md(cell_id, text):
    cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src(text)})


def code(cell_id, text):
    cells.append({"cell_type": "code", "id": cell_id, "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src(text)})


def keep(cell_id):
    c = dict(v4[cell_id])
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    cells.append(c)


md("changelog-v5", "## Changelog\n\nWird nach dem vollständigen Lauf ergänzt.")
for c in ("title", "task-desc", "prep-header", "display-fn"):
    keep(c)

code("imports", """
%load_ext autoreload
%autoreload 2

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, callbacks

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from scipy.ndimage import convolve
""")

keep("params-header")
code("params", f"""
IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 100

# Räumlicher Engpass: 8x8 x LATENT_CHANNELS. 8 Kanäle ergeben 512 Dimensionen (Lauf R3),
# 4 Kanäle ergeben 256 Dimensionen wie in v4 (Lauf R1).
LATENT_CHANNELS = 8
EMBEDDING_DIM = 8 * 8 * LATENT_CHANNELS
WIDTHS = (32, 64, 128)

# CPU-Budget: 1.500 Schritte wie in v4. Auf einer GPU TRAIN_SAMPLES = 45000 und EPOCHS = 25 setzen.
TRAIN_SAMPLES = {2000 if smoke else 30000}
VAL_SAMPLES = {500 if smoke else 5000}
EPOCHS = {1 if smoke else 5}

LEARNING_RATE = 1e-3
SSIM_WEIGHT = 0.2
""")

keep("prepare-header")
keep("load-data")
keep("preprocess")
code("split", """
# v5: Die Validierung kommt aus dem Trainingsset, das Testset dient nur der Endauswertung.
x_val = x_train[-VAL_SAMPLES:]
x_fit = x_train[:TRAIN_SAMPLES]
""")
keep("display-sample")
keep("build-header")

code("res-block", """
def res_block(x, filters):
    \"\"\"Zwei Convs mit Skip innerhalb des Blocks, GroupNormalization statt BatchNormalization.\"\"\"
    skip = x if x.shape[-1] == filters else layers.Conv2D(filters, 1)(x)
    y = layers.GroupNormalization(groups=8)(x)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    y = layers.GroupNormalization(groups=8)(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, padding="same")(y)
    return layers.Add()([skip, y])
""")
code("encoder", """
# Encoder
# v5: bleibt räumlich bis 8x8, kein Flatten und kein Dense.
encoder_input = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input"
)
x = layers.Conv2D(WIDTHS[0], 3, padding="same")(encoder_input)
x = res_block(x, WIDTHS[0])
x = layers.Conv2D(WIDTHS[1], 3, strides=2, padding="same")(x)
x = res_block(x, WIDTHS[1])
x = layers.Conv2D(WIDTHS[2], 3, strides=2, padding="same")(x)
x = res_block(x, WIDTHS[2])
x = layers.GroupNormalization(groups=8)(x)
x = layers.Activation("relu")(x)
encoder_output = layers.Conv2D(LATENT_CHANNELS, 1, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
""")
code("decoder", """
# Decoder
# v5: UpSampling2D + Conv2D statt Conv2DTranspose, um Schachbrettmuster zu vermeiden.
# Bewusst keine Skips vom Encoder, sie würden den Engpass umgehen.
decoder_input = layers.Input(shape=encoder_output.shape[1:], name="decoder_input")
x = layers.Conv2D(WIDTHS[2], 3, padding="same")(decoder_input)
x = res_block(x, WIDTHS[2])
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2D(WIDTHS[1], 3, padding="same")(x)
x = res_block(x, WIDTHS[1])
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2D(WIDTHS[0], 3, padding="same")(x)
x = res_block(x, WIDTHS[0])
x = layers.GroupNormalization(groups=8)(x)
x = layers.Activation("relu")(x)
decoder_output = layers.Conv2D(
    CHANNELS, 3, activation="sigmoid", padding="same", name="decoder_output"
)(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
""")
keep("autoencoder-model")

md("loss-header", """
## 2a. Loss: L1 + SSIM

L1 bestraft Pixelabweichungen und neigt weniger zum Mitteln als MSE. SSIM vergleicht lokale Helligkeit, Kontrast und Struktur. Beide kommen ohne vortrainiertes Netz aus, das Hochfrequenzmuster übersehen könnte.
""")
code("loss-fn", """
@keras.saving.register_keras_serializable(package="autoencoder_v5")
def l1_ssim_loss(y_true, y_pred):
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=7)
    return l1 + SSIM_WEIGHT * (1.0 - ssim)
""")

keep("train-header")
code("compile", """
# Autoencoder kompilieren
# v5: Cosine-Lernrate statt ReduceLROnPlateau.
steps_per_epoch = len(x_fit) // BATCH_SIZE
schedule = keras.optimizers.schedules.CosineDecay(
    LEARNING_RATE, steps_per_epoch * EPOCHS, alpha=0.01
)
autoencoder.compile(optimizer=keras.optimizers.Adam(schedule), loss=l1_ssim_loss)
""")
code("callbacks", """
# Checkpoint zum Speichern des Modells anlegen
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint_v5.keras",
    save_weights_only=False,
    save_freq="epoch",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs_v5")
""")
code("fit", """
# v5: horizontale Flips als Augmentierung.
train_ds = (
    tf.data.Dataset.from_tensor_slices(x_fit)
    .shuffle(len(x_fit))
    .batch(BATCH_SIZE)
    .map(tf.image.random_flip_left_right, num_parallel_calls=tf.data.AUTOTUNE)
    .map(lambda batch: (batch, batch))
    .prefetch(tf.data.AUTOTUNE)
)

autoencoder.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=(x_val, x_val),
    validation_batch_size=BATCH_SIZE,
    callbacks=[model_checkpoint_callback, tensorboard_callback],
    verbose=2,
)
""")
code("save-models", """
# Finale Modelle speichern
Path("./models").mkdir(parents=True, exist_ok=True)
autoencoder.save("./models/autoencoder_v5.keras")
encoder.save("./models/encoder_v5.keras")
decoder.save("./models/decoder_v5.keras")
""")

keep("reconstruct-header")
keep("example-data")
code("reconstruct", """
predictions = autoencoder.predict(x_test, batch_size=500, verbose=0)

print("Example real images")
display(x_test)
print("Reconstructions")
display(predictions)
""")

md("zoom-header", """
### 4a. Muster sichtbar machen: Zoom und Differenz

Die Bilder sind 4-fach mit Nearest Neighbor vergrößert, damit kein Weichzeichnen der Anzeige Muster verdeckt. Obere Zeile Original, Mitte v5, darunter v3, unten die Differenz von v5 zum Original, 4-fach verstärkt.
""")
code("zoom", """
autoencoder_v3 = models.load_model("./models/autoencoder_v3.keras", compile=False)
predictions_v3 = autoencoder_v3.predict(x_test, batch_size=500, verbose=0)

rows = [
    ("Original", x_test),
    ("v5", predictions),
    ("v3", predictions_v3),
    ("Differenz v5 x4", np.clip(np.abs(x_test - predictions) * 4, 0, 1)),
]
n = 10
fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.6, len(rows) * 1.7))
for r, (label, images) in enumerate(rows):
    for i in range(n):
        axes[r, i].imshow(np.kron(images[i], np.ones((4, 4, 1))), interpolation="nearest")
        axes[r, i].axis("off")
    axes[r, 0].set_title(label, loc="left", fontsize=10)
fig.tight_layout()
plt.show()
""")

md("metrics-header", """
### 4b. Messen: PSNR, SSIM, Schärfe und Spektrum

- **PSNR und SSIM** messen die Treue zum Original.
- **Radiales Leistungsspektrum:** Liegt eine Kurve unter dem Original, fehlen Details. Eine Anhebung am rechten Ende spricht für Artefakte.
- **Nyquist-Leistung:** Leistung bei der höchsten Frequenz, wo ein Schachbrettmuster liegen würde.
- **Laplace-Schärfe nur mit Vorsicht lesen:** Auch Muster erhöhen sie. Allein belegt sie keine bessere Bildqualität.

Die Zeilen R0 bis R3 und die Basislinien stammen aus den Einzelläufen in `v5_experiments/runs/`. Die Zeile „v5 (dieses Notebook)“ wird hier neu gemessen und weicht wegen des zufälligen Starts leicht von R3 ab.
""")
code("metrics", """
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def radial_spectrum(images):
    luma = images @ LUMA_WEIGHTS
    luma = luma - luma.mean(axis=(1, 2), keepdims=True)
    power = (np.abs(np.fft.fftshift(np.fft.fft2(luma), axes=(1, 2))) ** 2).mean(axis=0)
    yy, xx = np.indices(power.shape)
    radius = np.hypot(yy - 16, xx - 16).astype(int)
    return (np.bincount(radius.ravel(), power.ravel()) / np.bincount(radius.ravel()))[:17]


def measure(recon):
    mse = float(np.mean((x_test - recon) ** 2))
    ssim = np.concatenate([
        tf.image.ssim(x_test[i:i + 500], recon[i:i + 500], max_val=1.0).numpy()
        for i in range(0, len(x_test), 500)
    ])
    luma = recon @ LUMA_WEIGHTS
    return {
        "mse": mse,
        "psnr": -10 * np.log10(mse),
        "ssim": float(ssim.mean()),
        "sharpness": float(np.mean([convolve(l, LAPLACIAN_KERNEL, mode="reflect").var() for l in luma[:5000]])),
        "nyquist_power": float((np.abs(np.fft.fft2(luma)) ** 2)[:, 16, 16].mean()),
        "spectrum": radial_spectrum(recon),
    }


RUNS = Path("./v5_experiments/runs")
baselines = json.loads((RUNS / "baselines.json").read_text())
results = {
    "v3 (32 Dim.)": baselines["v3"],
    "PCA 256": baselines["pca256"],
    "PCA 512": baselines["pca512"],
}
RUN_LABELS = {
    "R0": "R0: v4-Referenz (256)",
    "R1": "R1: neu, L1+SSIM (256)",
    "R2": "R2: R1 + VGG-Nachtraining (256)",
    "R3": "R3: neu, 512 Dim.",
}
for run, label in RUN_LABELS.items():
    path = RUNS / f"{run}_metrics.json"
    if path.exists():
        results[label] = json.loads(path.read_text())
results["v5 (dieses Notebook)"] = measure(predictions)

print(f"{'':34s}{'MSE':>9s}{'PSNR':>8s}{'SSIM':>8s}{'Laplace':>9s}{'Nyquist':>9s}")
for label, m in results.items():
    print(f"{label:34s}{m['mse']:9.5f}{m['psnr']:8.2f}{m['ssim']:8.3f}{m['sharpness']:9.4f}{m['nyquist_power']:9.4f}")
original = baselines["original"]
print(f"{'Original':34s}{'':9s}{'':8s}{'':8s}{original['sharpness']:9.4f}{original['nyquist_power']:9.4f}")
""")
code("spectrum-plot", """
curves = [
    ("Original", original["spectrum"], "#2a78d6", "-"),
    ("v3", results["v3 (32 Dim.)"]["spectrum"], "#eb6834", "--"),
    ("PCA 512", results["PCA 512"]["spectrum"], "#1baf7a", ":"),
    ("v5 (dieses Notebook)", results["v5 (dieses Notebook)"]["spectrum"], "#e87ba4", "-"),
]
if RUN_LABELS["R0"] in results:
    curves.insert(3, ("R0: v4-Referenz", results[RUN_LABELS["R0"]]["spectrum"], "#eda100", "-."))

fig, ax = plt.subplots(figsize=(8, 4.5))
for label, spectrum, color, style in curves:
    ax.plot(range(1, 17), spectrum[1:], label=label, color=color, linestyle=style, linewidth=2)
ax.set_yscale("log")
ax.set_xlabel("Radiale Frequenz (Zyklen pro Bild, 16 = Nyquist)")
ax.set_ylabel("Mittlere Leistung (log)")
ax.set_title("Radiales Leistungsspektrum der Rekonstruktionen")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.show()
""")

keep("encode-header")
code("embeddings", """
# Beispielbilder kodieren
# v5: Das räumliche 8x8xC-Embedding wird für die Auswertung flach gemacht.
embeddings = encoder.predict(example_images, verbose=0).reshape(len(example_images), -1)
""")
keep("embeddings-print")
keep("viz-header")
for cell_id in ("pca-plot", "tsne-plot"):
    c = dict(v4[cell_id])
    c["outputs"], c["execution_count"] = [], None
    text = "".join(c["source"])
    text = text.replace('plt.title("PCA projection of the 32-dim embedding, coloured by class")',
                        'plt.title(f"PCA projection of the {EMBEDDING_DIM}-dim embedding, coloured by class")')
    text = text.replace('plt.title("t-SNE projection of the 32-dim embedding, coloured by class")',
                        'plt.title(f"t-SNE projection of the {EMBEDDING_DIM}-dim embedding, coloured by class")')
    c["source"] = src(text)
    cells.append(c)
keep("knn-header")
keep("knn-probe")
md("takeaways", "### Fazit\n\nWird nach dem vollständigen Lauf ergänzt.")

out = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
(NB / target).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", NB / target, len(cells), "cells")
