"""Baut das v6-Notebook aus v5. Aufruf: build_nb.py <GAN-Tag> <Zielname>

Changelog und Fazit kommen aus nb_changelog.md und nb_fazit.md, alle Zahlen aus runs/.
Das Notebook trainiert standardmäßig nicht neu, sondern lädt die Gewichte der Läufe.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent
gan_tag, target = sys.argv[1], sys.argv[2]
v5_nb = json.load(open(NB / "autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb"))
v5 = {c["id"]: c for c in v5_nb["cells"]}
meta = dict(v5_nb["metadata"])
meta["kernelspec"] = {"display_name": "Python 3 (ae_v6)", "language": "python", "name": "python3"}
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
    c = dict(v5[cell_id])
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    cells.append(c)


md("changelog-v6", (HERE / "nb_changelog.md").read_text())
for c in ("title", "task-desc", "prep-header", "display-fn"):
    keep(c)

code("imports", """
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from tensorflow.keras import datasets, models

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# v6: Architektur, Losses und Training liegen in exp.py, damit Notebook und Läufe denselben Code nutzen.
sys.path.insert(0, "./v6_experiments")
import exp
""")

keep("params-header")
code("params", f"""
IMAGE_SIZE = 32
CHANNELS = 3
BATCH_SIZE = 100

# 8x8 x LATENT_CHANNELS: 512 Dimensionen wie in v5.
LATENT_CHANNELS = 8
EMBEDDING_DIM = 8 * 8 * LATENT_CHANNELS
WIDTHS = (64, 128, 256)

TRAIN_SAMPLES = 45000
VAL_SAMPLES = 5000
EPOCHS = 25

# Die beiden Modelle dieses Notebooks.
FIDELITY_TAG = "B1"       # beste Treue
NATURAL_TAG = "{gan_tag}"  # bester Lauf mit Adversarial-Prüfer

# False lädt die Gewichte aus v6_experiments/runs. True trainiert neu, das dauert auf der GPU rund 30 Minuten je Lauf.
RETRAIN = False
RUNS = Path("./v6_experiments/runs")
""")

keep("prepare-header")
keep("load-data")
keep("preprocess")
code("split", """
# Die Validierung kommt aus dem Trainingsset, das Testset dient nur der Endauswertung.
x_val = x_train[-VAL_SAMPLES:]
x_fit = x_train[:TRAIN_SAMPLES]
""")
keep("display-sample")
keep("build-header")
code("build", """
# Architektur wie v5 (exp.build_v5), aber doppelt so breit: 3,86 statt 0,97 Mio. Parameter.
def build_autoencoder():
    return exp.build_v5(LATENT_CHANNELS, list(WIDTHS))

build_autoencoder().summary()
""")

md("loss-header", """
## 2a. Die beiden Prüfer

- **Treue-Modell:** `L1 + 0,2 × (1 − SSIM)` wie in v5.
- **Natürlichkeits-Modell:** zusätzlich ein Adversarial-Prüfer. Ein kleines zweites Netz (PatchGAN) lernt, echte Bildausschnitte von nachgemalten zu unterscheiden. Der Autoencoder lernt, es zu täuschen. Der Code steht in `exp.GanTrainer` und `exp.build_discriminator`.
""")
keep("train-header")
code("train", """
COMMANDS = {
    "B1": "R3 --r3-loss R1 --epochs 25 --widths 64 128 256 --tag B1",
    "B2": "R3 --r3-loss R2 --vgg-size 48 --epochs 25 --widths 64 128 256 --tag B2",
    "B3": "R3 --r3-loss R1 --gan-weight 0.01 --epochs 25 --widths 64 128 256 --tag B3",
    "B4": "R3 --r3-loss R2 --vgg-size 48 --gan-weight 0.01 --epochs 25 --widths 64 128 256 --tag B4",
}
if RETRAIN:
    for tag in (FIDELITY_TAG, NATURAL_TAG):
        subprocess.run([sys.executable, "exp.py", *COMMANDS[tag].split()], cwd="./v6_experiments", check=True)


def load_run(tag):
    model = build_autoencoder()
    model.load_weights(RUNS / f"{tag}.weights.h5")
    model.compile(jit_compile=False)
    return model


autoencoder = load_run(FIDELITY_TAG)
natural = load_run(NATURAL_TAG)
encoder = keras.Model(autoencoder.input, autoencoder.get_layer("latent").output)
""")

keep("reconstruct-header")
keep("example-data")
code("reconstruct", """
predictions = autoencoder.predict(x_test, batch_size=500, verbose=0)
predictions_natural = natural.predict(x_test, batch_size=500, verbose=0)

print("Example real images")
display(x_test)
print("Reconstructions, Treue-Modell")
display(predictions)
print("Reconstructions, Natürlichkeits-Modell")
display(predictions_natural)
""")

md("zoom-header", """
### 4a. Muster sichtbar machen: Zoom und Differenz

4-fach mit Nearest Neighbor vergrößert, damit kein Weichzeichnen der Anzeige Muster verdeckt. Die Differenz zum Original ist 4-fach verstärkt.
""")
code("zoom", """
rows = [
    ("Original", x_test),
    (f"Treue ({FIDELITY_TAG})", predictions),
    (f"Natürlichkeit ({NATURAL_TAG})", predictions_natural),
    ("Differenz Treue x4", np.clip(np.abs(x_test - predictions) * 4, 0, 1)),
    ("Differenz Natürlichkeit x4", np.clip(np.abs(x_test - predictions_natural) * 4, 0, 1)),
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
### 4b. Messen

Alle Werte stammen vom Testset mit 10.000 Bildern und aus den gespeicherten Ergebnissen der Läufe. Die Zeilen für `B1` und den Natürlichkeits-Lauf werden zur Kontrolle unten neu gerechnet.

- **KID:** Natürlichkeit über Inception-V3-Merkmale gegen die echten Testbilder. Niedriger ist besser.
- **VGG-Abstand:** normierter Merkmalsabstand. Gleiche Idee wie LPIPS, aber ohne dessen gelernte Gewichte, daher nicht LPIPS genannt.
- **Raster:** Nyquist-Leistung. Das Kriterium lautet: nicht über dem Original.
""")
code("metrics", """
V5_RUNS = Path("./v5_experiments/runs")
baselines = json.loads((V5_RUNS / "baselines.json").read_text())
references = json.loads((RUNS / "references_perceptual.json").read_text())


def row(tag):
    metrics = json.loads((RUNS / f"{tag}_metrics.json").read_text())
    path = RUNS / f"{tag}_perceptual.json"
    return metrics, (json.loads(path.read_text())["test"] if path.exists() else {})


table = {"PCA 512": (baselines["pca512"], references["pca512"]),
         "v5": (exp.evaluate(models.load_model("./models/autoencoder_v5.keras", compile=False).predict(x_test, batch_size=500, verbose=0), x_test), references["v5"])}
for tag in ("B0", "B1", "B2", "B3", "B4"):
    if (RUNS / f"{tag}_metrics.json").exists():
        table[tag] = row(tag)

print(f"{'':10s}{'MSE':>9s}{'PSNR':>8s}{'SSIM':>8s}{'Raster':>9s}{'Laplace':>9s}{'KID':>9s}{'VGG':>8s}")
for label, (m, p) in table.items():
    print(f"{label:10s}{m['mse']:9.5f}{m['psnr']:8.2f}{m['ssim']:8.3f}{m['nyquist_power']:9.4f}{m['sharpness']:9.4f}"
          f"{p.get('kid', float('nan')):9.4f}{p.get('vgg_distance', float('nan')):8.4f}")
original = baselines["original"]
print(f"{'Original':10s}{'':9s}{'':8s}{'':8s}{original['nyquist_power']:9.4f}{original['sharpness']:9.4f}{references['real_train']['kid']:9.4f}")

# Kontrolle: Stimmen die gespeicherten Werte mit den geladenen Gewichten überein?
for tag, recon in ((FIDELITY_TAG, predictions), (NATURAL_TAG, predictions_natural)):
    fresh, stored = exp.evaluate(recon, x_test), table[tag][0]
    print(f"Kontrolle {tag}: MSE {fresh['mse']:.5f} gegen {stored['mse']:.5f}, SSIM {fresh['ssim']:.3f} gegen {stored['ssim']:.3f}")
""")
code("spectrum-plot", """
curves = [
    ("Original", original["spectrum"], "#2a78d6", "-"),
    ("PCA 512", baselines["pca512"]["spectrum"], "#1baf7a", ":"),
    ("v5", table["v5"][0]["spectrum"], "#eda100", "-."),
    (f"Treue ({FIDELITY_TAG})", table[FIDELITY_TAG][0]["spectrum"], "#e87ba4", "-"),
    (f"Natürlichkeit ({NATURAL_TAG})", table[NATURAL_TAG][0]["spectrum"], "#eb6834", "--"),
]
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

md("unsharp-header", """
### 4c. Schärfer wirken: Unsharp Mask

Ein klassischer Filter hebt Kanten an: `Bild + Stärke × (Bild − weichgezeichnetes Bild)`. Er fügt keine Information hinzu. Berechnet mit `phase4a.py`.
""")
code("unsharp", """
unsharp = json.loads((RUNS / "B1_unsharp.json").read_text())
print(f"{'Stärke':8s}{'MSE':>9s}{'SSIM':>8s}{'Raster':>9s}{'Laplace':>9s}{'KID':>9s}{'VGG':>8s}")
for amount, m in unsharp.items():
    print(f"{amount:8s}{m['mse']:9.5f}{m['ssim']:8.3f}{m['nyquist_power']:9.4f}{m['sharpness']:9.4f}{m['kid']:9.4f}{m['vgg_distance']:8.4f}")
samples = np.load(RUNS / "B1_unsharp_first10.npz")
fig, axes = plt.subplots(len(samples.files) + 1, 10, figsize=(16, 7))
for r, (label, images) in enumerate([("Treue-Modell", predictions[:10])] + [(f"Stärke {k}", samples[k]) for k in samples.files]):
    for i in range(10):
        axes[r, i].imshow(np.kron(images[i], np.ones((4, 4, 1))), interpolation="nearest")
        axes[r, i].axis("off")
    axes[r, 0].set_title(label, loc="left", fontsize=10)
fig.tight_layout()
plt.show()
""")

md("sr-header", """
### 4d. Schärfer als das Original: Real-ESRGAN

Ein fertiges Super-Resolution-Modell (BSD-3-Clause, Xintao Wang) vergrößert von 32×32 auf 128×128, berechnet mit `phase4b_sr.py` in einer eigenen PyTorch-Umgebung.

**Die Details sind erfunden.** Zu CIFAR-10 gibt es keine höher aufgelöste Fassung, daher lässt sich nicht messen, ob sie stimmen. Die rote Schrift auf Bild 4 zeigt das gut: Sie wird scharf, aber unleserlich.
""")
code("sr", """
sr_rows = [("Original, vergrößert", np.kron(np.load(RUNS / "sr_input_original.npy"), np.ones((4, 4, 1)))),
           ("Real-ESRGAN auf Original", np.load(RUNS / "sr_output_original.npy")),
           (f"Real-ESRGAN auf {FIDELITY_TAG}", np.load(RUNS / "sr_output_B1.npy"))]
fig, axes = plt.subplots(len(sr_rows), 10, figsize=(16, 5.2))
for r, (label, images) in enumerate(sr_rows):
    for i in range(10):
        axes[r, i].imshow(images[i], interpolation="nearest")
        axes[r, i].axis("off")
    axes[r, 0].set_title(label, loc="left", fontsize=10)
fig.tight_layout()
plt.show()
""")

keep("encode-header")
keep("embeddings")
keep("embeddings-print")
keep("viz-header")
keep("pca-plot")
keep("tsne-plot")
keep("knn-header")
keep("knn-probe")
md("takeaways", (HERE / "nb_fazit.md").read_text())

out = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
(NB / target).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("wrote", NB / target, len(cells), "cells")
