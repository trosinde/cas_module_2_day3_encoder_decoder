"""Stellt das öffentliche Repo zusammen: Notebooks laden trainierte Modelle, statt neu zu trainieren.

Aufruf: prepare_release.py <Zielordner> [--train-course]
Mit --train-course bleibt TRAIN_MODEL in den beiden Kurs-Notebooks True, um deren Modelle einmalig zu erzeugen.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
DEST = Path(sys.argv[1])
TRAIN_COURSE = "--train-course" in sys.argv


def ae(path):
    return f'autoencoder.set_weights(models.load_model("./models/{path}", compile=False).get_weights())'


def enc_dec(enc, dec):
    objs = 'custom_objects={"Sampling": Sampling}, safe_mode=False'
    return (f'encoder.set_weights(models.load_model("./models/{enc}", compile=False, {objs}).get_weights())\n'
            f'decoder.set_weights(models.load_model("./models/{dec}", compile=False, {objs}).get_weights())')


UUID_FIT, UUID_SAVE = "b3c497b7-fa40-48df-b2bf-541239cc9400", "edb847d1-c22d-4923-ba92-0ecde0f12fca"
NOTEBOOKS = {
    "autoencoder.ipynb": dict(fit=UUID_FIT, save=UUID_SAVE, load=ae("autoencoder_course.keras"), course=True,
                              names={"autoencoder.keras": "autoencoder_course.keras", "encoder.keras": "encoder_course.keras",
                                     "decoder.keras": "decoder_course.keras"}),
    "autoencoder_on_CIFAR-10_v1_bugfix.ipynb": dict(fit=UUID_FIT, save=UUID_SAVE, load=ae("autoencoder_v1.keras"),
        names={"autoencoder.keras": "autoencoder_v1.keras", "encoder.keras": "encoder_v1.keras", "decoder.keras": "decoder_v1.keras"}),
    "autoencoder_on_CIFAR-10_v2_vae.ipynb": dict(fit=UUID_FIT, save=UUID_SAVE, load=enc_dec("encoder_v2.keras", "decoder_v2.keras"),
        names={"vae.keras": "vae_v2.keras", "encoder.keras": "encoder_v2.keras", "decoder.keras": "decoder_v2.keras"}),
    "autoencoder_on_CIFAR-10_v3_latent_quality.ipynb": dict(fit="fit", save="save-models", load=ae("autoencoder_v3.keras")),
    "autoencoder_on_CIFAR-10_v4_perceptual_loss.ipynb": dict(fit="fit", save="save-models", load=ae("autoencoder_v4.keras")),
    "autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb": dict(fit="fit", save="save-models", load=ae("autoencoder_v5.keras")),
}
NOTE = """> **Trainierte Modelle:** Dieses Notebook lädt das fertige Modell aus `models/` und trainiert nicht neu. Zum Neutrainieren in den Parametern `TRAIN_MODEL = True` setzen."""


def src(text):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def indent(text):
    return "\n".join("    " + l if l.strip() else l for l in text.strip("\n").split("\n"))


def patch(name, cfg):
    nb = json.load(open(SRC / name))
    for old, new in cfg.get("replace", {}).items():
        hits = 0
        for c in nb["cells"]:
            if c["cell_type"] == "code" and old in "".join(c["source"]):
                c["source"] = src("".join(c["source"]).replace(old, new))
                hits += 1
        assert hits, (name, old)
    cells = {c.get("id"): c for c in nb["cells"]}
    fit, save = cells[cfg["fit"]], cells[cfg["save"]]
    params = next(c for c in nb["cells"] if c["cell_type"] == "code" and "BATCH_SIZE =" in "".join(c["source"]))
    train = bool(cfg.get("course") and TRAIN_COURSE)
    params["source"] = src("".join(params["source"]) + f"\n\n# Eigene Modelle laden statt neu trainieren. True trainiert neu.\nTRAIN_MODEL = {train}")
    fit["source"] = src(f"if TRAIN_MODEL:\n{indent(''.join(fit['source']))}\nelse:\n{indent(cfg['load'])}")
    fit["outputs"], fit["execution_count"] = [], None
    text = "".join(save["source"])
    for old, new in cfg.get("names", {}).items():
        text = text.replace(f'"./models/{old}"', f'"./models/{new}"')
    save["source"] = src(f"if TRAIN_MODEL:\n{indent(text)}")
    nb["cells"].insert(0, {"cell_type": "markdown", "id": "trained-model-note", "metadata": {}, "source": src(NOTE)})
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    (DEST / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("gepatcht", name)


DEST.mkdir(parents=True, exist_ok=True)
for name, cfg in NOTEBOOKS.items():
    patch(name, cfg)
for name in ("autoencoder_on_CIFAR-10.ipynb",):
    shutil.copy(SRC / name, DEST / name)
shutil.copytree(SRC / "images", DEST / "images", dirs_exist_ok=True)
