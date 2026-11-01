"""Trägt in die Notebooks v1 bis v6 ein, wo die Gewichte der Note festgelegt und verwendet werden.

Aufruf: add_weight_notes.py <Ordner mit den Notebooks>. Mehrfach ausführbar. Ausgaben bleiben erhalten,
es kommen nur eine Markdown-Zelle und Kommentare hinzu. Die Anteile stammen aus note_terms.py.
"""
import hashlib
import json
import sys
from pathlib import Path

HEAD = "### Gewichtung der Note: wo festgelegt, wo verwendet\n\n"
TABLE = "| Gewicht | Wert | Festgelegt | So ermittelt | Verwendet |\n|---|---|---|---|---|\n"
SEARCH = "Im Code sind die Stellen mit `[Gewicht]` markiert.\n\n"
SOURCE = "\n\nAnteile: erste 1.000 Testbilder, berechnet mit `v6_experiments/note_terms.py`, Werte in `v6_experiments/runs/note_terms.json`."

NOTES = {
    "autoencoder_on_CIFAR-10_v1_bugfix.ipynb": {
        "before": 'loss="binary_crossentropy"',
        "note": HEAD + "**Es gibt kein Gewicht.** Die Note ist allein `binary_crossentropy`, festgelegt im `compile`-Aufruf direkt unter diesem Text. "
        "Sie stammt unverändert aus der Übung.\n\nDie Note sinkt von 0,693 am frisch initialisierten Modell auf 0,634." + SOURCE,
        "comments": [],
    },
    "autoencoder_on_CIFAR-10_v2_vae.ipynb": {
        "before": "class VAE(models.Model):",
        "note": HEAD + SEARCH + TABLE
        + "| `BETA` (Pixelvergleich gegen KL) | 500 | Abschnitt 0, Parameter | Nicht ermittelt: derselbe Wert wie im Kursbeispiel `vae.ipynb`. Keine Testreihe, kein Annealing, der Wert bleibt im ganzen Training gleich | `train_step` direkt unter diesem Text, in `reconstruction_loss` |\n\n"
        "`reconstruction_loss` ist BETA × BCE (Durchschnitt über alle Pixel). `kl_loss` wird über die 2 Code-Zahlen summiert und über den Stapel gemittelt. "
        "`total_loss` addiert beide.\n\n"
        "**Anteile laut Trainingslog:** BCE 99,3 % und KL 0,7 % im ersten Durchgang, 99,0 % und 1,0 % im fünften. Die KL-Strafe zählt also kaum.",
        "comments": [
            ("BETA = 500  # VAE-Ergänzung: Gewicht des Rekonstruktions-Loss relativ zum KL-Loss",
             "BETA = 500  # VAE-Ergänzung: Gewicht des Rekonstruktions-Loss relativ zum KL-Loss [Gewicht: festgelegt]"),
            ("            total_loss = reconstruction_loss + kl_loss\n",
             "            # [Gewicht: verwendet] BETA steckt in reconstruction_loss, hier entsteht die Summe.\n            total_loss = reconstruction_loss + kl_loss\n"),
        ],
    },
    "autoencoder_on_CIFAR-10_v3_latent_quality.ipynb": {
        "before": 'autoencoder.compile(optimizer="adam", loss="mse")',
        "note": HEAD + "**Es gibt kein Gewicht.** Die Note ist allein `mse`, festgelegt im `compile`-Aufruf direkt unter diesem Text.\n\n"
        "Die Note sinkt von 0,0636 am frisch initialisierten Modell auf 0,0112." + SOURCE,
        "comments": [],
    },
    "autoencoder_on_CIFAR-10_v4_perceptual_loss.ipynb": {
        "before": "def perceptual_loss(y_true, y_pred):",
        "note": HEAD + SEARCH + TABLE
        + "| `PERCEPTUAL_WEIGHT` (VGG gegen MSE) | 0,00001 | Abschnitt 0, Parameter | Von Hand gesetzt, keine Testreihe. Der Code nennt ihn einen Platzhalter | Letzte Zeile von `perceptual_loss` direkt unter diesem Text, aktiv über `compile` in Abschnitt 3 |\n\n"
        "**Die Diagnosezelle direkt über diesem Text zeigt die Wirkung:** Am untrainierten Modell ist die MSE 0,0606 und der gewichtete VGG-Teil 3,36, also rund 55-mal so groß. "
        "Der Wert gleicht die beiden Teile demnach nicht aus. Er blieb trotzdem für diesen Lauf.\n\n"
        "**Anteile:** VGG stellt am Start 98,1 % der Summe (50-mal die MSE auf diesen Testbildern), am Ende 98,7 % (78-mal).\n\n"
        "**Die drei VGG-Stellen** zählen laut Formel gleich. Weil ihre Rohwerte verschieden groß sind, stellt am Ende `block3_conv3` rund 54 %, `block2_conv2` rund 41 % und `block1_conv2` rund 5 % des VGG-Werts." + SOURCE,
        "comments": [
            ("PERCEPTUAL_WEIGHT = 1e-5  #", "PERCEPTUAL_WEIGHT = 1e-5  # [Gewicht: festgelegt]"),
            ("    return pixel_mse + PERCEPTUAL_WEIGHT * feature_mse",
             "    # [Gewicht: verwendet]\n    return pixel_mse + PERCEPTUAL_WEIGHT * feature_mse"),
        ],
    },
    "autoencoder_on_CIFAR-10_v4b_nachlauf_256.ipynb": {
        "before": "def perceptual_loss(y_true, y_pred):",
        "note": HEAD + SEARCH + TABLE
        + "| `PERCEPTUAL_WEIGHT` (VGG gegen MSE) | 0,000001 | Abschnitt 0, Parameter. Im Lauf R0 fest in `make_v4_loss(1e-6)`, `v6_experiments/exp.py` Zeile 302 | Ungetesteter Startwert: ein Zehntel des v4-Werts, weil der dort das 55-Fache ergab | Letzte Zeile von `perceptual_loss` direkt unter diesem Text, aktiv über `compile` in Abschnitt 3 |\n\n"
        "**Die Diagnosezelle direkt über diesem Text zeigt die Wirkung:** MSE 0,0606, gewichteter VGG-Teil 0,336, also rund 5,5-mal so groß.\n\n"
        "**Anteile im Lauf R0** (festes Prüfbild-Paket aus `v5_experiments/runs/R0_terms.json`): Nach Durchgang 1 ist VGG 4,7-mal so groß wie die MSE, am Ende 9,6-mal. VGG dominiert also weiterhin.",
        "comments": [
            ("PERCEPTUAL_WEIGHT = 1e-6  #", "PERCEPTUAL_WEIGHT = 1e-6  # [Gewicht: festgelegt]"),
            ("    return pixel_mse + PERCEPTUAL_WEIGHT * feature_mse",
             "    # [Gewicht: verwendet]\n    return pixel_mse + PERCEPTUAL_WEIGHT * feature_mse"),
        ],
    },
    "autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb": {
        "before": "def l1_ssim_loss(y_true, y_pred):",
        "note": HEAD + SEARCH + TABLE
        + "| `SSIM_WEIGHT` (SSIM-Teil gegen L1) | 0,2 | Abschnitt 0, Parameter | Gesetzt, keine Testreihe. Der Plan nennt nur die Formel, die Herkunft der 0,2 ist nicht dokumentiert | Letzte Zeile von `l1_ssim_loss` direkt unter diesem Text, aktiv über `compile` in Abschnitt 3 |\n\n"
        "In den Vergleichsläufen A1, R1 und R3 ist es derselbe Wert: `--ssim-weight`, Standard 0,2.\n\n"
        "**SSIM** nutzt im Training Fenster von 7×7 Pixeln (`filter_size=7`) mit Gauß-Gewicht der Breite 1,5 (Standardwert von `tf.image.ssim`).\n\n"
        "**Anteile:** L1 58,4 % und SSIM 41,6 % am Start, 60,5 % und 39,5 % am Ende. Keiner der beiden dominiert." + SOURCE,
        "comments": [
            ("SSIM_WEIGHT = 0.2", "SSIM_WEIGHT = 0.2  # [Gewicht: festgelegt]"),
            ("    return l1 + SSIM_WEIGHT * (1.0 - ssim)",
             "    # [Gewicht: verwendet]\n    return l1 + SSIM_WEIGHT * (1.0 - ssim)"),
        ],
    },
    "autoencoder_on_CIFAR-10_v6_sharpness.ipynb": {
        "after": "## 2a. Die beiden Prüfer",
        "note": HEAD + "Die Gewichte stehen in `v6_experiments/exp.py`. Die Befehle der Läufe stehen im Abschnitt 3, dort mit `[Gewicht]` markiert.\n\n" + TABLE
        + "| SSIM-Teil | 0,2 | `--ssim-weight`, Zeile 470 | Unverändert aus v5, keine Testreihe | `l1_ssim_terms`, Zeile 125 |\n"
        "| VGG-Teil (nur B2, B4) | 0,1 | `--vgg-weight`, Zeile 471 | Standardwert, keine Testreihe | `make_r2_loss`, Zeile 154 |\n"
        "| Glättung (nur B2, B4) | 0,01 | `--tv-weight`, Zeile 472 | Standardwert, keine Testreihe | `make_r2_loss`, Zeile 157 |\n"
        "| GAN-Teil (nur B3, B4) | 0,01 | `--gan-weight` im Befehl, Zeile 476 | Im Versuch: Startwert 0,1 aus `PLAN_v6.md` kippte, 0,01 blieb stabil. Zwischenwerte nicht getestet | `GanTrainer.train_step`, Zeile 224 |\n\n"
        "**Beleg für das GAN-Gewicht:** Mit 0,1 stieg `val_loss` nach dem Zuschalten von 0,0317 auf 0,3096, der Lauf wurde abgebrochen (`v6_experiments/runs/B3_w0.1_abgebrochen.log`). "
        "Mit 0,01 stieg er nur kurz von 0,0312 auf 0,0392 und endete bei 0,0291 (`v6_experiments/runs/B3_history.json`). `val_loss` enthält bei diesen Läufen nur L1 und SSIM.\n\n"
        "**Der Fälschungs-Erkenner** hat eine eigene Note (Hinge-Loss, Zeile 221) und einen eigenen Optimierer (Adam, Lernrate 0,0002, `beta_1=0.5`, Zeile 342). "
        "Er und der GAN-Teil sind in den ersten 30 % der Schritte abgeschaltet (`--gan-start`, Zeile 477, also 3.375 von 11.250 Schritten). Danach lernen beide Netze im selben Schritt je einmal.\n\n"
        "**Anteile am Ende:**\n\n"
        "| Lauf | L1 | SSIM | VGG | Glättung | GAN |\n|---|---|---|---|---|---|\n"
        "| B1 | 69,8 % | 30,2 % | | | |\n"
        "| B2 | 60,2 % | 26,6 % | 9,8 % | 3,4 % | |\n"
        "| B3 | 62,5 % | 27,8 % | | | 9,7 % |"
        + SOURCE + " Der GAN-Anteil kommt aus dem Trainingslog des letzten Durchgangs, weil der Erkenner nicht gespeichert wurde.",
        "comments": [
            ("COMMANDS = {\n",
             "# [Gewicht] Das GAN-Gewicht steht hier als --gan-weight, die übrigen Gewichte sind Standardwerte in exp.py.\nCOMMANDS = {\n"),
        ],
    },
}


def lines(text):
    return text.splitlines(keepends=True)


folder = Path(sys.argv[1])
for name, spec in NOTES.items():
    path = folder / name
    raw = path.read_text(encoding="utf8")
    nb = json.loads(raw)
    cells = nb["cells"]
    if any("Gewichtung der Note" in "".join(c["source"]) for c in cells):
        print(name, "schon eingetragen")
        continue
    for old, new in spec["comments"]:
        hits = [c for c in cells if c["cell_type"] == "code" and old in "".join(c["source"])]
        if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
            raise SystemExit(f"{name}: {old[:60]!r} nicht eindeutig")
        hits[0]["source"] = lines("".join(hits[0]["source"]).replace(old, new))
    marker = spec.get("before") or spec["after"]
    kind = "code" if "before" in spec else "markdown"
    pos = [i for i, c in enumerate(cells) if c["cell_type"] == kind and marker in "".join(c["source"])]
    if len(pos) != 1:
        raise SystemExit(f"{name}: Stelle {marker!r} nicht eindeutig: {pos}")
    at = pos[0] + (0 if "before" in spec else 1)
    cell = {"cell_type": "markdown", "metadata": {}, "source": lines(spec["note"])}
    if "id" in cells[0]:
        cell = {"cell_type": "markdown", "id": hashlib.sha1(name.encode()).hexdigest()[:8], "metadata": {}, "source": lines(spec["note"])}
    cells.insert(at, cell)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf8")
    print(name, "eingetragen an Zelle", at)
