# Vom Farbklumpen zum Frosch

Ein Autoencoder auf CIFAR-10, Schritt für Schritt verbessert. Entstanden im CAS Generative AI, Modul 2, Tag 3.

[![Vorschau der Erklärseite: anklicken für die interaktive Fassung](docs/vorschau.png)](https://claude.ai/artifact/HvTcvwZd3tdFHB2sB4nnF6)

**[Zur interaktiven Erklärseite](https://claude.ai/artifact/HvTcvwZd3tdFHB2sB4nnF6)**

## Schnellstart

Die Notebooks laden fertig trainierte Modelle aus `models/`. Niemand muss neu trainieren, eine GPU ist nicht nötig.

```bash
git clone https://github.com/trosinde/cas_module_2_day3_encoder_decoder.git
cd cas_module_2_day3_encoder_decoder
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Zum Neutrainieren in den Parametern eines Notebooks `TRAIN_MODEL = True` setzen.

## Notebooks

| Notebook | Inhalt |
|---|---|
| `autoencoder` | Kursbeispiel auf Fashion-MNIST |
| `autoencoder_on_CIFAR-10` | Ausgangspunkt der Übung. Bricht mit dem ursprünglichen Fehler ab, v1 enthält die Korrektur |
| `..._v1_bugfix` | Übung repariert, Engpass 2 Zahlen |
| `..._v2_vae` | Umbau zum VAE |
| `..._v3_latent_quality` | Engpass 32, tieferes Modell, MSE |
| `..._v4_perceptual_loss` | VGG-Prüfer, Fraktale, bester Klassen-Treffer |
| `..._v4b_nachlauf_256` | Nachlauf mit 256 Zahlen, Vergleichswert für v5 |
| `..._v5_sharp_decoder` | Neuer Decoder, Karte als Engpass, L1 + SSIM |

v6 folgt nach Abschluss der Läufe. Skripte und Messwerte liegen in `v5_experiments/` und `v6_experiments/`.
