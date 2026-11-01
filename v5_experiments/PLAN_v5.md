# Übergabe: Autoencoder v5 auf einem schnelleren Rechner fortsetzen

Stand: 2026-11-01, 09:00 Uhr. Das v5-Notebook existiert noch nicht. Der vollständige Plan steht unten.

## Status

| Schritt                                                   | Stand                                               |
|-----------------------------------------------------------|-----------------------------------------------------|
| Basislinien v3, PCA 256, PCA 512                          | fertig, siehe `runs/baselines.json`                 |
| R1, R0, R2, R3                                            | keine Ergebnisse, auf dem neuen Rechner neu starten |
| Notebook `autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb` | noch nicht angelegt                                 |

## Gemessene Basislinien (Testset, 10.000 Bilder)

|          | MSE     | PSNR    | SSIM  | Laplace-Schärfe | Nyquist-Leistung |
|----------|---------|---------|-------|-----------------|------------------|
| Original | –       | –       | –     | 0,0344          | 0,0561           |
| v3       | 0,01125 | 19,5 dB | 0,512 | 0,0021          | 0,0414           |
| PCA 256  | 0,00259 | 25,9 dB | 0,877 | 0,0116          | 0,0050           |
| PCA 512  | 0,00102 | 29,9 dB | 0,948 | 0,0218          | 0,0065           |

- Die hohe Nyquist-Leistung von v3 passt zu einem Schachbrettmuster aus dem `Conv2DTranspose`-Decoder. Das ist ein Hinweis, kein Beweis. Für v4 fehlen Messwerte, bis R0 gelaufen ist.
- Zielmarke für Kriterium 2 ist jetzt die gemessene PCA-256-MSE von **0,00259** statt 0,00267.
- Die Annahmen in der Diagnose unten sind bis auf Punkt 8, 9 und 11 (direkt im Code ablesbar) nicht durch Läufe belegt.

## Gemessene Schrittzeiten auf dem alten Rechner (reine CPU, Batch 100)

| Lauf | Breiten   | VGG   | s/Schritt |
|------|-----------|-------|-----------|
| R0   | v4        | 32×32 | 0,91      |
| R1   | 32/64/128 | –     | 1,03      |
| R1   | 24/48/96  | –     | 0,71      |
| R2   | 32/64/128 | 64×64 | 3,92      |
| R2   | 24/48/96  | 48×48 | 2,29      |

## Abweichungen vom Plan, nur wegen der langsamen CPU

Auf dem CPU-Rechner waren 1.500 Schritte für R0, R1 und R3 (R1 und R3 mit 30.000 Bildern, 5 Epochen) und ein kurzes Nachtrainieren von R2 aus den R1-Gewichten vorgesehen. Auf einem schnelleren Rechner ist beides nicht nötig. Zuerst mit `--timing 20` die Schrittzeit messen und daraus das Budget ableiten.

## Fortsetzen

Benötigt werden `tensorflow`, `keras`, `scikit-learn`, `scipy`, `matplotlib`, `ipykernel` und `jupyter`. Der alte Rechner nutzte TF 2.21 und Keras 3.15. Der Ordner `notebooks/` muss synchronisiert sein, weil `models/autoencoder_v3.keras` gebraucht wird. Das Skript sucht ihn relativ zu seinem eigenen Ort. CIFAR-10 und die VGG-Gewichte lädt Keras bei Bedarf herunter.

```bash
cd v5_experiments
python exp.py R1 --timing 20           # zuerst prüfen: läuft es auf der GPU, wie schnell?
python exp.py baselines                # optional, Werte oben liegen schon vor
python exp.py R1 --epochs 25           # alle 45.000 Bilder
python exp.py R0 --epochs 15           # v4-Referenz, bewusst 10.000 Bilder wie in v4
python exp.py R2 --epochs 25           # voller Lauf statt Nachtraining
python exp.py R3 --epochs 25 --r3-loss R1   # R2 einsetzen, falls R2 gewinnt
```

Hinweise:

- Ergebnisse landen in `v5_experiments/runs/` (Metriken, Zoom-Raster, Gewichte, Loss-Terme pro Epoche).
- R0 hat dann weniger Schritte als R1. Für einen fairen Vergleich zusätzlich R1 mit `--train-samples 10000 --epochs 15` laufen lassen.
- Danach das Notebook nach Abschnitt „Vorgehen“ bauen und in frischem Kernel komplett ausführen.
- Der Kernelname `embeddings312` im v4-Notebook gehört zum alten Rechner und muss eventuell angepasst werden.

---

# Plan: Autoencoder v5, Rekonstruktion ohne Muster-Artefakte und mit mehr Schärfe

## Context

Nach v4 sind die Rekonstruktionen laut Nutzerbeobachtung näher am Original, zeigen aber starke fraktalartige Muster und wirken weiter unscharf.

**Datenlage:** Die v4-Datei auf der Platte hat keine gespeicherten Ausgaben, und `models/*_v4.keras` existiert nicht. Die Diagnose unten stützt sich also auf den Code und die Beschreibung, nicht auf gemessene v4-Werte. Darum wird v4 als Referenz einmal unverändert nachtrainiert.

## Diagnose: was in v4 noch ungelöst ist

| #  | Problem                                                                                                          | Stelle in v4        | Wirkung                                                                                                                                  |
|----|------------------------------------------------------------------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `Conv2DTranspose` mit Kernel 3 und Stride 2 überlappt ungleichmäßig                                              | Decoder             | Klassische Ursache für Schachbrettmuster                                                                                                 |
| 2  | VGG-Loss auf rohen Features ohne Regularisierung                                                                 | `perceptual_loss`   | Pooling-Schichten sehen bestimmte Hochfrequenzmuster kaum, also werden sie nicht bestraft. Wahrscheinlich die Verstärkung der „Fraktale“ |
| 3  | Rohe Feature-MSE, Gewicht nur am untrainierten Modell geprüft                                                    | `PERCEPTUAL_WEIGHT` | Schichten mit großen Aktivierungen dominieren. Das Verhältnis zum Pixel-Term im Training wurde nie gemessen                              |
| 4  | VGG sieht 32×32 direkt, trainiert wurde es auf 224×224                                                           | `vgg-extractor`     | Filter passen schlecht zur Objektgröße                                                                                                   |
| 5  | Engpass bei 2×2 mit `Flatten` + `Dense`                                                                          | Encoder/Decoder     | Jede Ortsinformation muss durch eine globale Schicht, was feine Details kostet                                                           |
| 6  | Nur eine Conv pro Stufe, keine Skip-Verbindungen innerhalb der Blöcke, nach dem letzten Upsampling nur eine Conv | Encoder/Decoder     | Wenig Kapazität für hohe Frequenzen                                                                                                      |
| 7  | BatchNorm im Decoder                                                                                             | Decoder             | Kann Artefakte und Abweichungen zwischen Training und Inferenz erzeugen                                                                  |
| 8  | Nur 10.000 Bilder, 15 Epochen, also etwa 1.500 Schritte                                                          | `params`            | Wahrscheinlich untertrainiert. In v4 bereits als Grenze notiert                                                                          |
| 9  | Validierung auf `x_test[:2000]`, Auswertung auf `x_test[:5000]`                                                  | `fit`               | Modellauswahl und Test überlappen. `VALIDATION_SPLIT` bleibt ungenutzt                                                                   |
| 10 | **Laplace-Schärfe belohnt Artefakte**                                                                            | Abschnitt 4a        | Ein Muster erhöht die Metrik, v4 kann also „schärfer“ wirken, obwohl es schlechter aussieht                                              |
| 11 | Plot-Titel sagen fest „32-dim“                                                                                   | PCA-/t-SNE-Zellen   | Falsche Beschriftung bei 256 Dimensionen                                                                                                 |
| 12 | Zwei Änderungen gleichzeitig (Engpass und Loss)                                                                  | Changelog v4        | Ursache nicht trennbar, darum jetzt Einzelvergleiche                                                                                     |

**Physikalische Grenze:** Bei 256 Dimensionen liegt die PCA bei MSE 0,00267 (etwa 25,7 dB), bei 512 bei 0,00107 (etwa 29,7 dB). Ohne größeren Engpass oder Adversarial-Term sind keine fotoscharfen Bilder zu erwarten. Ein GAN-Term bleibt ausgeklammert, weil er auf dieser CPU teuer und instabil wäre.

## Vorgehen

Neues Notebook `autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb` im Ordner `module 2 - day 3/notebooks/`, aufgebaut aus v4 (gleiche Struktur, deutsches Changelog als erste Zelle). v1 bis v4 bleiben unverändert.

### 1. Architektur

- **Upsampling:** `UpSampling2D` (bilinear) + `Conv2D` statt `Conv2DTranspose`.
- **Räumlicher Engpass:** 32 → 16 → 8, dann 1×1-Conv auf `LATENT_CHANNELS`. 8×8×4 ergibt wieder 256 Dimensionen, also gleich viele wie v4, aber mit erhaltener Ortsinformation. Kein `Dense`.
- **Residual-Blöcke** je Stufe (2 Convs mit Skip innerhalb des Blocks, **keine** U-Net-Skips vom Encoder zum Decoder, sonst umgeht man den Engpass).
- `GroupNormalization` statt BatchNorm.
- Ausgabe weiter `Conv2D` mit Sigmoid.

### 2. Loss

- **Hauptkandidat:** `L1 + SSIM_WEIGHT × (1 − SSIM)` über `tf.image.ssim` mit kleinem Filter. Kommt ohne vortrainiertes Netz aus und ist auf der CPU viel günstiger.
- **Zweiter Kandidat:** zusätzlich VGG-Features, kanalweise normiert wie bei LPIPS, Eingabe auf 64×64 hochskaliert, kleines Gewicht, plus Total Variation gegen Hochfrequenzmuster. Die Terme werden im Training einzeln geloggt, damit das Verhältnis sichtbar ist.

### 3. Training

- Mehr Daten: alle 45.000 Trainingsbilder, falls die gemessene Schrittzeit es erlaubt. Sonst wird die Teilmenge offen genannt.
- Horizontale Flips, Cosine-LR statt `ReduceLROnPlateau`.
- Validierung aus den letzten 5.000 Trainingsbildern, Testset nur für die Endauswertung.

### 4. Messung (Abschnitt 4a neu)

- **PSNR und SSIM** auf dem Testset, gegen PCA gleicher Größe und v3.
- **Radiales Leistungsspektrum** gegen Original: zeigt fehlende Details (Abfall) und Artefakte (Spitzen).
- **Differenzbilder** und **4-fach gezoomte Bilder** mit Nearest Neighbor, damit Muster sichtbar werden.
- Laplace-Schärfe bleibt, aber mit dem Hinweis auf Problem 10.
- Plot-Titel nutzen `EMBEDDING_DIM`.

### 5. Einzelvergleiche, je gleiches Budget und gleicher Split

| Lauf | Architektur | Loss                     | Latent | Frage                                              |
|------|-------------|--------------------------|--------|----------------------------------------------------|
| R0   | v4          | v4                       | 256    | Referenz: Lassen sich die Fraktale reproduzieren?  |
| R1   | neu         | L1 + SSIM                | 256    | Verschwinden die Muster, was kostet das an Textur? |
| R2   | neu         | R1 + normierter VGG + TV | 256    | Bringt VGG mit neuem Decoder Textur ohne Muster?   |
| R3   | neu         | bester aus R1/R2         | 512    | Wie viel bringt der größere Engpass?               |

Die Läufe laufen als Skript im Scratchpad im Hintergrund. Das v5-Notebook enthält danach den Gewinner, die Vergleichstabelle und das Fazit, und wird vollständig ausgeführt, sodass die Ausgaben gespeichert sind.

**Laufzeit:** Vor dem Start wird die Schrittzeit jedes Laufs gemessen. Ziel sind insgesamt etwa 60–90 Minuten auf der CPU (16 Threads, etwa 4 GB freier RAM). Wird das überschritten, kürze ich R0 und R2 und nenne das im Changelog. Der Rechner ist währenddessen stark ausgelastet. Parameter für einen vollen GPU-Lauf werden dokumentiert.

## Nicht in v5

- **Klassifizierung:** Ein räumlicher Engpass hilft der k-NN-Probe vermutlich nicht und kann sie sogar verschlechtern. Die Probe bleibt zur Beobachtung drin. Das Klassenproblem löst nur ein geändertes Ziel (überwachter Kopf oder kontrastiv), das wäre ein eigenes v6.
- **Adversarial-Term:** nächster Schritt, falls Schärfe danach noch der Engpass ist.

## Kritische Dateien

- Vorlage (nur lesen): `autoencoder_on_CIFAR-10_v4_perceptual_loss.ipynb`, wiederverwendet werden `display`, `preprocess`, `to_vgg_input`, die Probe aus 5b.
- Referenz: `models/autoencoder_v3.keras`.
- Neu: `autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb`, `models/*_v5.keras`, `checkpoint_v5.keras`, `logs_v5/`.
- Umgebung: `/home/tro/.venvs/embeddings312` (TF 2.21, Keras 3.15). `nbconvert` fehlt, ausgeführt wird mit `jupyter-execute`.

## Verifikation

Vorab festgelegte Kriterien, geprüft auf dem Testset:

1. **Keine sichtbaren Muster:** Ich sehe mir die gezoomten Bilder und Differenzbilder selbst an. Das Spektrum zeigt keine Spitze nahe der Nyquist-Frequenz gegenüber dem Original.
2. **Pixeltreue:** R1 unterbietet die PCA-256-MSE von 0,00267.
3. **Struktur:** SSIM über v3 und R0.
4. **Detail:** Das Spektrum liegt bei hohen Frequenzen näher am Original als bei v3.

Wird ein Kriterium verfehlt, steht das so im Fazit des Notebooks, mit den gemessenen Zahlen. Das Notebook muss in frischem Kernel von oben bis unten durchlaufen.