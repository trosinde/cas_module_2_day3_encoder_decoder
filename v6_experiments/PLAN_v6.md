# Plan: Autoencoder v6

Stand 2026-11-01. Dieser Plan ist so geschrieben, dass er ohne die frühere Unterhaltung auf einem anderen Rechner ausführbar ist. Noch ist nichts davon gelaufen.

## Kurz zum Start

```bash
cd "<Nextcloud>/Notizen/Notability/00_CAS Generative AI/module 2 - day 3/notebooks/v6_experiments"
python3.12 -m venv ~/.venvs/ae_v6
source ~/.venvs/ae_v6/bin/activate
pip install -r requirements.txt
pip install "tensorflow[and-cuda]==2.21.0"   # nur auf Linux mit NVIDIA-GPU
python check_setup.py
```

`check_setup.py` muss mit `ERGEBNIS: OK` enden und die GPU anzeigen. Danach an Claude Code: „Setze `v6_experiments/PLAN_v6.md` um.“

**Vorher:**
- Auf dem alten Rechner warten, bis Nextcloud fertig synchronisiert hat. `check_setup.py` meldet fehlende Dateien.
- Betriebssystem beachten: TensorFlow nutzt die GPU unter Windows nur über WSL2, auf dem Mac ist die GPU-Unterstützung separat zu prüfen.
- Ohne NVIDIA-GPU ist Google Colab eine Alternative. Dann den Ordner `notebooks` einbinden und die Python-Befehle dort ausführen.

## Ausgangslage

Die Erklärung aller bisherigen Versionen steht auf https://claude.ai/artifact/HvTcvwZd3tdFHB2sB4nnF6, die Details stehen im Fazit von `../autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb`.

v5 hat die Fraktale beseitigt, ist aber nicht fotoscharf. Werte auf dem Testset mit 10.000 Bildern:

| Kennzahl | v5 | Vergleichswert |
|---|---|---|
| Abweichung (MSE) | 0,00216 | PCA 512: 0,00102 |
| Ähnlichkeit (SSIM) | 0,903 | PCA 512: 0,948 |
| Raster (Nyquist-Leistung) | 0,0075 | Original: 0,0561 |
| Klassen-Treffer (k-NN) | 23,7 % | v3: 33,8 % |

v5 hatte nur 1.500 Trainingsschritte auf der CPU. Solange unklar ist, wie viel längeres Training bringt, lassen sich weitere Änderungen nicht sauber bewerten.

**Offene Entscheidung:** Der Nutzer hat das Hauptziel noch nicht festgelegt. Falls nichts anderes gesagt wird, gilt die Empfehlung: Phase 1 und 2, danach 4a und 4b. Phase 3 nur auf Wunsch.

## Dateien

| Datei | Zweck |
|---|---|
| `v6_experiments/exp.py` | Laufskript, Kopie aus v5 mit `--tag` für eigene Ergebnisnamen. Schreibt nach `v6_experiments/runs/` |
| `v6_experiments/check_setup.py` | Startprüfung, einschließlich Nachrechnen der v5-MSE |
| `v6_experiments/requirements.txt` | Paketversionen des alten Rechners |
| `v5_experiments/runs/` | Alle v5-Ergebnisse und die Gewichte von R0, R1, R3 |
| `v5_experiments/build_nb.py` | Baut das v5-Notebook, Vorlage für das v6-Notebook |
| `v5_experiments/gather.py`, `feat.py` | Rechnen Bilder, Cluster und Merkmalskarten für die Erklärseite |
| `v5_experiments/article/` | Quellen der Erklärseite. `template.html` und `build_page.py` ergeben die Seite, im Ordner `article` ausführen |

## Phase 0: Schrittzeit messen

```bash
python exp.py R1 --timing 20
python exp.py R2 --timing 20
```

Daraus das Budget ableiten und nennen, bevor längere Läufe starten. Zum Vergleich auf der alten CPU: R1 1,03 s und R2 3,92 s pro Schritt. Bei 11.250 Schritten wären das auf der CPU rund 2,5 Stunden pro Lauf bei der Geschwindigkeit des v5-Notebooks.

## Phase 1: Basislinie neu bestimmen

Gleicher Prüfer wie v5 (L1 + SSIM), 512 Zahlen, 45.000 Bilder, 25 Epochen.

```bash
python exp.py R3 --r3-loss R1 --epochs 25 --tag B0
python exp.py R3 --r3-loss R1 --epochs 25 --widths 64 128 256 --tag B1
```

- **B0:** Holt v5 durch längeres Training die PCA ein?
- **B1:** Bremst die Modellgröße noch?

Der bessere Lauf wird die neue Basis. Phase 2 läuft in jedem Fall, das Ergebnis zeigt aber, wie viel Schärfe ohne erfundene Details erreichbar ist.

## Phase 2: Mehr sichtbare Schärfe

Auf der besten Basis, je gleiches Budget:

| Lauf | Prüfer | Frage |
|---|---|---|
| B2 | + normierter VGG-Prüfer, voller Lauf | Bringt VGG auf dem neuen Decoder doch etwas? v5 hat das nur 300 Schritte getestet |
| B3 | + Adversarial-Prüfer (PatchGAN) | Wie viel Schärfe bringt ein Fälschungs-Erkenner? |
| B4 | beides | Ergänzen sie sich? |

**Umsetzung:**
- B2 läuft mit `make_r2_loss` aus `exp.py`: `python exp.py R3 --r3-loss R2 --epochs 25 --tag B2`, bei Bedarf mit angepasster `--widths`.
- B3 und B4 brauchen neuen Code in `exp.py`: ein kleines Diskriminator-Netz und einen eigenen `train_step`, der abwechselnd beide Netze trainiert.
- Startwerte für den GAN-Term, jeweils ungetestet: Hinge-Loss, Gewicht 0,1, erst nach den ersten 30 % der Schritte zuschalten. Wird das Training instabil, zuerst das Gewicht senken.
- Für `--widths 64 128 256` gilt: Der Lauf mit der besten Basis bestimmt die Breiten der folgenden Läufe.

**Neue Messungen, ergänzend zu SSIM, MSE, Raster und Zoom-Bildern:**
- **Wahrgenommene Ähnlichkeit:** LPIPS. Das übliche Paket braucht PyTorch. Ohne PyTorch den normierten VGG-Abstand aus `make_r2_loss` nutzen und ihn im Notebook nicht LPIPS nennen.
- **Natürlichkeit:** KID über Inception-V3-Merkmale aus `keras.applications`, gegen die echten Testbilder.

**Zu erwarten:** Mit dem Adversarial-Prüfer wirken die Bilder schärfer, SSIM und MSE werden aber vermutlich schlechter. Das Modell malt dann glaubwürdige Details, die im Original so nicht stehen müssen. Darum werden beide Varianten getrennt bewertet.

**Risiko:** GANs sind instabil und können Raster zurückbringen. Das Raster bleibt Pflichtmessung.

## Phase 3: Klassen trennen

Eigenes Modell und eigenes Ziel, damit der Zielkonflikt messbar bleibt:

- Zusätzlicher Klassifikationskopf auf dem Engpass, mit gewichtetem Kreuzentropie-Loss. `y_train` wird dafür gebraucht, `exp.py` lädt die Labels bisher nicht.
- Der Kopf bekommt die 8×8-Karte über Global Average Pooling, denn sie ist räumlich aufgebaut.
- Messung: k-NN wie in `gather.py` und die Treffer des Kopfes selbst, dazu der Verlust bei SSIM.
- Drei Gewichte testen, um die Kurve Klassen gegen Bildqualität zu zeigen.

## Phase 4: Schärfer als das Original

Getrennt vom Autoencoder, weil das Ziel ein anderes ist. Der Autoencoder selbst kann sein Original nie übertreffen.

| Schritt | Methode | Ergebnis |
|---|---|---|
| 4a | Unsharp Mask, ein klassischer Schärfefilter | Kanten wirken knackiger, neue Information entsteht nicht. Ab einer gewissen Stärke Lichthöfe |
| 4b | Fertiges Super-Resolution-Modell, z. B. Real-ESRGAN. Verfügbarkeit und Lizenz vorher prüfen | 32×32 auf 128×128. Details sind erfunden |
| 4c | Eigenes Training mit STL-10 (96×96): verkleinern auf 32×32, zurück auf 96×96 lernen | Es gibt ein echtes hochaufgelöstes Original, daher ist messbar, ob die Details stimmen |

CIFAR-10 hat keine höher aufgelöste Fassung. Nur 4c erlaubt deshalb eine Messung gegen die Wahrheit. STL-10 ist nicht in `keras.datasets` enthalten und muss separat geladen werden.

## Lieferung

1. Notebook `autoencoder_on_CIFAR-10_v6_<thema>.ipynb` im Ordner `notebooks`, aufgebaut wie v5, in frischem Kernel vollständig ausgeführt, mit Changelog als erster Zelle und Fazit als letzter.
2. Ergebnisse in `v6_experiments/runs/`.
3. Erklärseite um v6 ergänzen:
   - In Claude Code mit `Artifact`, Aktion `read` und der URL oben zuerst die aktuelle Fassung lesen.
   - Dann mit derselben URL veröffentlichen, damit der Link gleich bleibt. Das geht nur mit demselben claude.ai-Konto.
   - v6 im selben Raster wie v1 bis v5 ergänzen: Überlegung, Änderung, Ergebnis, Bilder, Cluster.
4. `v1` bis `v5` nicht verändern.

## Kriterien

Testset, festgelegt vor den Läufen:

| Kriterium | Messung | Marke |
|---|---|---|
| Keine Fraktale | Raster, Zoom-Bilder ansehen | nicht über dem Original (0,0561) |
| Treue-Modell | MSE, SSIM | besser als v5, Ziel PCA 512 |
| Natürlichkeits-Modell | wahrgenommene Ähnlichkeit, KID, Bilder ansehen | besser als die Treue-Variante |
| Klassen-Modell | k-NN | über 33,8 % |

Wird ein Kriterium verfehlt, steht das so im Fazit, mit den gemessenen Zahlen.

## Nicht in v6

- **Architektur-Teile einzeln messen:** Upsampling, Karte, Residual-Blöcke und Normalisierung. Interessant, hilft dem Ergebnis aber nicht direkt.
- **Kontrastives Lernen (SimCLR):** nur falls der Klassifikationskopf nicht ausreicht.
