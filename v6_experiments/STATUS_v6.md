# Autoencoder v6: aktueller Stand

Dient zum Fortsetzen nach einem Abbruch. Fortschritt der Warteschlangen steht in `runs/queue_status.txt`.

- **Ziel laut Nutzer:** Schärfe, also Phase 0, 1, 2, 4a und 4b aus `PLAN_v6.md`. Phase 3 und 4c entfallen.
- **Basis:** B1 (Breite 64/128/256), gewählt über `val_loss` 0,02646 gegen B0 0,02948.
- **Abweichung vom Plan:** Breites Modell mit VGG auf 64×64 sprengt den Grafikspeicher, B2 und B4 laufen mit `--vgg-size 48`.
- **GAN-Gewicht:** 0,1 war instabil, `val_loss` stieg nach dem Zuschalten von 0,0317 auf 0,2165 bis 0,3096 (`runs/B3_w0.1_abgebrochen.log`). B3 läuft mit 0,01.
- **B4:** `run_after_b2.sh` wartet auf das Ende von B2, testet die XLA-Korrektur im Kurzlauf und startet B4 mit Gewicht 0,01. Von Hand: `python exp.py R3 --r3-loss R2 --vgg-size 48 --gan-weight <Gewicht> --epochs 25 --widths 64 128 256 --tag B4 > runs/B4.log 2>&1`
- **Danach für jeden Lauf:** `perceptual.py <tag>`. Unsharp Mask und Real-ESRGAN zusätzlich auf den GAN-Sieger.
- **Umgebungen:** `~/.venvs/ae_v6` (TensorFlow, GPU), `~/.venvs/ae_v6_sr` (PyTorch, CPU).
- **Läufe starten:** mit `setsid nohup`, sonst sterben sie mit der Claude-Sitzung.
- **Erklärseite, gelber Rand:** Erledigt für alle 10 Bilder und v1 bis v5, siehe `sample_points.py`. Die Cluster von v2 bis v5 sind dafür neu berechnet (`points` und `sample` in `article/data.json`). Für v6 dieselbe Rechnung ergänzen, damit `versions.v6.sample` existiert, sonst bricht das Zeichnen des Rands dort ab.
- **Noch offen:** Notebook `autoencoder_on_CIFAR-10_v6_sharpness.ipynb`, Ergänzung der Erklärseite https://claude.ai/artifact/HvTcvwZd3tdFHB2sB4nnF6
