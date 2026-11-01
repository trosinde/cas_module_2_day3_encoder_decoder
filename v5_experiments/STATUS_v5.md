# Autoencoder v5: aktueller Stand

Stand 2026-11-01, 11:00 Uhr. Diese Datei ersetzt die Statustabelle in `PLAN_v5.md`, die auf dem Stand von 09:00 Uhr ist.

- Alle Läufe (R0 bis R3, A1, A2) sind auf dem CPU-Rechner gelaufen, die Ergebnisse liegen in `runs/`.
- Das Notebook `autoencoder_on_CIFAR-10_v5_sharp_decoder.ipynb` ist vollständig ausgeführt, die Bewertung steht in seinem Fazit.
- Offen ist der längere Lauf auf einem schnelleren Rechner. Die Befehle dafür stehen in `PLAN_v5.md` im Abschnitt „Fortsetzen“.
- `gather.py` rechnet Bilder und Cluster aller Versionen für die Erklärseite neu und erwartet die Gewichte `runs/R0.weights.h5` und `runs/R1.weights.h5`. Diese Gewichte wurden nicht in den Nextcloud-Ordner kopiert.
