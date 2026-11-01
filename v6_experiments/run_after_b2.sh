#!/usr/bin/env bash
# Wartet auf das Ende von B2 (Prozess-ID als Argument), testet die XLA-Korrektur im Kurzlauf und startet dann B4.
cd "$(dirname "$0")"
source ~/.venvs/ae_v6/bin/activate
log() { echo "$1 $(date +%H:%M:%S)" >> runs/queue_status.txt; }

tail --pid="$1" -f /dev/null
if [ -f runs/B2_metrics.json ]; then log "B2 fertig"; else log "B2 ohne Metriken beendet, siehe B2.log"; fi
(CUDA_VISIBLE_DEVICES="" nice -n 10 python perceptual.py B2 > runs/B2_perceptual.log 2>&1; log "perceptual B2 exit $?") &

SMOKE=$(mktemp -d)
timeout 600 python exp.py R3 --r3-loss R2 --vgg-size 48 --gan-weight 0.01 --widths 64 128 256 \
  --train-samples 300 --epochs 1 --tag smoke --out "$SMOKE" > runs/B4_smoke.log 2>&1
status=$?
if [ "$status" -ne 0 ] || [ ! -f "$SMOKE/smoke_metrics.json" ]; then
  log "Kurzlauf GAN gescheitert, exit $status, B4 nicht gestartet"
  rm -rf "$SMOKE"
  exit 1
fi
log "Kurzlauf GAN bestanden"
rm -rf "$SMOKE"

python exp.py R3 --r3-loss R2 --vgg-size 48 --gan-weight 0.01 --epochs 25 --widths 64 128 256 --tag B4 > runs/B4.log 2>&1
log "B4 exit $?"
CUDA_VISIBLE_DEVICES="" nice -n 10 python perceptual.py B4 > runs/B4_perceptual.log 2>&1
log "perceptual B4 exit $?"
