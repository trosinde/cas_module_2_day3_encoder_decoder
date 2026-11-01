#!/usr/bin/env bash
# B4 mit mehr Grafikspeicher: TensorFlow reserviert sonst nur rund 2,1 GB, B4 braucht rund 2,25 GB.
cd "$(dirname "$0")"
source ~/.venvs/ae_v6/bin/activate
log() { echo "$1 $(date +%H:%M:%S)" >> runs/queue_status.txt; }
TF_FORCE_GPU_ALLOW_GROWTH=false AE_GPU_MEM_MB=3400 python exp.py R3 --r3-loss R2 --vgg-size 48 --gan-weight 0.01 --epochs 25 --widths 64 128 256 --tag B4 > runs/B4.log 2>&1
log "B4 exit $?"
CUDA_VISIBLE_DEVICES="" nice -n 10 python perceptual.py B4 > runs/B4_perceptual.log 2>&1
log "perceptual B4 exit $?"
