#!/usr/bin/env bash
# Phase 4 für den GAN-Sieger: erst Real-ESRGAN (Sekunden), dann Unsharp Mask (rund 30 Minuten auf der CPU).
cd "$(dirname "$0")"
source ~/.venvs/ae_v6/bin/activate
log() { echo "$1 $(date +%H:%M:%S)" >> runs/queue_status.txt; }
python -c "import numpy as np; np.save('runs/sr_input_$1.npy', np.load('runs/$1_recon_first100.npy')[:10])"
~/.venvs/ae_v6_sr/bin/python phase4b_sr.py runs/sr_input_$1.npy runs/sr_output_$1.npy > runs/sr_$1.log 2>&1
log "sr $1 exit $?"
CUDA_VISIBLE_DEVICES="" nice -n 10 python phase4a.py $1 > runs/$1_phase4a.log 2>&1
log "phase4a $1 exit $?"
