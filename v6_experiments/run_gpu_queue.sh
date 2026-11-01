#!/usr/bin/env bash
# GPU-Warteschlange der Phase 2. Start: setsid nohup ./run_gpu_queue.sh >/dev/null 2>&1 &
cd "$(dirname "$0")"
source ~/.venvs/ae_v6/bin/activate
W="--epochs 25 --widths 64 128 256"
python exp.py R3 --r3-loss R1 --gan-weight 0.01 $W --tag B3 > runs/B3.log 2>&1
echo "B3 exit $? $(date +%H:%M:%S)" >> runs/queue_status.txt
python exp.py R3 --r3-loss R2 --vgg-size 48 $W --tag B2 > runs/B2.log 2>&1
echo "B2 exit $? $(date +%H:%M:%S)" >> runs/queue_status.txt
