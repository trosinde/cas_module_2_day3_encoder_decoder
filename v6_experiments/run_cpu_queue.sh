#!/usr/bin/env bash
# CPU-Warteschlange: eine Aufgabe nach der anderen, damit das GPU-Training möglichst wenig gebremst wird.
cd "$(dirname "$0")"
source ~/.venvs/ae_v6/bin/activate
export CUDA_VISIBLE_DEVICES=""
nice -n 10 python perceptual.py B1 > runs/B1_perceptual.log 2>&1
echo "perceptual B1 exit $? $(date +%H:%M:%S)" >> runs/queue_status.txt
nice -n 10 python phase4a.py B1 > runs/B1_phase4a.log 2>&1
echo "phase4a B1 exit $? $(date +%H:%M:%S)" >> runs/queue_status.txt
python - <<'E' 2>/dev/null
import numpy as np, keras
(_, _), (x, _) = keras.datasets.cifar10.load_data()
np.save("runs/sr_input_original.npy", x[:10].astype("float32") / 255)
np.save("runs/sr_input_B1.npy", np.load("runs/B1_recon_first100.npy")[:10])
E
for n in original B1; do
  nice -n 10 ~/.venvs/ae_v6_sr/bin/python phase4b_sr.py runs/sr_input_$n.npy runs/sr_output_$n.npy > runs/sr_$n.log 2>&1
  echo "sr $n exit $? $(date +%H:%M:%S)" >> runs/queue_status.txt
done
