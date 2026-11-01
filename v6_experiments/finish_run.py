"""Wertet einen Lauf aus den gespeicherten Gewichten aus, wenn nur die Auswertung am Laufende scheiterte.

Aufruf: finish_run.py <tag> --gan-weight <g> --r3-loss <R1|R2> --seconds <s>   Läuft auf der CPU.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import exp

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--gan-weight", type=float, required=True)
p.add_argument("--r3-loss", required=True)
p.add_argument("--seconds", type=float, required=True)
p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
args = p.parse_args()

x_train, x_test = exp.load_data()
x_val = x_train[45000:]
generator = exp.build_v5(8, args.widths)
generator.load_weights(exp.OUT / f"{args.tag}.weights.h5")
generator.compile(jit_compile=False)
history = json.loads((exp.OUT / f"{args.tag}_history.json").read_text())
val = exp.evaluate(generator.predict(x_val, batch_size=500, verbose=0), x_val)
recon = generator.predict(x_test, batch_size=500, verbose=0)
metrics = exp.evaluate(recon, x_test)
metrics.update(run=args.tag, seconds=args.seconds, params=generator.count_params(),
               train_samples=45000, epochs=25, seed=42, widths=args.widths, r3_loss=args.r3_loss,
               gan_weight=args.gan_weight, gan_start=0.3, gpu_peak_mb=None,
               val_loss=history["val_loss"][-1],
               val={k: v for k, v in val.items() if k != "spectrum"},
               note="Auswertung nachträglich auf der CPU aus den gespeicherten Gewichten")
(exp.OUT / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=1))
np.save(exp.OUT / f"{args.tag}_recon_first100.npy", recon[:100])
exp.save_grid(exp.OUT / f"{args.tag}_grid.png", x_test, recon)
print({k: v for k, v in metrics.items() if k != "spectrum"}, flush=True)
