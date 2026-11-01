"""Phase 4a: Unsharp Mask auf fertigen Rekonstruktionen. Läuft auf der CPU neben dem Training.

Aufruf: phase4a.py B1 [B3 ...]
Ein Filter nach dem Decoder fügt keine Information hinzu. Gemessen wird daher auch, was er an Treue kostet.
"""
import json
import sys

from scipy.ndimage import gaussian_filter
import numpy as np

import perceptual as pc
import exp

SIGMA = 1.0
AMOUNTS = (0.5, 1.0, 2.0)


def unsharp(images, amount):
    blurred = gaussian_filter(images, sigma=(0, SIGMA, SIGMA, 0), mode="reflect")
    return np.clip(images + amount * (images - blurred), 0, 1).astype("float32")


def main(tags):
    _, _, x_test = pc.load_splits()
    test_feats = pc.real_features("test", x_test)
    for tag in tags:
        recon = pc.rebuild(tag).predict(x_test, batch_size=500, verbose=0)
        result = {}
        grids = {}
        for amount in AMOUNTS:
            sharp = unsharp(recon, amount)
            metrics = exp.evaluate(sharp, x_test)
            metrics.update(pc.score(x_test, sharp, test_feats))
            result[str(amount)] = metrics
            grids[str(amount)] = sharp[:10]
            print(tag, amount, {k: v for k, v in metrics.items() if k != "spectrum"}, flush=True)
            exp.save_grid(exp.OUT / f"{tag}_unsharp{amount}_grid.png", x_test, sharp)
        (exp.OUT / f"{tag}_unsharp.json").write_text(json.dumps(result, indent=1))
        np.savez(exp.OUT / f"{tag}_unsharp_first10.npz", **grids)


if __name__ == "__main__":
    main(sys.argv[1:])
