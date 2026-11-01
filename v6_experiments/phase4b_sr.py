"""Phase 4b: Real-ESRGAN x4 (BSD-3-Clause, Xintao Wang) auf 32x32-Bildern. Eigene Umgebung ~/.venvs/ae_v6_sr.

Aufruf: phase4b_sr.py <eingabe.npy> <ausgabe.npy>   Eingabe: N x 32 x 32 x 3, Werte 0 bis 1.
Das Netz ist aus dem offiziellen Aufbau RRDBNet nachgebaut, um basicsr nicht installieren zu müssen.
Es gibt zu diesen Bildern keine höher aufgelöste Wahrheit, die Details sind also erfunden und nicht messbar.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS = Path.home() / ".cache" / "ae_v6" / "realesrgan" / "RealESRGAN_x4plus.pth"


class ResidualDenseBlock(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.conv1 = nn.Conv2d(feat, grow, 3, 1, 1)
        self.conv2 = nn.Conv2d(feat + grow, grow, 3, 1, 1)
        self.conv3 = nn.Conv2d(feat + 2 * grow, grow, 3, 1, 1)
        self.conv4 = nn.Conv2d(feat + 3 * grow, grow, 3, 1, 1)
        self.conv5 = nn.Conv2d(feat + 4 * grow, feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, feat=64, grow=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(feat, grow)
        self.rdb2 = ResidualDenseBlock(feat, grow)
        self.rdb3 = ResidualDenseBlock(feat, grow)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, feat=64, blocks=23, grow=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(feat, grow) for _ in range(blocks)])
        self.conv_body = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(feat, feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_model():
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, WEIGHTS)
    # weights_only verhindert, dass beim Laden der fremden Datei Code ausgeführt wird.
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = RRDBNet()
    model.load_state_dict(state["params_ema"], strict=True)
    return model.eval()


def main(source, target):
    images = np.load(source)
    model = load_model()
    out = []
    with torch.inference_mode():
        for i, image in enumerate(images):
            tensor = torch.from_numpy(image.astype("float32")).permute(2, 0, 1).unsqueeze(0)
            out.append(model(tensor).clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy())
            print(f"{i + 1}/{len(images)}", flush=True)
    np.save(target, np.stack(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
