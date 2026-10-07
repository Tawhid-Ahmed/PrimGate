"""Phase 0 smoke test: import blocks, forward PrimGate on random tensor."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.blocks import MAMBA_AVAILABLE, CrossGatedSymptomFiltering
from models import PrimGate
from loader import TEA_CLASSES, TeaLeafDataset
from configs.config_tea import TeaConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"mamba_ssm_available={MAMBA_AVAILABLE}")
    print(f"num_tea_classes={len(TEA_CLASSES)}")

    ds = TeaLeafDataset(TeaConfig.data_root)
    print(f"scanned_images={len(ds)}")

    model = PrimGate(num_classes=9, feat_dim=64).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    out = model(x)
    assert out["logits"].shape == (2, 9), out["logits"].shape
    assert out["embedding"].shape == (2, 64), out["embedding"].shape

    cgsf = CrossGatedSymptomFiltering(32).to(device)
    s = torch.randn(2, 32, 28, 28, device=device)
    c = torch.randn(2, 32, 28, 28, device=device)
    s2, c2 = cgsf(s, c)
    assert s2.shape == s.shape and c2.shape == c.shape

    n_params = sum(p.numel() for p in model.parameters())
    print(f"primgate_params={n_params:,}")
    print("smoke_test_ok")


if __name__ == "__main__":
    main()
