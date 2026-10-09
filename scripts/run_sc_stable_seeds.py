"""3-seed stable retrain of winning PrimGate-SC recipe (Gate C v3.2)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.local_cache import setup_local_caches

setup_local_caches()
ENV = os.environ.copy()

SEEDS = [42, 43, 44]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lambda-sib", type=float, default=0.5)
    p.add_argument("--sel-beta", type=float, default=0.35)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    for seed in SEEDS:
        name = f"primgate_sc_stable_l0p50_seed{seed}"
        # run-name already includes seed suffix handling in trainer; pass base without seed
        # trainer appends _seed{seed}, so use name without seed and set --seed
        base = "primgate_sc_stable_l0p50"
        cmd = [
            sys.executable,
            str(ROOT / "train_primgate_sc.py"),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--label-smoothing",
            str(args.label_smoothing),
            "--lambda-sib",
            str(args.lambda_sib),
            "--selection",
            "composite",
            "--sel-beta",
            str(args.sel_beta),
            "--patience",
            str(args.patience),
            "--min-epochs",
            "8",
            "--no-group-proto",
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--eval-test",
            "--run-name",
            base,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True, env=ENV)
    print("stable 3-seed suite done -> experiments/primgate_sc/")


if __name__ == "__main__":
    main()
