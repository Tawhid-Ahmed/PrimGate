"""CGSF + SibSoftmax (no GroupProto) lambda_sib sweep for Gate C v3.1."""

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

LAMBDAS = [0.15, 0.25, 0.35]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    for lam in LAMBDAS:
        name = f"primgate_sc_cgsf_sib_l{lam:.2f}".replace(".", "p")
        cmd = [
            sys.executable,
            str(ROOT / "train_primgate_sc.py"),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--lambda-sib",
            str(lam),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--no-group-proto",
            "--eval-test",
            "--run-name",
            name,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True, env=ENV)
    print("lambda_sib sweep done -> experiments/primgate_sc/tea_results.csv")


if __name__ == "__main__":
    main()
