"""Train PrimGate-SC full + ablations sequentially."""

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

RUNS = [
    ("primgate_sc_full", []),
    ("primgate_sc_noCGSF", ["--no-cgsf"]),
    ("primgate_sc_noGroupProto", ["--no-group-proto"]),
    ("primgate_sc_noSibSoftmax", ["--no-sib"]),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-sib", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    for name, flags in RUNS:
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
            str(args.lambda_sib),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
            "--eval-test",
            "--run-name",
            name,
            *flags,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True, env=ENV)
    print("PrimGate-SC suite done -> experiments/primgate_sc/tea_results.csv")


if __name__ == "__main__":
    main()
