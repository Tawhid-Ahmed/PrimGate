"""Launch Phase-1 baseline suite B1–B3 (ResNet-50 + CE/LDAM/Balanced Softmax)."""

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
# Ensure child train_cls processes inherit the same cache roots
_ENV = os.environ.copy()

BASELINES = [
    ("resnet50", "ce"),
    ("resnet50", "ldam"),
    ("resnet50", "balanced_softmax"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-test", action="store_true", default=True)
    p.add_argument("--no-eval-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    eval_test = False if args.no_eval_test else args.eval_test
    for model, loss in BASELINES:
        cmd = [
            sys.executable,
            str(ROOT / "train_cls.py"),
            "--model",
            model,
            "--loss",
            loss,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if eval_test:
            cmd.append("--eval-test")
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True, env=_ENV)
    print("All baselines finished. See experiments/baselines/tea_results.csv")


if __name__ == "__main__":
    main()
