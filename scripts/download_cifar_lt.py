"""Download CIFAR-100-LT — Phase 4."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.local_cache import setup_local_caches

setup_local_caches()
print(
    "Phase 4 placeholder. CIFAR downloads will go under PrimGate/.cache "
    f"and/or {ROOT / 'data'}."
)
