"""Force heavy downloads (Torch Hub, HF, etc.) into PrimGate/.cache on this drive."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"


def setup_local_caches(verbose: bool = True) -> Path:
    """
    Point common ML cache env vars at PrimGate/.cache.

    Call this before importing torchvision/timm/huggingface code that downloads weights.
    Uses setdefault so an explicit user override still wins.
    """
    torch_home = CACHE_ROOT / "torch"
    hf_home = CACHE_ROOT / "huggingface"
    xdg = CACHE_ROOT / "xdg"
    torch_ext = CACHE_ROOT / "torch_extensions"
    timm_cache = CACHE_ROOT / "timm"

    for path in (
        CACHE_ROOT,
        torch_home,
        torch_home / "hub" / "checkpoints",
        hf_home,
        hf_home / "hub",
        hf_home / "transformers",
        xdg,
        torch_ext,
        timm_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_ext))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))
    os.environ.setdefault("TIMM_CACHE_DIR", str(timm_cache))

    if verbose:
        print(f"local_cache={CACHE_ROOT}")
        print(f"TORCH_HOME={os.environ['TORCH_HOME']}")

    return CACHE_ROOT
