# PrimGate

**Cross-Gated Symptom Primitive Factorization for Fine-Grained Long-Tailed Visual Recognition**  
Case study: severely imbalanced tea leaf pathology classification.

Built from [MambaLiteUNet](../MambaLiteUNet) principles (AMF, LGFM, MambaSSM, CGA→CGSF).  
This repo is **classification-only** (no segmentation training).

## Environment status (Phase 0)

| Component | Status |
|-----------|--------|
| PyTorch + CUDA | Required |
| `timm` | Required |
| `mamba_ssm` | Preferred; Conv1d fallback used if missing (scaffold only) |

```bash
cd "f:\Shakil thesis\PrimGate"
pip install -r requirements.txt
# Optional real Mamba:
# pip install --no-build-isolation causal_conv1d mamba_ssm
```

### Local download cache (this drive)

Training/eval scripts call `utils.local_cache.setup_local_caches()` so Torch Hub,
Hugging Face, and related weights land under:

`f:\Shakil thesis\PrimGate\.cache\`

(not `C:\Users\...\.cache`). That folder is gitignored.

## Data

Tea images live at:

`f:\Shakil thesis\processed_dataset\`

9 class folders (Brown Blight … Healthy Leaf). Splits JSON is created in **Phase 1**.

## Phase 0 smoke test

```bash
cd "f:\Shakil thesis\PrimGate"
python scripts/smoke_test.py
```

## Layout

```
PrimGate/
├── models/
│   ├── blocks/          # AMF, LGFM, MambaSSM, CGSF, CGA reference
│   ├── PrimGate.py      # dual encoder + CGSF classifier scaffold
│   └── MambaLiteUNet_ref.py
├── loader/tea_loader.py
├── losses/              # CE/LDAM stubs, SPF, SATC
├── engine_cls.py
├── train_cls.py
├── eval_cls.py
├── configs/
├── scripts/
├── analysis/
├── splits/              # Phase 1 JSON
└── experiments/
```

## Phase 1 — splits & baselines

```bash
cd "f:\Shakil thesis\PrimGate"

# Frozen stratified 70/15/15 split
python scripts/make_splits.py

# Single baseline (ResNet-50 + CE)
python train_cls.py --model resnet50 --loss ce --epochs 50 --eval-test

# Full B1–B3 suite
python scripts/run_baselines.py --epochs 50
```

Results land in `experiments/baselines/` and `experiments/baselines/tea_results.csv`.

## Train (smoke)

```bash
python train_cls.py --model resnet50 --loss ce --epochs 1 --batch-size 8 --eval-test
```
