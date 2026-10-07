# Phase 0 notes

- Scaffold created under `PrimGate/`.
- Reused from MambaLiteUNet: `MambaSSMLayer`, `DepthwiseConvolution`, LGFM, AMF, CGA (reference).
- New: `CrossGatedSymptomFiltering` (CGSF) for classification gating.
- `mamba_ssm` may be unavailable on Windows; fallback Conv1d is for smoke tests only.
- Do not train the full `MambaLiteUNet` segmentation model for this thesis.
