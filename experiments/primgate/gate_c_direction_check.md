# Gate C — PrimGate vs Balanced Softmax (frozen tea 70/15/15, seed 42)

**Decision: FAIL (direction not yet CVPR-ready)**

Primary reference (Phase 1 winner): ResNet-50 + Balanced Softmax  
`test_macro_f1=0.433`, `test_sibling_error=0.707`, `test_acc_few=0.391`

## Results

| Run | test Macro-F1 | sib_err | few Acc | notes |
|---|---:|---:|---:|---|
| ResNet50 + BS | **0.433** | 0.707 | **0.391** | capacity ~25M |
| PrimGate full | 0.347 | 0.584 | 0.261 | ~0.9M; Conv1d Mamba fallback |
| w/o CGSF | 0.330 | 0.493 | 0.391 | |
| w/o SPF | 0.329 | 0.522 | 0.478 | |
| w/o SATC | 0.310 | **0.440** | 0.087 | |

## Gate criteria (thesis protocol)

1. **Macro-F1 ≥ BS** (or clearly better few / sibling tradeoff without collapsing head Acc) → **FAIL** (−8.6 pp F1).
2. **Sibling-error among errors ↓ vs BS** → **PASS** (0.584 vs 0.707), but ablations show **noSATC is lower still** → SATC not the causal driver.
3. **Ablation: full ≥ each removed module on Macro-F1** → **PASS** (full best F1 among PrimGate runs).
4. **Implementation fidelity** → **FAIL caveat**: `mamba_ssm` unavailable; AMF uses Conv1d scaffold. Also stem downsampled 224→28 for 16GB VRAM (LGFM O(N²)).

## Interpretation

- Dual-encoder PrimGate **does not beat** the BS baseline on the headline metric under current capacity / fallback SSM.
- Sibling-error improves vs BS, but module ablations do **not** attribute that gain to SATC/SPF/CGSF in the intended way.
- Full > ablations on Macro-F1 is a weak positive signal that the stack is not pure noise.

## Required next steps before claiming novelty

1. **Capacity-matched backbone**: ResNet-50 (or shared ImageNet stem) + CGSF/SPF/SATC heads — same protocol, same seed.
2. **Real Mamba SSM** (Linux/WSL CUDA wheel) or replace AMF with a published efficient SSM that installs on this machine; drop Conv1d scaffold from reported runs.
3. **Retune** `λ_satc`, `λ_bal`, lr; verify SATC reduces sib_err vs noSATC under the matched backbone.
4. Re-run Gate C; only proceed to paper claims if Macro-F1 is competitive **and** sibling/few gains are ablation-supported.
