# Gate C v2 — Capacity-matched PrimGate (ResNet-50 dual stream) vs BS

**Decision: FAIL (F1 recovered; sibling novelty still not shown)**

Protocol: frozen tea 70/15/15, seed 42, 30 epochs, AMP, λ_satc=0.3, λ_bal=0.5.  
Backbone: shared ResNet-50 stem + dual layer4 + AMF (Conv1d Mamba fallback; `mamba_ssm` still unavailable on Windows).

## Reference

| Run | Macro-F1 | sib_err | few Acc |
|---|---:|---:|---:|
| ResNet50 + Balanced Softmax | **0.433** | **0.707** | **0.391** |

## Capacity-matched PrimGate

| Run | Macro-F1 | sib_err | few Acc |
|---|---:|---:|---:|
| R50 full (CGSF+SPF+SATC) | 0.424 | 0.782 | 0.304 |
| R50 w/o CGSF | 0.354 | 0.654 | 0.174 |
| R50 w/o SPF | **0.439** | 0.724 | 0.217 |
| R50 w/o SATC | 0.352 | 0.769 | 0.087 |

Lite PrimGate (Gate C v1) was Macro-F1 **0.347** — capacity match closed most of the gap to BS.

## Checks

| Criterion | Result |
|---|---|
| Macro-F1 ≥ BS (full) | **FAIL** (−0.9 pp; within noise, not a win) |
| Sibling-error ↓ vs BS (full) | **FAIL** (0.782 > 0.707) |
| CGSF ablation hurts F1 | **PASS** (0.424 → 0.354) |
| SATC ablation hurts F1 / few | **PASS** (F1 0.424→0.352; few 0.304→0.087) |
| SPF ablation hurts F1 | **FAIL** (noSPF **better** test F1 0.439) |
| Real mamba_ssm | **FAIL** (Conv1d fallback) |

## Interpretation

1. **Capacity was the main Gate C v1 failure.** ResNet dual-stream restores near-BS Macro-F1.
2. **Headline novelty (sibling disentanglement) is not evidenced** — full sib_err is worse than BS.
3. **CGSF and SATC help classification F1/few** under this backbone; **SPF currently hurts** test F1 (likely λ_bal / prototype occupancy under-tuned or over-regularizing).
4. Training showed very large `train_loss` spikes on some runs (composite loss scale) — stabilize before paper claims.

## Next (only if pursuing CVPR angle)

1. Drop or redesign SPF (start from **CGSF+SATC only**, which already matches BS F1 at 0.439 without SPF).
2. Retarget SATC / metrics so sibling-error among errors drops vs BS with an ablation that attributes the drop to SATC.
3. Install real `mamba_ssm` (WSL/Linux) or replace AMF with a Windows-installable sequence mixer before claiming Mamba.
4. Re-run Gate C; require sib_err ↓ **and** Macro-F1 ≥ BS with supporting ablations.
