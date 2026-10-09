# Gate C v3 — PrimGate-SC vs Balanced Softmax

**Decision: PARTIAL PASS on novelty (G1); FAIL on Macro-F1**

Protocol: frozen tea 70/15/15 seed 42. Primary metrics: Macro-F1, **G1 blight**, hijack_L2 (hijack deferred to follow-up). Secondary: few Acc, Healthy F1.

## Results (test)

| Run | Macro-F1 | G1 blight | few Acc | Healthy F1 |
|---|---:|---:|---:|---:|
| ResNet50 + BS | **0.433** | 0.310 | 0.391 | 0.400 |
| SC full (CGSF+GroupProto+SibSoftmax) | 0.351 | **0.189** | **0.652** | 0.031 |
| SC noCGSF | 0.372 | 0.222 | 0.478 | 0.000 |
| SC noGroupProto | **0.403** | 0.210 | 0.435 | 0.000 |
| SC noSibSoftmax | 0.353 | 0.251 | 0.391 | 0.000 |

## Checks

| Criterion | Result |
|---|---|
| Macro-F1 ≥ BS | **FAIL** (best SC 0.403 still −3 pp; full −8 pp) |
| G1 < BS | **PASS** (full 0.189 vs 0.310; −12 pp) |
| SibSoftmax causes G1 drop | **PASS** (noSib G1 0.251 > full 0.189) |
| GroupProto helps F1 | **FAIL** (noGroupProto best F1 among SC) |
| Healthy not collapsed | **WEAK** (full 0.031) |

## Interpretation

1. **Sibling-conditional softmax is the right lever** for the thesis RQ (G1). Embedding-margin SATC was not.
2. **Group-conditional protos currently hurt Macro-F1**; keep as optional / redesign later.
3. Full SC **over-optimizes sibling/few** (few Acc 0.65) and underperforms head Macro-F1 — `λ_sib=0.5` is too strong with BS jointly.
4. CGSF + SibSoftmax without GroupProto is the strongest SC config so far (F1 0.403, G1 0.210).

## Immediate next (surgery, not new architecture)

1. Train **CGSF + SibSoftmax only** (`--no-group-proto`) with `λ_sib ∈ {0.15, 0.25, 0.35}`.
2. Target: Macro-F1 ≥ 0.43 **and** G1 ≤ 0.25.
3. Then remasure hijack_L2 and declare Gate C v3.1.
