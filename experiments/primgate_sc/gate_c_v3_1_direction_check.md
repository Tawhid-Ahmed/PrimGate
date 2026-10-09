# Gate C v3.1 — λ_sib sweep (CGSF + SibSoftmax, no GroupProto)

**Decision: FAIL on test Macro-F1; G1 still improved vs BS**

Config fixed: ResNet-50 dual-stream, CGSF on, GroupProto off, cosine LR, early stop on val Macro-F1.

## Results (test)

| Run | λ_sib | val Macro-F1 | **test Macro-F1** | **G1** | few | Healthy F1 |
|---|---:|---:|---:|---:|---:|---:|
| BS | — | — | **0.433** | 0.310 | 0.391 | 0.400 |
| SC (prior) noGroupProto | 0.50 | 0.419 | **0.403** | **0.210** | 0.435 | 0.000 |
| sweep | 0.15 | 0.433 | 0.362 | 0.243 | 0.391 | 0.000 |
| sweep | 0.25 | **0.440** | 0.375 | 0.263 | 0.348 | 0.111 |
| sweep | 0.35 | 0.430 | 0.373 | 0.230 | 0.478 | 0.000 |

## Checks

| Criterion | Result |
|---|---|
| test Macro-F1 ≥ BS | **FAIL** (best sweep 0.375; prior λ=0.5 still best SC at 0.403) |
| G1 < BS | **PASS** (all ≤ 0.263) |
| Lower λ recovers F1 | **FAIL** — val F1 rose, **test F1 fell** (val–test gap) |

## Interpretation

1. Early-stop on **val** Macro-F1 is selecting checkpoints that look BS-competitive on val but drop ~6–7 pp on test.
2. Among SC configs so far, **λ_sib=0.5 + no GroupProto** remains the best test tradeoff (F1 0.403 / G1 0.210).
3. Novelty (G1↓) is real; closing the last ~3 pp Macro-F1 to BS needs better selection / regularization, not another λ grid alone.

## Recommended next

1. Re-score all SC checkpoints with **test-blind** selection: pick by val Macro-F1 **and** val G1 Pareto, or average last-k epochs.
2. Or train λ=0.5 noGroupProto longer with stronger weight decay / label smoothing and report mean±std over 3 seeds.
3. Only then claim Gate C pass if test Macro-F1 ≥ BS and G1 < BS.
