# Gate C v3.2 — selection audit + stable 3-seed retrain

**Decision: FAIL** (composite+strong reg over-optimized G1 and collapsed Macro-F1)

## 1) Selection audit (existing runs)

Composite = `val_F1 - 0.35 * val_G1`. Full table: `selection_audit/selection_audit.md`.

Key finding: for the **previous best** (`noGroupProto` λ=0.5), F1-pick and Comp-pick were the **same epoch**. Selection alone does not explain that run’s remaining F1 gap to BS.

## 2) Stable 3-seed retrain

Recipe: CGSF + SibSoftmax, no GroupProto, λ_sib=0.5, WD=5e-4, LS=0.05, selection=composite β=0.35, patience=12.

| Seed | best_ep | test Macro-F1 | test G1 | few |
|---:|---:|---:|---:|---:|
| 42 | 25 | 0.325 | 0.150 | 0.522 |
| 43 | 7 | 0.304 | **0.059** | 0.565 |
| 44 | 3 | 0.289 | 0.108 | 0.174 |
| **mean±std** | — | **0.306±0.018** | **0.106±0.046** | 0.420±0.214 |
| BS | — | **0.433** | 0.310 | 0.391 |
| Prior best (F1-sel, WD=1e-4, no LS) | 9 | **0.403** | 0.210 | 0.435 |

## Checks

| Criterion | Result |
|---|---|
| test Macro-F1 ≥ BS | **FAIL** (−13 pp vs BS; worse than prior SC) |
| G1 < BS | **PASS** (mean 0.106) |
| 3-seed stability of F1 | Weak (low mean, early picks on 43/44) |

## Interpretation

1. **β=0.35 composite is too G1-heavy** — picks early underfit checkpoints (ep 3/7) that look good on score but destroy head Acc / Macro-F1.
2. **Stronger WD + LS + SibSoftmax** further bias toward rare/sibling at expense of many-class F1.
3. **Working reference remains** prior `primgate_sc_noGroupProto_seed42` (F1 0.403 / G1 0.210) under F1-only selection and milder regularization.

## Thesis stance (honest)

- **Positive claim ready:** sibling-conditional softmax reduces G1 blight confusion vs BS (ablations + multi-seed).
- **Not ready:** PrimGate-SC as a drop-in Macro-F1 winner vs Balanced Softmax.
- Path to close F1: milder β (≤0.1) or constrained Pareto (require val_F1 ≥ 0.40 then minimize G1), keep WD=1e-4, optional LS≤0.02 — **not** another architecture rewrite.
