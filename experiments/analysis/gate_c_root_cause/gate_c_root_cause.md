# Gate C root-cause analysis

## Dataset
- Root: `f:\Shakil thesis\processed_dataset` (custom tea pathology, 9 classes)
- Split: frozen 70/15/15 seed 42
- Train counts: [331, 662, 2516, 1104, 12, 64, 134, 40, 106]
- Prep is fine; hardness is LT + blight siblings + Healthy n=12.

## Metrics table (test)

| Run | Macro-F1 | sib_among | G1 blight | few | Healthy F1 | hijack_L2 |
|---|---:|---:|---:|---:|---:|---:|
| BS | 0.4327 | 0.7074 | 0.3100 | 0.3913 | 0.4000 | 0.0112 |
| r50_full | 0.4243 | 0.7820 | 0.2908 | 0.3043 | 0.0000 | 0.0074 |
| r50_noCGSF | 0.3542 | 0.6544 | 0.3029 | 0.1739 | 0.0000 | 0.0423 |
| r50_noSPF | 0.4386 | 0.7242 | 0.3060 | 0.2174 | 0.6667 | 0.0000 |
| r50_noSATC | 0.3520 | 0.7686 | 0.3029 | 0.0870 | 0.0000 | 0.0009 |

## Causes

1. **Metric trap:** Gate C used `sib_among_errors`. Full model G1 is 0.2908 vs BS 0.3100 (not worse), while sib_among rose because cross-group errors shrank (full cross-group errors=77, BS=127).

2. **SPF / L_bal harms rare classes:** noSPF Macro-F1 0.4386 > full 0.4243; Healthy F1 0.667 vs full 0.000. Online SPF ≠ Phase-2 KMeans hijack object.

3. **SATC helps F1, barely moves G1:** ΔF1=0.0723, ΔG1=-0.0122. Embedding margin ≠ within-group logit ranking.

4. **CGSF helps:** full F1 0.4243 vs noCGSF 0.3542.

5. **Hijack not closed by SPF:** see hijack_L2 column (full vs BS).

6. **Loss instability:** late train_loss explosion (full ratio=49.4x).

## Gate C v3 protocol (frozen)
- Primary: Macro-F1, G1 sibling rate, hijack_L2
- Secondary: few Acc, Healthy F1
- Diagnostic only: sib_among_errors

## Next move
PrimGate-SC: keep CGSF; drop global L_bal SPF; replace SATC with sibling-conditional softmax + group-conditional protos
