# SC checkpoint selection audit

Composite score = val_macro_f1 - 0.35 * val_G1

| Run | F1-pick ep | val F1 | val G1 | Comp-pick ep | val F1 | val G1 | d_ep | test F1 | test G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primgate_sc_cgsf_sib_l0p15_seed42 | 7 | 0.433 | 0.253 | 7 | 0.433 | 0.253 | +0 | 0.362 | 0.243 |
| primgate_sc_cgsf_sib_l0p25_seed42 | 26 | 0.440 | 0.249 | 22 | 0.439 | 0.243 | -4 | 0.375 | 0.263 |
| primgate_sc_cgsf_sib_l0p35_seed42 | 9 | 0.430 | 0.206 | 9 | 0.430 | 0.206 | +0 | 0.373 | 0.230 |
| primgate_sc_full_seed42 | 19 | 0.366 | 0.211 | 19 | 0.366 | 0.211 | +0 | 0.351 | 0.189 |
| primgate_sc_noCGSF_seed42 | 26 | 0.371 | 0.247 | 13 | 0.361 | 0.193 | -13 | 0.372 | 0.222 |
| primgate_sc_noGroupProto_seed42 | 9 | 0.419 | 0.214 | 9 | 0.419 | 0.214 | +0 | 0.403 | 0.210 |
| primgate_sc_noSibSoftmax_seed42 | 19 | 0.367 | 0.253 | 19 | 0.367 | 0.253 | +0 | 0.353 | 0.251 |
| primgate_sc_stable_l0p50_seed42 | 30 | 0.341 | 0.152 | 25 | 0.340 | 0.144 | -5 | 0.325 | 0.150 |
| primgate_sc_stable_l0p50_seed43 | 19 | 0.321 | 0.125 | 7 | 0.314 | 0.071 | -12 | 0.304 | 0.059 |
| primgate_sc_stable_l0p50_seed44 | 14 | 0.321 | 0.139 | 3 | 0.313 | 0.106 | -11 | 0.289 | 0.108 |

## Takeaway
- If Comp-pick epoch differs from F1-pick, F1-only selection ignored better G1 tradeoffs.
- Retrain with composite selection + stronger regularization for Gate C v3.2.
