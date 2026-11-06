# Info-matched copies pilot — result (2026-10-16 23:45 KST)

**Decision: branch closed.** Neither primary comparison shows a predictive benefit on real held-out noisy rows.
- New-renderer copies add nothing to the linear DTA partners beyond what duplicating the same source rows does.
- They harm TabPFN, less than the production copies but clearly.
- No new candidate. No Stage B.

Protocol: `PLAN.md`, frozen at 21:50 before any copy existed, plus amendment A1 (one copy per source per variant), made before any label result. Runtime about 2 h of the 2.5 h budget; 9 arm types.

## What was tested
- **Copies.** Two renderer variants (V1, V2) from the child session's label-free renderer: phase-contrast compression, boundary erasure, blur, shading and noise matched to the real noise ≥ 12 pool.
- **Pipeline.** One copy of each of the 249 clean sources per variant (498 copies), run through the production copy feature pipeline.
- **Evaluation.** Real dev rows with noise ≥ 12 (124 per partition), official 3 partitions × 5 folds.
  - Every copy sits in its source's fold. Copies of holdout-100 sources were never used.
  - Per-image squared errors are averaged over the partitions before bootstrapping, because the three partitions are correlated evidence.

## Diagnostics (not decision criteria)
- **D3, hardness population** (all 500 train rows).
  - Mean: clean sources 194.5, real noisy rows 193.6 (KS p 0.49), so no mean shift.
  - Spread is wider on real noisy rows: SD 19.6 vs 17.1, ratio 95% CI [0.99, 1.33], Levene p 0.09. Holdout-100 shows the same direction (18.0 vs 15.1).
  - Rendering-robust structure measures show no significant difference: pore diameter, orientation, elongation (KS p 0.21–0.78).
  - This label-population difference cannot be produced by any renderer, and it should not be counted as a rendering defect.
- **D1, structure preservation** (copy vs its own source, correlation).

  | measure | V1 | V2 | production deg4 |
  |---|---:|---:|---:|
  | phase fraction (u_ph) | 0.69 | 0.70 | 0.82 |
  | size heterogeneity (het4_u) | 0.80 | 0.68 | 0.89 |
  | pore fraction | 0.45 | 0.31 | 0.60 |
  | grain size, elongation, orientation | ≥ 0.95 | ≥ 0.95 | ≥ 0.95 |

  The new renderer removes more phase, heterogeneity and pore information, as intended.
- **D2, label information** (Ridge 5-fold CV, n = 161, 20 draws, R² relative to each population's label SD).

  | rows | R² |
  |---|---:|
  | real noisy rows | 0.48 |
  | V1 copies | 0.46 |
  | V2 copies | 0.42 |
  | production copies | 0.42 |
  | the same sources' clean rows | 0.72 |

  Measured as explained variance, the copies are roughly information-matched to real noisy images. Their lower raw RMSE (12.6–13.1 vs 14.1) comes mostly from the narrower clean-source label spread.
- **Pixel-level copy-vs-real AUC** (child session): 0.90–0.98 for every variant. This is a diagnostic only. The child's 0.60 gate is retired.

## Decision comparisons
Gated real noisy rows; dMSE is new minus reference (negative = better); `work/pilot-results.json`.

| comparison | dMSE ± SE | P(better) | partitions | original-partition folds | without top-5 helped |
|---|---:|---:|---|---|---:|
| **Ridge R3 vs R1** (production + new copies vs production copies) | **+1.11 ± 6.65** | 0.43 | +2.8 / +1.7 / −1.1 | +11 / −0.5 / −3 / −3 / +9 | +7.9 |
| **logRidge R3 vs R1** | **+1.34 ± 6.75** | 0.42 | +3.8 / +1.8 / −1.6 | +11 / −3 / −1 / −1 / +11 | +8.3 |
| **TabPFN T2 vs T0** (new copies vs real only) | **+23.6 ± 14.7** | 0.05 | +21 / +30 / +20 | +25 / −2 / +19 / +32 / +33 | +37.5 |
| Ridge / logRidge R2 vs R1 (new instead of production copies) | +1.0 / +2.5 | 0.45 / 0.39 | mixed | | |
| Ridge / logRidge R3 vs RD (vs duplicating the same sources) | +0.46 / −0.10 | ≈0.5 | mixed | | |
| Ridge / logRidge R3 vs RS (vs shuffled-label copies) | −30.3 / −29.3 | 0.99 | all negative | | |
| Ridge / logRidge R1 vs R0 (production copies vs real only) | −17.2 / −20.0 | 0.88 / 0.93 | all negative | +30 / −62 / −74 / −9 / −16 | |
| TabPFN T1 vs T0 (production copies) | +35.2 ± 18.9 | 0.02 | all worse | | |
| TabPFN TD vs T0 (duplicates only) | +0.19 ± 3.2 | 0.48 | mixed | | |

**Gated RMSE:**

| arm | Ridge | logRidge | TabPFN |
|---|---:|---:|---:|
| real only (R0 / T0) | 12.94 | 12.84 | 12.31 |
| production copies (R1 / T1) | 12.26 | 12.03 | 13.67 |
| new copies only (R2 / T2) | 12.30 | 12.13 | 13.24 |
| production + new (R3) | 12.30 | 12.09 | — |
| duplicate control (RD / TD) | 12.28 | 12.09 | 12.32 |
| shuffled-label control (RS) | 13.48 | 13.24 | — |

## Interpretation
- **Established.**
  - The copies carry real hardness signal: the shuffled-label control is much worse.
  - Production copies still help linear models: R1 vs R0, consistent with the production design.
  - New copies give linear models nothing beyond extra weight on the same sources.
  - Flexible TabPFN is harmed by both copy kinds, and the harm is not explained by source reweighting.
- **Not established.** Why TabPFN is harmed. A joint feature-structure gap remains (pixel AUC ≥ 0.90, feature-level GBM AUC 0.73–0.79), and the narrower label population of copies is a plausible second factor (D3).
- **Exploratory caveat.** The renderer was calibrated on pixels of all real noisy train images, including validation rows. That can only flatter Stage A; since Stage A failed, fold-honest recalibration (Stage B) was not run.

## Artifacts and commands
- Code: `work/render.py` (child's, unchanged; SHA in `work/copied-from-child.sha256`), `work/render_set.py`, `pipe/` (child's, unchanged), `work/pilot.py`.
- Data: `synth/train_{v1a,v1b,v2a,v2b}` (v1b/v2b rendered but not featurised), `feat/synthrows-{v1a,v2a}-u1dmavgH2Z-anghcor.npz`.
- Results: `work/oof-ridge.npz`, `work/oof-tab-{T1,T2,TD}.npz` (T0 = `../stepchange-20261016/work/e1/oof-F0.npz`), `work/pilot-results.json`, `work/diag/{d1d2,d3}.json`.
- Re-run commands: see HANDOVER.md.
