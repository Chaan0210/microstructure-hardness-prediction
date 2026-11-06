# infogain-20261017: report (2026-10-17, 14:00–15:30 KST)

**Verdict: no improvement.** Retain **082** (Public 8.9343098075) as the incumbent and **076** (8.9646534604) as the rollback.
Three hypotheses were tested, each under a protocol frozen before its result, and all three were rejected. A fourth was withdrawn
untested because it repeated an earlier experiment. No candidate reached confirmation, so no test predictions were generated and
no CSV was written. Nothing was submitted, and the final selection was not touched.

All scores below are conditional development OOF on the 400 reused dev images (original partition, 5 folds) unless labelled
otherwise. Every label in them was inspected by earlier sessions, so these are reused-validation numbers, not Private forecasts.

## 1. Diagnosis (existing OOF only; `diag/`)
Baseline: the source-restricted strict-076 reconstruction (9.4924) and strict-082 (9.2567). My rebuild from the six components
reproduces strict-076 to 1.1e-11.

| band | n | RMSE 082 | SSE share | corr(pred, y) |
|---|---:|---:|---:|---:|
| clean (<9.5) | 205 | 8.01 | 38% | 0.889 |
| mid | 71 | 8.14 | 14% | 0.851 |
| noisy (≥12) | 124 | 11.51 | 48% | 0.837 |

1. **Clean and mid rows look like a finite-window sampling floor, not a measurement defect.**
   - Error falls with grain count N: clean err² ≈ 53 + 2.6e3/N; coarse tertile RMSE 9.8 vs fine tertile 6.2.
   - Prediction quality does not fall with N: corr(pred, y) is 0.86–0.88 in every tertile and the slopes are ≈1.
   - Coarse images simply have more hardness spread (SD 21 vs 13).
   - The same holds for elongated or oblique microstructures: larger error, proportionally larger spread, no orientation defect.
   - **Implication:** re-weighting, shrinkage or recalibration cannot help these rows.
2. **On noisy images, the measurement that breaks is phase, in fine- and medium-grained images.**
   - Feature–label correlation, clean → noisy: u_ph 0.76→0.52, z_f25 0.77→0.53, sh_cor15 0.79→0.51, sp_contiguity 0.60→0.16.
   - Coarse noisy images keep almost all of their signal: corr 0.877 vs 0.882 on clean.
3. **Components.**
   - No component is harmful. The grain-set model's 244-HV error sits on a segmentation-failed row that already falls back to
     CNN021; removing or clipping the set component changes 082 by ≤0.001.
   - Only 2 rows are failed, both with noise 0.
   - Component disagreement does not predict error (Spearman −0.01 clean, 0.11 noisy).
   - Stratified re-weighting gains 0.30 in-sample but collapses when cross-fitted, so that family stays closed.
4. **What the 082 residual CNN actually uses** (`diag/cnn_probe.py`).
   - Method: the saved round-4 CNNs were re-run on manipulated validation images with their contexts held fixed. The rebuilt
     contexts reproduced the stored predictions to <1e-6.

   | input to the CNN | gated dMSE vs strict-076 | corr with original correction |
   |---|---:|---:|
   | original | −8.53 | 1 |
   | Fourier phase randomised | −7.66 | 0.95 |
   | blur σ = 2 | −7.45 | 0.94 |
   | 32-px patch shuffle | −7.21 | 0.89 |
   | 8-px patch shuffle | −5.68 | 0.61 |
   | per-image mean/SD standardised | **−3.02** | **0.48** |

   - The correction is mostly a function of global intensity and the power spectrum (phase-randomisation-invariant texture
     statistics), not of shapes or arrangement.
   - It correlates −0.56 with raw image mean. The 98 tabular columns had img_std but no image mean.

## 2. Hypothesis ledger
| # | hypothesis / mechanism | matched control | primary test, frozen rule | result | reason |
|---|---|---|---|---|---|
| H1 | Blur-invariant first-moment phase fraction φ1 = (M − mean)/(M − P) | U-Net u_ph and H9 p34 stability on the same copy pairs | label-free gates G1 (Spearman with u_ph on clean ≥ 0.70) and G2 (copy–source stability > u_ph and H9) | **rejected, no labels used** | G1 0.83 passes. G2 fails: 0.42 vs u_ph 0.82 and H9 0.75, because the histogram-mode reference is itself blur-dependent |
| H2 | Random-subspace TabPFN (variance from redundant columns) | extra-seed ensemble | — | **withdrawn** | already tested in push-500 (`dev_eval.py` COLBAG, 75% of columns: worse, 10.257) |
| pre-check | Equal-weight fusion of 5 phase estimators | best single estimator | copy-pair stability | not pursued | fused 0.851 < sh_cor15 0.868: the estimators share their degradation errors |
| H3 | Raw image mean (one column) in TabPFN: a blur-invariant first moment that the CNN exploits (§1.4) | one N(0,1) column | slot 044 × 4 variants × 3 seeds × 5 folds, production protocol on CPU, BASE re-run (CPU vs stored MPS 0.02 RMS). Advance only if, on strict-082: Δ ≤ −0.010, beats random by ≥ 0.010, ≥ 3/5 folds, gain ≥ 0 without top 10 | **rejected** | see below: Δ and the random-column margin miss the thresholds, and the gain vanishes without the top 10 |
| H4 | Replace the fragile CNN correction with RidgeCV on 23 explicit spectral columns + the same 6 context inputs | context-only ridge (same protocol) | dev vs 082; holdout-100; frozen replacement and addition rules | **rejected** | see below: worse than 076, behind its own control, harmful on the holdout |

H3 detail (`h3/screen-result.json`):
- On strict-082: 9.2567 → 9.2501 (Δ −0.0067), P(better) 0.82, CI [−0.008, +0.021].
- Fold deltas: 0.000 / −0.018 / −0.009 / −0.006 / −0.001.
- Without the 10 most-helped images: +0.0053.
- Random-column control: +0.0012. On strict-076: −0.0085.
- The direction is consistent: gated member RMSE improved in all four variants (12.52/12.41/12.42/12.52 → 12.43/12.35/12.38/12.43),
  while the random column made them worse (12.57/12.49/12.54/12.58).
- But it is small, and fragile to the most-helped images.

H4 detail (`h4/result.json`):

| arm | dev RMSE (strict) |
|---|---:|
| 076 | 9.4924 |
| 076 + spectral correction (S) | 9.5182 |
| 076 + context-only control (C) | 9.4783 |
| 082 | 9.2567 |
| 082 + S | 9.3000 |

- Holdout-100, 35 gated rows, on the retest 400→100 teacher base: S +4.85 ± 2.64 (P(better) 0.03), CNN +6.17, C +1.51.
- Re-partitions 1/2 (frozen λ): S −0.29 / +0.35 vs CNN −7.09 / −4.30.
- The CNN's signal is a nonlinear function of texture statistics; a linear readout of explicit band powers does not reproduce it.

## 3. Strongest defensible candidate
None beats 082 under the frozen rules. 082 and 076 stay as they are. Because nothing advanced, no full-pipeline confirmation, test
fit or CSV was produced.

## 4. What was learned
- **Clean and mid rows (52% of SSE) behave as if at a sampling floor:** calibrated, with constant explained variance across
  grain counts. No modelling change in this study or earlier ones moved them.
- **Noisy rows lose phase information in the image itself:** all phase estimators fail together on copies, so fusion cannot recover it.
- **The 082 correction reads phase-randomisation-invariant texture statistics, mainly mean intensity.**
  - It helps re-partitions of the dev images but not the 100 holdout images (also with the base used here).
  - Its explicit linear surrogate fails.
  - The CNN's dev gain is therefore real on dev-like rows but its generalisation is unproven, as the 10-17 retest also concluded.
- **The image mean carries a small, consistent signal on noisy rows** that TabPFN can use, worth roughly 0.007 per slot on dev.
  That is far below the 0.10 target.
- **A 0.10 gain is not supported by any mechanism found.** The evidence points to roughly 0.01–0.03 of remaining recoverable
  signal with the current information.

## 5. Single most justified next experiment
A fresh, pre-registered **full-block test of `im_mean`**:
- all 12 TabPFN members, nb+h9 and full+h9 feature sets;
- evaluated on re-partitions 20261031/20261101 and at the block level on holdout-100;
- acceptance tied to the holdout and to robustness against the top-10 images.

It is the only change here with a physical mechanism, a consistent sign across folds and variants, and a beaten random control.
Its expected size is small (≈0.01–0.02 on dev), and it reuses already-inspected images. Cost is ≈2.5 h CPU; `confirm/` is ready.
Do not run it as a rescue of H3 under looser rules: it needs its own frozen rule.

## 6. Integrity
- All new files are in this directory. Round-4 code was imported read-only (`-B`, `sys.dont_write_bytecode`; its `__pycache__`
  still has 15 files).
- No file outside this directory, or the concurrent `structure-context-20261017-1415` session's directory, was modified after 14:00.
- `protect.py start|end`: 865 recorded 076/082 dependency files unchanged against `stepchange-20261016/record/protected-sha256-v2.json`.
  076 `b1be09b6…`, 082 `0fa34cec…`. Git HEAD `f8f2b3ad`.
- No installs, environment changes, external data, paid compute or test-data use. No test predictions were made.
- TabPFN ran on CPU because another session was using MPS. Its job was not touched.

## 7. Limitations
- All labels are reused. The three dev partitions share images. The holdout has only 35 gated rows.
- The U-Net and Cellpose representations were trained label-free on all 500 train images (inherited by every arm), so this is not
  strictly inductive for them.
- H3 was screened on one slot; a full-block effect was not measured.
- The CNN probe holds the context fixed and uses the per-fold λ; it describes the deployed CNNs, not every possible image model.

## 8. Decision (2026-10-17, user)
Stop here: 082 stays final, 076 the rollback; the full-block im_mean experiment in section 5 is **not** to be run.
