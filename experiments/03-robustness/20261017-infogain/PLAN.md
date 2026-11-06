# infogain-20261017 — plan and pre-registration (written 2026-10-17 ~15:20 KST, before any new hardness score)

Incumbent: **082** (Public 8.9343098075), rollback **076** (8.9646534604). Nothing here modifies them; every new file lives in this directory.
Validation baseline: the source-restricted strict-076 reconstruction (AI, `076-complement-20261016/baseline076.npz`,
dev OOF 9.4924) and strict-082 = strict-076 + fold-selected residual-CNN correction (`highnoise-ssl-20261016-2312/oof.npz`, 9.2567).
Rebuild check: 0.16·G + 0.384·tab + 0.128·ridge + 0.128·logridge + 0.05·set + 0.15·gmm reproduces strict-076 to 1.1e-11 (`diag/diagnose.py`).

## Phase-1 diagnosis (existing OOF only; `diag/diagnose.py`, `diag/diag.json`)
| band | n | RMSE 082 | SSE share | corr(pred,y) | slope y~pred |
|---|---:|---:|---:|---:|---:|
| clean (<9.5) | 205 | 8.01 | 38% | 0.889 | 1.02 |
| mid (9.5–12) | 71 | 8.14 | 14% | 0.851 | 1.00 |
| noisy (≥12) | 124 | 11.51 | 48% | 0.837 | 1.24 (known dev artefact; holdout 0.93) |

1. **Clean/mid rows look like a finite-window sampling floor, not a measurement defect.** Error grows as grain count N falls
   (clean err² ≈ 53 + 2.6e3/N; coarse tertile RMSE 9.8 vs fine 6.2), but corr(pred,y) is the same in every N tertile (0.86–0.88),
   slopes are ≈1, and the coarse tertile simply has larger hardness spread (SD 21 vs 13). Better shrinkage or re-weighting cannot help there.
2. **Noisy rows lose phase information in fine/medium-grained images.** Clean→noisy feature-label correlation: u_ph 0.76→0.52 (fine),
   z_f25 0.77→0.53, sh_cor15 0.79→0.51, sp_contiguity 0.60→0.16; coarse noisy images keep ~all their signal (corr 0.877 vs clean 0.882).
   Size heterogeneity (het/m_la_std) is mainly a coarse-image signal and is largely retained.
3. **Components.** Residual correlations with the blend 0.78–0.97; disagreement does not predict error (Spearman −0.01 clean, 0.11 noisy).
   The grain-set component's 244-HV outlier sits on a segmentation-failed row (already CNN021 fallback): removing/clipping set changes
   082 by ≤0.001 → no harmful component. Only 2 failed rows (both noise 0). Stratified in-sample re-weighting gains 0.30 but cross-fitted
   it collapses → re-weighting family stays closed.
4. **Concentration.** Top-10 rows = 16% of SSE, top-40 = 44%. Fold RMSE 8.15–9.90.

## Hypothesis ledger (what failed, what is untested)
| family | status | evidence |
|---|---|---|
| new tabular columns screened on dev residuals | failed (selection bias) | push-600 random-column control beats them |
| segmentation-free intensity ACF / map ACF / token readouts | redundant | push-1300 wacf, push-900 dir-2, tokset |
| zone het repair (variance-component, ZS-net, zone-ACF) | failed | push-1200/1300 |
| U-Net phase recalibration (u1b, deg11/12, class-weight H1, label-supervised readout) | failed | push-1100/1300, plateau-break |
| 3rd/4th-cumulant phase (H9) | **adopted (076)** | +0.012 TabPFN, random control 0 |
| degraded copies for TabPFN/GBM | failed | E1, infomatched pilot |
| copies for linear partners (DTA) | **adopted (061)** | Public-confirmed |
| re-weighting / stretch / TA / WLS / gating | failed on Public 7/7 | push-1300 meta-analysis |
| SSL, relational, raw+raw, EffNet, DINO partners | failed / share 0 | highnoise-ssl, highnoise-relational, 076-complement |
| residual-CNN correction | **adopted (082)**, fragile | re-partitions +, holdout −, Public −0.030 |
| distribution kernel, corruption-invariant kernel | failed | plateau-break H2/H3 |
| PADRE, transfer ridge, invariant-column TabPFN | failed | stepchange E1–E3 |
| **blur-invariant first-moment phase estimate** | **untested** | H9 uses k3/k4 (blur-sensitive); U-Net learns blur from copies |
| **random-subspace TabPFN (column-redundancy variance)** | **untested** | only group deletion (073) and seeds tested |

## H1 — blur-invariant first-moment phase fraction (information quality, targets diagnosis item 2)
- **Mechanism.** Blur and zero-mean noise leave the image mean unchanged; 3rd/4th cumulants (H9), local classification (U-Net) and
  segmentation all lose phase signal when blur mixes small dark grains into the matrix (fine noisy images). With matrix level M
  (histogram mode of a lightly smoothed image) and pore level P (dark reference), φ1 = (M − mean of non-pore pixels)/(M − P) estimates
  phase fraction × relative phase darkness without resolving individual grains. Two columns: global, and 4×4-block (64-px, shading-robust).
- **Difference from prior work.** No prior estimator uses the first moment with per-image level references; "brightness columns"
  (push-400) were raw intensity summaries without M/P normalisation.
- **Label-free gates (fixed now).** G1: on clean rows Spearman(φ1, u_ph) ≥ 0.70. G2: on the existing deg4 copies (realdeg2) vs their clean
  sources, Pearson(φ1_copy, φ1_source) must exceed the same statistic for u_ph and for H9 p34 (sigma 1) on the same pairs.
  Fail either → reject without labels.
- **Labeled screen (fixed now).** Production protocol for the TabPFN block (TabPFN v3.5, seeds 100·sd+fold, sd 0–2, contexts = training
  folds of the original partition, nb columns on non-gated rows, full columns on gated rows), slot 044 × 4 U-Net variants, three arms:
  BASE (re-run, must reproduce stored dev predictions), +φ1 (2 columns), +RAND (2 N(0,1) columns, seed 20261017). Blend effect computed as
  in the H9 screen (only the 044 slot's 4 members change; 082 otherwise unchanged).
  **Advance only if** (a) blend ΔRMSE(+φ1 vs BASE) ≤ −0.010 overall, (b) +φ1 beats +RAND by ≥ 0.010, (c) ≥3/5 folds improve,
  (d) gain stays ≥ 0 after removing the 10 most-helped images. Otherwise reject.
- **Runtime estimate.** Measurement < 5 min; screen ≈ 3 arms × 4 variants × 3 seeds × 5 folds = 180 TabPFN fits (benchmark first).

## H2 — random-subspace TabPFN block (component variance from redundant columns)
- **Mechanism.** 073 showed redundant/collinear columns hurt TabPFN on clean rows; deleting a selected group risks selection bias.
  Averaging members fitted on random 70% column subsets reduces that variance without label-based selection.
- **Matched control.** Same number of fits on all columns with extra seeds (separates subspace diversity from extra averaging).
- **Screen.** Slot 044 × 4 variants, K = 3 subspaces per member-seed vs K = 3 extra seeds; same advancement rule as H1 (a, c, d) with
  (b) replaced by "subspace beats extra-seed control by ≥ 0.010".
- Run only after H1's screen, if budget allows.

## H3 — reserved
Chosen after H1/H2 screens, only if a result points to a specific mechanism. Not more than three hypotheses in total.

## Confirmation (if anything advances; at most two finalists)
Freeze recipe; refit all 12 TabPFN members (3 slots × 4 variants × 3 seeds) with the change on the original partition and on
partitions 20261031/20261101 (fold-honest, same contexts), assemble the full 076 blend + existing correction (= 082 structure), compare
with strict-082 on identical rows; report overall/band RMSE, paired per-image bootstrap by image ID, fold and partition stability,
top-10 removal, and the untouched-by-this-study holdout-100 (TabPFN block 400→100) as a reused-but-unselected check.
Acceptance (fixed now): pooled ΔRMSE ≤ −0.02 vs 082, bootstrap P(better) ≥ 0.9, all three partitions ≤ 0, holdout-100 not worse
with P ≥ 0.8. Anything weaker is labelled "promising but inconclusive" or "no improvement".

## Validation limits (stated up front)
All 500 labels have been inspected before; dev and holdout are reused. U-Net / Cellpose representations were trained label-free on all
500 train images (inherited by both arms), so this is not strictly inductive for those representations. No test data are used.

## Amendments (appended in time order; nothing above was edited)
**A1 (≈15:35). H1 rejected label-free.** G1 pass (Spearman φ1_global–u_ph on clean 0.83), G2 fail: copy–source Pearson
φ1 0.42/0.38 vs u_ph 0.82, H9 p34 0.75 (`h1/gates.json`). The histogram-mode matrix reference is itself blur-dependent.
**A2 (≈15:45). H2 withdrawn without running: it is a repeat.** Column-subset bagging (75 % columns per seed, `push-300/dev_eval.py`
COLBAG) was tested in push-500 (NOTES l.52: 10.257, worse). Ledger corrected.
**A3 (≈15:50). Label-free pre-check, not a hypothesis slot:** equal-weight fusion of 5 phase estimators on deg4 copy pairs reaches
0.851 vs best single sh_cor15 0.868 → estimator errors are shared; fusion not pursued.
**A4 (≈16:05). Mechanism probe of the 082 residual CNN** (`diag/cnn_probe.py`, saved round4 aux CNNs, outer refits, contexts rebuilt
and stored predictions reproduced to <1e-6): gated dMSE vs strict-076 −8.53 (orig) → phase-randomised −7.66 (corr 0.95),
blur σ2 −7.45, shuffle32 −7.21, shuffle8 −5.68, **per-image mean/SD standardised −3.02 (corr 0.48)**. The correction correlates
−0.56 with raw image mean; image mean correlates −0.20 with the strict-076 residual on gated rows. The 98 tabular columns contain
img_std but no image mean.

## H3 — raw global mean intensity as a tabular column (pre-registered ≈16:10, before any labeled evaluation of it)
- **Mechanism.** The first moment of the image is unchanged by blur and zero-mean noise; with the renderer's fixed grey levels,
  darker images carry more dark (hard) phase. Fine/medium noisy images lose local phase resolution (diagnosis 2) but not the mean.
  The 082 CNN obtains most of its gain from this quantity (A4); giving it to the tabular learners turns a fragile residual into a
  measured input. Difference from H1: no per-image reference levels (the step that failed G2). Difference from push-400
  "brightness columns": that test predates H9, the multi-variant TabPFN block and the noise-gated feature sets.
- **Column.** Exactly one: `im_mean` = mean grey level / 255 of the original image (`h3/spectral.py`). No other spectral column.
- **Label-free gate.** deg4 copy–source Pearson(im_mean) ≥ 0.9 (needed for the copy-augmented partners).
- **Screen S3 (TabPFN, production protocol, original partition).** Slot 044 × 4 U-Net variants × seeds 0–2 × 5 folds, nb+h9 columns on
  non-gated rows and full+h9 on gated rows, three arms: BASE (re-run; must reproduce stored predictions to ≤1e-3), +MEAN, +RAND
  (one N(0,1) column, seed 20261017). Blend effect = replace only the 044 slot's 4 members in the 12-member block, evaluated on top
  of strict-082 (primary) and strict-076 (secondary).
  **Advance only if** on top of strict-082: (a) ΔRMSE(+MEAN − BASE) ≤ −0.010, (b) +MEAN beats +RAND by ≥ 0.010, (c) ≥ 3/5 folds
  improve, (d) gain ≥ 0 after removing the 10 most-helped images. Otherwise reject.
- If S3 advances, confirmation follows the pre-registered confirmation section (all 12 members, three partitions, holdout-100).
**A5 (14:35, timestamp correction).** The clock times written in this file ("≈15:20" … "≈16:10") were wrong estimates. File
modification times establish the true order: diagnosis 14:04, H1 gates 14:19, CNN probe 14:26, spectral measurement 14:27,
H3 pre-registration appended 14:28, screen evaluator written 14:31, first screen output 14:33. The order (pre-registration
before any labeled evaluation of im_mean in a learner) is unchanged. Session start ≈14:00 KST; 8-hour budget ends ≈22:00.
**A6 (15:10). H3 screen result: REJECTED by its rule** (`h3/screen-result.json`). On strict-082: 9.2567 → 9.2501 (Δ −0.0067; rule a
needs ≤ −0.010 → fail); +MEAN vs +RAND −0.0079 (rule b needs ≤ −0.010 → fail); folds improved 4/5 (c pass); without the 10
most-helped images +0.0053 (d fail). Consistent direction (fold deltas 0/−0.018/−0.009/−0.006/−0.001, P 0.82, gated member RMSE
12.52→12.43 in all four variants) but small and concentrated. Not advanced; no confirmation run.

## H4 — explicit spectral replacement of the residual-CNN correction (component stability; pre-registered 15:12, before any result)
H2 was withdrawn untested (repeat), so this is the third tested hypothesis.
- **Mechanism.** A4 shows the CNN correction depends almost only on phase-randomisation-invariant statistics (mean intensity +
  power spectrum). A ridge model on an explicit, low-dimensional version of those statistics should carry the same information
  with far lower estimation variance than a 70k-parameter CNN fitted to ~320 residuals, i.e. a simpler, more stable replacement.
- **Exact difference from prior work.** Same targets (y − inner-OOF T073 teacher, round4 `teacher_oof`), same 6 context inputs
  (`raw_context`), same A213→B107 coefficient / T320→V80 refit protocol and gate (noise ≥ 12, not failed) as the deployed CNN;
  only the image model changes: CNN → RidgeCV on the 23 pre-computed `h3/spec-train.npz` columns + context. The round4
  "statistics-only" control used just image mean/SD inside the CNN head; this uses the band spectrum and a linear readout.
- **Matched control.** C = same pipeline with the 6 context inputs only.
- **Bases.** Coefficient on B against strict-076 B predictions (`highnoise-relational-20261016/anchors`, p76B), clip [0, 1].
  Outer: replacement = p76V + λ·S; addition = p82V + λ_add·S with λ_add fitted on B against p82B.
- **Holdout-100.** Fit on the 400 dev rows with targets from the rescnn-retest partition-3 teachers, coefficient = mean of the five
  fold λ (frozen), base = its 400→100 teacher (the same base used for the CNN's holdout +6.2); compare S and CNN on its 35 gated rows.
- **Decision (fixed now).** *Supported replacement* if all: (1) RMSE(076+S) ≤ RMSE(082) + 0.010 on the 400 dev rows;
  (2) RMSE(076+S) ≤ RMSE(076+C) − 0.020; (3) holdout gated dMSE(S) < 0 with bootstrap P ≥ 0.8 and below the CNN's;
  (4) ≥ 4/5 folds improve vs 076. *Supported addition* if: RMSE(082+S) ≤ RMSE(082) − 0.020, bootstrap P ≥ 0.9, ≥ 4/5 folds,
  gain ≥ 0 without the 10 most-helped images, holdout not worse with P ≥ 0.8. Otherwise reject.
