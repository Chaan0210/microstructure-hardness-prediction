# Step-change study — 2026-10-16 (stepchange-20261016)

## Bottom line
- **No step change was found.** Three new hypotheses were each tested under a protocol frozen before its results, and all three were rejected. Three further ideas were falsified without labels or on design grounds.
- The diagnosis indicates an **information plateau**, not a pipeline defect. Models built from pixels, grain objects and engineered features make essentially the same errors on clean and on noisy rows.
- **The incumbent 076 is retained, untouched and re-verified.** Its assembly chain reproduces byte-identically, and all 17,245 protected files are unchanged.
- **Strongest validated candidate: 082** (= 076 + residual-CNN correction). It was built earlier by another session, and this study re-audited it.
  - Expected private gain ≈ −0.026 RMSE, P(private better) ≈ 0.67.
  - Its dev gain is fragile to its top 20 images.
  - Promotion needs your approval. Nothing was submitted or reassigned.

## 1. Facts verified
- **Best submitted file: 076**, Public 8.9646534604, rank 5 (leader 8.5843 at 20:30). The page is in auto mode, so the final selection is 076.
- **082 was never submitted.**
  - 082 − 076 equals 080 − 078 exactly (max diff 0.0).
  - The 078 change (681 rows) and the correction (318 rows) do not overlap.
  - Therefore Public(082) = √(8.9647² + 8.9998² − 9.0299²) = **8.9343**. This is arithmetic from Public readings, not a measurement and not a fit.
- **Dev baseline** (Study900, 400 rows × 3 partitions): 9.5462.

  | band | rows | RMSE | share of SSE |
  |---|---:|---:|---:|
  | clean (noise < 9.5) | 205 | 7.98 | 36% |
  | mid (9.5–12) | 71 | 8.17 | 13% |
  | noisy (≥ 12) | 124 | 12.27 | 51% |

- **Noise is a discrete preset grid**, k × 0.7413 gray for k = 6..28. Hardness is independent of it: ANOVA across levels p = 0.41, Levene p = 0.10.
- **Calibration slopes** (y on prediction): clean 1.035, mid 0.996, noisy 1.248. The noisy value is the known atypical dev spread; the independent holdout gave 0.93.

## 2. Diagnosis
### Established (code- and prediction-level evidence)
1. **Shared error across fundamentally different extractors** (`work/diag/diagnose.py` → `diag.json`).
   - **Clean rows.** TabPFN's MSE is 61.6, and its residual covariance with the pixel CNN (CNN021) is 61.7 ± 7.7. Its covariance with the grain-object set model is 58.2 ± 6.1.
     - So essentially all of TabPFN's clean error is also made by models that see raw pixels or segmented grains.
     - Clean residuals are Gaussian (skew 0.08, excess kurtosis 0.05).
   - **Noisy rows.** TabPFN MSE 154.0; covariance with the CNN 156.8 ± 18.9.
   - **Tabular family.** Residual correlations among the tabular components are 0.85–0.98. Only CNN021 (0.83) and the set model (0.80) differ materially from the blend.
   - **What this does and does not say.** It is not proof of irreducibility. The covariance bounds label noise plus information none of these models extracts, and it includes a shared shrinkage term. What it does establish: none of the existing extractors carries the missing signal, so re-weighting or recombining them cannot recover it.
2. **Noisy rows are measurement-limited, and the extra data (degraded copies) is off-domain in its joint structure.**
   - The copies keep faint polygon edges and crisp, dark second-phase grains. Real noisy images show soft blobs with weak phase contrast (side-by-side montage at matched noise 14.1/17.0).
   - Copy-vs-real-noisy discrimination, label-free: multivariate AUC 0.76 (logistic) / 0.73 (GBM). The gap is spread over many columns; the largest single-column |AUC − 0.5| is 0.12.
   - Quantile-mapping the copies onto the real-noisy marginals lowers the logistic AUC to 0.63 but *raises* the GBM AUC to 0.79. The joint structure differs, not just the marginals.
   - The copies under-read elongation (copy 0.56 vs its source 0.63) but preserve grain size almost exactly (copy–source correlation 0.995). Real noisy images over-read grain size through merging at noise ≥ 17.
3. **The residual-CNN correction is information, not stretch, but its evidence is concentrated.**
   - On 076 (dev, original partition): +0.240, dMSE −4.50, 95% CI [−7.4, −1.8]. Stretch coefficient 0.01; non-stretch gain 0.226.
   - 93% of the gain lies in the top 20% of noisy rows by |tabular residual|. The correction correlates 0.35 with −residual.
   - The top 10 rows carry 81% of the net gain.
     - Without the top 10: −0.88 dMSE, P 0.85.
     - Without the top 20: +0.89, i.e. worse.
   - Per-fold gains: 0.45 / 0.01 / 0.63 / 0.04 / 0.00.
4. **Fold-to-full shift.**
   - On clean+mid rows, the prediction spread of test (final-500 fits) matches dev OOF: SD 14.67 vs 14.59.
   - On noisy rows, test predictions are narrower than dev OOF: SD 11.73 vs 13.12.
   - This matches the earlier finding that dev noisy rows are atypically spread (label SD 20.2 vs holdout 17.8). It is not evidence of a training-size defect.
5. **No preset-dependent magnification.** U-Net pore diameter is 7.6–8.0 px in every noise band. The rise in grain size at noise ≥ 17 is segmentation merging.

### Hypotheses (not established)
- Clean-row error is mostly label noise (SD ≲ 7.9). Supported by Gaussian residuals, the shared error, and the earlier learning-curve asymptote, but not identifiable.
- Real noisy renderings carry less phase information by design: the per-grain darkness modes are smeared, and copies are *more* informative than real noisy images. If so, part of the noisy excess is unrecoverable. The residual CNN shows that at least ~7% of the noisy MSE is recoverable.
- The leader's 8.58 cannot be explained from our data. Our evidence concerns our extractors only.

## 3. Ranked plan and outcomes
Order was set by mechanism × upside ÷ cost, with the falsification test fixed in advance.

| # | hypothesis (mechanism) | why ranked here | decisive test | outcome |
|---|---|---|---|---|
| E1 | Restrict TabPFN to copy-invariant columns so it can use the 732 copies on noisy rows (noisy rows are data-limited) | largest upside: 51% of SSE | gated RMSE of subset+copies (A1) vs all columns without copies (F0); shuffled-label control | **rejected** |
| E2 | Pairwise-difference regression (PADRE; n² pairs as a different learning problem) | new formulation for this project; cheap | standalone vs direct GBM; fold-honest blend gain ≥ 0.03 | **rejected** |
| E3 | Transfer ridge: noisy-domain coefficients shrunk toward clean coefficients (corrects pooled-coefficient misspecification under attenuation) | targets domain shift; cheap | same-weight replacement of the DTA RidgeCV partner on gated rows | **rejected** |
| — | Quantile-align copies to the real-noisy domain | E1 variant without dropping columns | label-free joint AUC | **rejected label-free** (GBM AUC 0.73 → 0.79) |
| — | Residual CNN trained with the permitted local transformations | user item 3 | design check | **rejected before running**: those transforms change the tabular teacher T, so y − T(original) is the wrong target. A correct version needs the full 12-member teacher recomputed per transformed image, which is infeasible here. |
| — | Preset magnification differs | would make pixel sizes non-comparable across presets | pore size by band | **rejected** |

Not repeated, because other sessions own them:
- Retargeting the residual CNN to the 076 teacher. The AI job is running; its 2-fold pilot correlates 0.9998 with the existing correction.
- U-Net class-weight probability calibration (`plateau-break-20261016-2035`).

## 4. Results (same 400 rows, official 3 partitions × 5 folds, all fitting inside training folds)
### E1 (`work/e1`)
TabPFN v3.5, seeds 0/1/2, u1dmavgH2Z anghcor columns. RMSE on gated rows, pooled over partitions; dMSE vs F0 is the paired bootstrap.

| arm | gated RMSE | dMSE vs F0 | clean RMSE |
|---|---:|---:|---:|
| F0: all 98 columns, real rows only | 12.314 | 0 | 8.23 |
| A0: 46 invariant columns, real rows only | 13.495 | +30.5 ± 11.2 | 9.00 |
| A1: A0 + all copies | **14.421** | +56.3 ± 17.1 | 10.36 |

- The copies made TabPFN worse even on columns where they are nearly in-domain, and they also damaged the clean rows.
- The stopping rule triggered, so arms C1, F1 and A2 were cancelled unrun.

### E2 (`work/e2`)
| model | all | clean | noisy | fold-honest blend gain over 076 |
|---|---:|---:|---:|---:|
| GBM direct | 11.907 | 10.53 | 14.66 | 0.000 (inner weights 0) |
| GBM-PADRE | 10.704 | 9.05 | 13.55 | **−0.011** |

PADRE is a large improvement over its own base learner (−1.2 RMSE). Its residuals still correlate 0.92 with 076, so it adds nothing to the blend.

### E3 (`work/e3`), gated rows
| partner | gated RMSE |
|---|---:|
| production DTA RidgeCV | 12.341 |
| plain pooled Ridge | 12.950 |
| transfer ridge (TR) | 13.201 |
| TR + copies (TRD) | 13.084 |

- Replacing the DTA RidgeCV partner in the blend, at the same weight: TR −0.050 (all three partitions negative); TRD −0.024.
- Noisy rows do not behave like an attenuated clean domain.

## 5. Candidate assessment and recommendation
- **Composition.** 082 changes only the 318 gated test rows: RMS 1.009, mean −0.383.
- **Evidence.**
  - Dev: +0.240 on 076 (original partition only, which is the partition the CNN was cross-fitted on).
  - Public: −0.0304 (exact arithmetic). Against an uninformative change of the same size, z ≈ −1.09.
- **Failure cases.**
  - The gain disappears if the 20 most-helped dev images are removed.
  - Fold 4's internally chosen coefficient was 0.
  - Rows the correction hurt include TRAIN_000271 (dMSE +156) and TRAIN_000157 (+75).
- **Compliance.** The correction's teacher relies on per-grain-offset U-Nets and boundary-blended copies. Your 10-16 DACON Q&A permits both.
- **Recommendation.**
  - Keep 076 as the protected incumbent and rollback.
  - 082 is the only candidate with positive expected private value: modest and fragile, but positive.
  - If 082 is submitted, its Public score will read 8.9343 and, in auto mode, it becomes the final selection automatically. That decision is yours.

## 6. Limitations
- **No untouched labels remain.** All 500 labels have been inspected by earlier sessions; the 400 dev rows were reused for selection many times; holdout-100 has also been used.
- The three dev partitions are not independent confirmations. Bootstraps here are conditional on the rows.
- E1/E2/E3 were evaluated once each, under their frozen protocols, with no tuning after the result.
- Reproduction covers prediction and assembly from frozen artifacts. Training of the U-Net, Cellpose, TabPFN and CNN021 was not re-run.

## 7. Remaining bottleneck and what could still move it
- **Bottleneck.** The noisy-row information channel. Real noisy images are less informative than any copy we can render, so flexible learners cannot use copies, and every extractor shares the same residual.
- **Next lever with real upside: information-matched copies.** Render copies matched in *information content* to real noisy images (heavier blur, reduced phase contrast, full boundary erasure).
  - Gate it label-free first: joint GBM AUC ≤ 0.6 against real noisy train rows, before any labeled run.
  - Past calibrations (deg5–deg12) failed this gate.
- **Lower upside:** variance reduction of the residual-CNN correction (multi-seed or inner-split bagging); the AI session is on it.

## 7a. Commands (from the project root; every output lands in this directory)
```
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/protect.py snapshot|verify
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/repro_chain.py chain|chain-post
(cd experiments/03-robustness/1300-robust-ensembles && ../../frameworks/tabpfn-v2/.venv/bin/python ../../04-residual-cnn/20261016-stepchange/repro/components/c14_partners.py .. ../../03-robustness/1200-degradation-augmentation deg,deg2,deg3,deg4 ../../04-residual-cnn/20261016-stepchange/repro/components/c14p-orig.npz)
(cd experiments/03-robustness/1300-robust-ensembles && ../../frameworks/tabpfn-v2/.venv/bin/python ../../04-residual-cnn/20261016-stepchange/repro/components/c14_blend.py .. abl ORIG076 fin-h9-tab-cpu.npz ../../04-residual-cnn/20261016-stepchange/repro/components/c14p-orig.npz ../../04-residual-cnn/20261016-stepchange/repro/components/c14b-orig)
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e1/domain_gap.py u1dmavgH2Z      # label-free
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e1/qm_gap.py                     # label-free
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e1/e1_tab.py {F0|A0|A1|C1|F1|A2} mps
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e1/e1_eval.py
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e2/padre.py && .venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e2/e2_eval.py
.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e3/transfer_ridge.py && .venv/bin/python experiments/04-residual-cnn/20261016-stepchange/work/e3/e3_eval.py
```
Incumbent provenance, checksums, environment and verification are in `INCUMBENT.md` and `record/`.

## 8. Follow-up (2026-10-16 21:40–23:50): info-matched copies pilot, 082 projection audit
- **Info-matched copies: branch closed.** Pilot in `experiments/03-robustness/20261016-infomatched-pilot/REPORT.md`; it took over the child session's renderer and pipeline.
  - On real held-out noisy rows, adding the new copies to the production DTA partners gives:

    | learner | dMSE | P(better) |
    |---|---:|---:|
    | Ridge | +1.1 ± 6.7 | 0.43 |
    | logRidge | +1.3 ± 6.8 | 0.42 |
  - That is no better than duplicating the same source rows (the weight control).
  - TabPFN gets worse with them: dMSE +23.6 ± 14.7, worse in all partitions. A duplicate-only control is neutral (+0.2).
  - Copies are roughly information-matched in R² terms (0.42–0.46 vs 0.48 for real noisy rows).
  - Real noisy rows have a somewhat wider hardness spread (SD 19.6 vs 17.1, CI of ratio 0.99–1.33). Copies cannot reproduce that, and it is not a rendering defect.
- **082 private projection: corrected.** The earlier single estimate ("−0.026, P 0.67") is withdrawn as a point claim.
  - The recorded Public gain of −0.0304 on 300 rows is a measured fact.
  - Its correction's dev evidence rests on one partition and about 10 images.
- **Reproducibility after this follow-up.**
  - 082 rebuilt byte-identically from 076/078/080 via the AI assembly script (`repro/082/`).
  - The 076 chain rebuilt byte-identically (`repro/chain-post2/`).
  - 17,246 protected files unchanged; no file modified or deleted.
