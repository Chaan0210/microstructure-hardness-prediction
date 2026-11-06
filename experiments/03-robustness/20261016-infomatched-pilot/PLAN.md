# Info-matched copies — bounded pilot (frozen 2026-10-16 21:50 KST, before any copy was rendered or any label-based result was read)

## Ownership and handover
- This directory takes over the child session "Information-matched noisy copies, label-free gate" (`local_07c6a848…`). Its turn was stopped at 21:41, mid-sweep.
- Its directory `experiments/03-robustness/20261016-infomatched-copies/` is left untouched and read-only. What it contains:
  - `work/render.py` (renderer)
  - `work/calib.py`, `work/pixstats.py` (label-free pixel calibration)
  - `pipe/` (copy feature pipeline), verified on a 10-image check set
  - `work/calib/sweep.log`: pixel-level copy-vs-real AUC 0.90–0.98 for every variant tried (base 0.967/0.930)
- **Corrected criterion.** The child's gate (AUC ≤ 0.60) is retired as a success criterion.
  - Discrimination AUC is a diagnostic only. AUC near 0.5 can mean the signal was destroyed; a high AUC does not prove the copies are useless.
  - The decision below rests on prediction for REAL held-out noisy train images.

## Variants (fixed before any label use; at most 2)
Both use `render.py` copied unchanged. Parameters come from the child's label-free calibration on pixel statistics of all real noise ≥ 12 train images, so they are **exploratory** (see Stage B).

| variant | parameters | what it is |
|---|---|---|
| V1 | `BASE` | phase-contrast compression of dark grains MU_D 0.675; matrix grain-to-grain spread; boundary erasure P 0.8–1.0, BW 1; blur U(0.75, 1.05); shading; white noise to a target drawn from the real noise ≥ 12 pool |
| V2 | `BASE` + KEEP_MEAN 1, BW 2, P_LO = P_HI = 1 | complete 2-px boundary erasure, mean restored |

- Sources: the 249 train images with noise below the train median (the same sources as production copies deg..deg4). Masks: `grain-cellpose-220/masks-v3-train.npz`.
- Two copies per source per variant (different seeds and noise targets): 996 copies.
- Features: the production copy pipeline (`pipe/run_set.sh`, U-Net variant u1dmavgH2Z, anghcor 98 columns).
- Operations used are only those your DACON Q&A permits: grain-region brightness/contrast, boundary blending/replacement within the same image, global blur/noise/contrast.

## Diagnostics (not decision criteria)
- **D1 structure preservation.** Copy–source correlation of the key physical columns (u_ph, het4_u, grain_eqd_mean, m_el_pix, px_pore_frac, ang_c4, sizefield_std_s32) for V1, V2 and production deg4. Compared with how far real noisy rows sit from their own population.
- **D2 label information.** Source-grouped 5-fold Ridge CV RMSE predicting the source label from copy features alone. Compared with the same model on real noisy rows (n = 159), using a size-matched subsample of copies (1 copy per source, 159 sources, 20 draws).
- **D3 hardness distribution, clean sources vs real noisy rows.** Label mean, SD, KS and Levene tests. Plus structure columns that are robust to rendering (pore diameter, ang_c4) as a check on structure differences that are not rendering defects.

## Predictive evaluation (decision)
- Rows: Study900 dev 400, the official 3 partitions × 5 folds. Evaluated on REAL held-out rows with noise ≥ 12 and not failed (124 per partition).
- Copies enter training only when their source is in the outer training fold, so every source and all its derivatives share a fold. Holdout-100 sources are never used.
- Learners and arms, all on the u1dmavgH2Z anghcor columns:

| learner | arm | training rows |
|---|---|---|
| DTA RidgeCV and DTA logRidge (alpha by source-grouped inner 5-fold CV scored on real noise ≥ 12 rows, as production `c14_partners.fitB`) | R0 | real only |
| | R1 | real + production copies (deg..deg4, noise ≥ 12) — the production configuration |
| | R2 | real + new copies (V1 + V2) |
| | R3 | real + production + new copies |
| | RD | weight control: real + production copies + duplicates of each new-copy source's own clean row, with the same multiplicity as R3's new copies (same extra source weight, no new domain information) |
| | RS | R3 with the new-copy labels permuted across sources |
| TabPFN v3.5, seeds 0/1/2 (100·seed + fold), MPS | T0 | real only (= E1 F0, reused) |
| | T1 | + production copies |
| | T2 | + new copies |
| | TD | duplicate control matching T2 |

- **Primary comparisons:** R3 vs R1 (Ridge and logRidge) and T2 vs T0.
- **Statistics:**
  - Per-image squared errors are averaged over the 3 partitions first, so the partitions are treated as correlated evidence, not three confirmations. Image bootstrap with 4000 draws.
  - Per-partition and per-fold (original partition) dMSE, and removal of the top-5 and top-10 influential images.
- **Credible predictive benefit requires all of:**
  - dMSE < 0 with P ≥ 0.95
  - the same sign in all 3 partitions and in at least 4/5 original-partition folds
  - still < 0 after removing the top-5 helped images
  - better than both RD and RS
- **Stage B, only if Stage A passes:** re-render with the calibration fit on training-fold real noisy images only (noise pool and phase-contrast moment), keep the same arms, then run a blend-level test. Stage A alone is reported as exploratory because its renderer saw all training images' pixels.

## Budget and stop rule
- At most 2 variants and at most 10 labelled arms. At most about 2.5 h of wall clock in total.
- No renderer search after any label-based number is seen.
- If the primary comparisons fail, the branch is closed, and the next hypothesis is chosen from the broader bottleneck analysis.

## Amendment A1 (2026-10-16 21:52, before any label-based result)
- The copy feature pipeline runs at about 10 s per image (4-view Cellpose), so 4 sets would take about 3 h and break the budget.
- Stage A therefore uses **one copy per source per variant**: sets v1a (V1) and v2a (V2), 498 new copies, run in parallel.
- v1b and v2b were rendered but are not featurised. They are used only if Stage A's primary comparison shows promise (then together with Stage B).
- All arms, statistics and decision rules are otherwise unchanged. `NEW` in `work/pilot.py` = [v1a, v2a].
- Also recorded: after its turn was first stopped at 21:41, the child session restarted a pixel sweep (`sw2.json`). It was stopped again at about 21:50, its processes ended, and a handover message was sent. No result from that sweep is used.
