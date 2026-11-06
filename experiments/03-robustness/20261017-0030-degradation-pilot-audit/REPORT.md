# Degradation-copy pilot audit — 17 October 2026

**Retain082 and076. The completed pilot supplies no credible incremental signal, so its conditional StageB validation against082 is not triggered.** This conclusion follows the pilot's original decision rule after independently reproducing its statistics, replaying its valid linear fits, and repairing a leaking shuffled-label control. It is not a claim that every possible augmentation or combination with082 must fail.

## Status and completed work

The updated pilot handover records completion/closure at23:50 KST on16 October. Feature chains finished at23:21:28(v1a) and23:31:37(v2a). Process inspection found no remaining pilot rendering/training job. Both249-source variants, all498 feature rows, Ridge/logRidge OOF arms and TabPFN T1/T2/duplicate caches exist. The additional v1b/v2b images were rendered but not featurized, as the pilot's amendment specified.

This audit kept that completed directory read-only and wrote only into this separate experiment directory. It performed:

- ID, target, feature-column and fold alignment checks; independent recomputation of13 saved pairwise comparisons, matching their MSE, bootstrap fractions, fold signs and influence-removal statistics.
- **90 valid outer linear fits replayed exactly:** R1(existing copies), R3(existing+new copies), RD(source-duplicate control), two learners, three partitions and five folds. All selected alpha values and predictions matched; maximum difference **0**.
- **30 repaired RS outer fits**, with source-local target permutations inside every inner training fit and final refit. No original leaking fit was rerun.
- A complete corrected fold0 Ridge fit repeated after poisoning all180 labels outside its320-source training subset, with dataset reads blocked. Predictions were exactly unchanged.

Refitting and verification took44.16 seconds with two CPU threads; peak RSS105.61MiB, within the declared20-minute/1GiB budget. No installations, GPU, paid compute, new rendering, uploads or submissions were used. TabPFN's existing predictions and fitting code were audited; its45-context-per-arm training was not rerun.

## Primary results

The three partitions reuse the same400 development sources. Squared errors were averaged per source across partitions before4000 paired source bootstrap draws. The principal evaluation set is the same124 eligible noisy real images in each partition. These are conditional development estimates, not independent confirmations or private-leaderboard forecasts.

Gain means reference RMSE minus new RMSE: positive is better. R3 adds new copies to the existing production-copy training set; TabPFN T2 adds new copies to real-only training.

| Primary comparison | Reference noisy RMSE | New noisy RMSE | Gain | Conditional95% gain interval | Gain after removing top5 helping sources |
|---|---:|---:|---:|---|---:|
|Ridge R3 vs R1|12.25724|12.30255|−0.04531|[−0.56376,+0.47722]|−0.32617|
|logRidge R3 vs R1|12.03224|12.08771|−0.05547|[−0.59765,+0.48177]|−0.35076|
|TabPFN T2 vs T0|12.31397|13.23630|−0.92233|[−1.97891,+0.12722]|−1.51655|

The bootstrap fractions favoring improvement were0.431,0.419 and0.046. These are resampling summaries, not probabilities of private-leaderboard success. The95% intervals all include zero; the result is **absence of credible improvement**, not a two-sided95% proof that every arm is harmful.

Ridge/logRidge improved in only one of three partitions and three of five original folds. TabPFN worsened in every partition and four of five original folds. Removing the ten largest helping sources made the respective gains−0.58213,−0.61532 and−1.82975. None passes the original requirements for mean benefit, bootstrap support, partition/fold consistency and influence robustness.

For completeness, the same raw component predictors on all400 images scored10.49486->10.62085(Ridge),10.40224->10.55117(logRidge), and9.77480->10.47058(TabPFN). These are **component** scores, not full082 ensemble scores; low-noise deployment uses different partners/gates. The frozen082 reference remains9.256739 overall and11.508128 on the124 noisy rows for its original conditional OOF protocol. No new full082 ensemble candidate is claimed or selected here.

## Controls: actual signal, no demonstrated extra value

Against duplicated source rows, R3's noisy-RMSE gains were−0.01886 for Ridge and+0.00395 for logRidge, with broad intervals[−0.57617,+0.56147] and[−0.56296,+0.58295]. Both reversed to losses greater than0.34 after removing five helpful sources. TabPFN with new copies also lost to its duplicate-only control, which itself approximately matched real-only training. Thus copying a label distribution more often does not explain all of the observed TabPFN degradation, and the new linear-copy arm has no demonstrated advantage over that source-weight control.

The original RS control was invalid. [pilot.py:58](../20261016-infomatched-pilot/work/pilot.py#L58) constructs a permutation across all249 clean sources; [line50](../20261016-infomatched-pilot/work/pilot.py#L50) then uses the donor's label after filtering only the image source. Across outer folds, **104–126 new-copy training rows received labels from outside the outer training partition**, including52–76 from the historical holdout100. Grouping the image derivatives did not isolate their assigned target donors. The genuine R1/R3/RD/T2 copy-source filters passed the audit; this target-donor flaw is specific to the shuffled control under examination.

The corrected RS permutes only among new-copy source IDs present in the actual fitting subset. Both derivatives share their source's donor; real and production-copy labels stay unchanged. Separate deterministic permutations are formed inside each inner training split and outer refit. The same inner permutation is reused across all alpha candidates. Every donor is checked against that fit's source set.

| Corrected comparison | Correct-label RMSE | Shuffled-control RMSE | Gain | Conditional95% interval | Gain after top10 removal |
|---|---:|---:|---:|---|---:|
|Ridge R3 vs corrected RS|12.30255|13.40836|+1.10581|[+0.22701,+1.98939]|+0.11723|
|logRidge R3 vs corrected RS|12.08771|13.20709|+1.11939|[+0.26797,+1.97294]|+0.16504|

The repaired control supports a narrower positive finding: the copies retain a usable relationship to their source labels. That does **not** establish incremental value over existing copies, useful invariance, physical accuracy, or benefit to082. It cannot rescue the failed primary comparisons.

## Diagnostic corrections and bottleneck

The D2 output is labeled `real_noisy_n161`, but its code actually uses **159** real feature rows versus **161** copied sources per draw. The renderer's raw-noise pool has161 images; the feature-based membership excludes TRAIN_000019 and TRAIN_000060. The real/copy diagnostic therefore is not exactly size matched. The two-source difference is not asserted to explain the primary failure; the primary124-source comparison was aligned and reproduced correctly.

More fundamentally, similar normalized Ridge predictability on different label samples is not an equivalence test for information content. The separate D3 diagnostic observed label SD17.13 in249 clean sources versus19.65 in the161-image raw-noise pool, but its SD-ratio interval[0.99,1.33] includes1; a population-level spread difference is not established. Repeated draws reuse those same sources. Reduced source-copy correlations in phase fraction and pore fraction are properties of the measurement pipeline, not ground-truth proof of which physical information was erased. The original phrase“information-matched” should be treated as a hypothesis that this predictive pilot failed to validate for transfer.

The renderer was calibrated on pixels of all training images, including the validation folds. Its shared auxiliary representations also retain historical fitting limits. This makes StageA exploratory. Contrary to the original report's wording, that calibration need not **only** flatter results: changing renderer parameters can change predictions in either direction. Failure here does not prove a fold-local renderer would fail. There is nevertheless no credible incremental result to trigger that expensive next stage under the frozen rule.

The clearest supported bottleneck is a **transfer mismatch**: correct-label degraded copies contain signal, yet their feature/target relationships do not improve prediction of real noisy sources beyond existing copies or source duplication. The evidence does not identify whether phase measurement shifts, pore/heterogeneity errors, the copy-source label distribution, model context effects, or another interaction causes that mismatch. No claim of irreducible image ambiguity follows.

Examples of harm are recorded for every source in per-source.csv. TRAIN_000186 has hardness184.226 and additional partition-averaged squared error405.38(Ridge),452.20(logRidge) and1265.27(TabPFN). TRAIN_000467, hardness151.980, adds277.52/255.20 in the two linear arms. These are substantial failures, not just a uniform tiny change. Beneficial sources are retained in the same ledger and in the top-contributor analysis; none were removed from the primary score.

No additional renderer operation was executed. The inspected historical renderer includes blur, noise and shading in addition to regional photometric changes and boundary replacement. This audit does not expand the two explicit Q&A permissions into authorization for arbitrary generation or new operations.

## Decision, preservation and reproducibility

**StageB is not warranted;082 remains the incumbent and076 the rollback.** No new coefficients were fitted against082 and no negative pilot was turned into a posthoc blend search. An untested full ensemble is not assigned a score. All three primary comparisons failed their benefit requirements, even after the defective negative control was repaired.

All **152,316 protected files** matched their initial hashes: both incumbent pipelines, shared upstream assets/environments, the completed SSL study and the completed pilot. Git head and tracked/staged diffs also matched. Both incumbent CSV hashes remained unchanged. The previous byte-exact082/076 inference-reproduction records are included in the protection inventory; this audit did not repeat test-image inference, and did not use test predictions for fitting or evaluation.

Audit protocol and code locks: PLAN.md and code-lock.json. Exact commands and input provenance: RUN.md and protected.json. Recomputed statistics: audit-before-refits.json and audited-results.json. Refits, alpha grids, source/donor records and poison replay: refit-verification.json and refit-oof.npz. Compact results: comparison.csv and per-source.csv. Conservation evidence: starting-state.json and preservation.json. The original pilot files remain untouched, including its superseded shuffled-control results.
