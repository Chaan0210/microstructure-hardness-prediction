# Completed degradation-copy pilot: independent audit

The user requested a status check, result evaluation, and fold-honest validation against082 only if credible signal exists. The pilot's updated handover says completed/closed at23:50 KST on16 October. Both copy feature chains and all planned OOF caches exist; process inspection found no pilot training/rendering job active. Its recorded primary results have already been inspected: Ridge/logRidge slightly worse, TabPFN substantially worse. This is an audit of existing evidence, not a claim that those results are unseen.

No original pilot file or082/076 dependency will be edited. Extend the preceding protection inventory with the completed SSL study and completed pilot. Every new output stays here. Training data only, existing local CPU environment, two threads, no installs, paid compute, GPU, submissions or test predictions. Preserve the previous exact incumbent inference records and verify their artifact hashes; this audit does not reread test pixels for modeling or validation.

## Fixed audit and stopping rule, before new refits

Budget: at most20 minutes of new CPU refitting, one sequential process,1GiB peak-RSS cap. Recompute saved OOF statistics independently with ID/label/fold alignment checks. Average squared errors across the three partitions per source BEFORE uncertainty estimation. Reproduce the pilot's4000-draw RandomState0 bootstrap and also report paired95% RMSE-gain intervals. Evaluate both all400 real sources and the124 eligible noisy sources, original-fold signs, and removal of top5/top10 helping sources. These reused partitions are correlated evidence, not independent confirmations.

Replay Ridge and logRidge R1(production copies), R3(production+new), RD(duplicated source control) on the same3x5 folds and source-grouped5fold alpha selection. Use the original grids, columns, imputation, scaling and log back-transform. All preprocessing fits inside the relevant training subset. Compare every prediction and selected alpha with the saved output: exact where deterministic, otherwise tolerance1e-7 hardness units for BLAS/thread numerical roundoff; report the observed maximum. Any substantive mismatch triggers investigation rather than result acceptance. No new candidate or blend is chosen from these refits.

## Control flaw and bounded repair

The original RS permutation is constructed across all249 clean sources, then looked up after outer source filtering. Therefore some actual training targets come from outer-validation and historical-holdout labels. That control is invalid for strict source isolation. Do not reproduce the leaking fit. Quantify donor leakage using source IDs alone.

Run one corrected RS control for Ridge and logRidge on the same3x5 folds. Permute NEW-copy targets only among source IDs present in the actual fitting subset, separately for each inner training fit and final outer refit; both views of a source share its donor. Production-copy and real targets remain unchanged. Seed2026101701+100*partition+10*outer_fold+inner_fold; final-refit offset9. Reuse the same per-inner-fold permutation across all alpha candidates. Alpha selection uses true real noisy validation targets, while no training donor may belong to that validation subset. Save source/donor exclusion records and a fold0 full-fit replay with outside labels poisoned. This is a repaired negative control, not a new predictive hypothesis.

## Advancement gate

Retain the pilot's frozen gate for the three primary comparisons: R3vsR1 for Ridge/logRidge; T2vsT0 for TabPFN. Require mean dMSE<0, bootstrap fraction favorable>=.95, beneficial signs in all3 partitions and>=4/5 original folds, positive gain after removing top5, and improvement beyond the applicable duplicate/shuffled controls. The corrected RS cannot rescue a primary comparison that fails those criteria. If no primary passes, stop without StageB, rendering,082 reblending or parameter search. The failure does not prove every possible combination with082 would fail; it supplies no credible trigger for that expensive conditional stage.

If a genuine primary signal survives the audit, separately freeze StageB's fold-local renderer/auxiliary-feature/calibration procedure and runtime budget before executing it, then evaluate against the frozen082 OOF using train-only nested selection. This conditional step is not assumed warranted.

## Interpretation checks

Audit the D2 sample counts and the claim of information matching. Predictive R² on different label populations is not an equivalence test of image information. Source-copy measurement correlations are not true physical measurement accuracy. All-train pixel calibration in StageA can change either direction of performance; it is not guaranteed to be optimistic. Preserve these limitations and distinguish component scores from full082 ensemble scores. The inherited baseline representations and all400 labels have historical exposure.
