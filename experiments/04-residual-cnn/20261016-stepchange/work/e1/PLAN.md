# E1 — domain-invariant columns + degraded copies in the noisy-row TabPFN (frozen 2026-10-16 21:10, before any label-based result)

Mechanism. Noisy rows (noise >= 12) carry 51% of dev SSE and are data-limited (earlier learning curve), but TabPFN could not use the 732
label-preserving degraded copies because copies sit in a different feature domain (multivariate AUC 0.76 here). Linear partners already use them.
Hypothesis: restricting TabPFN to columns whose copy-vs-real distributions match (label-free selection) lets TabPFN use the copies, and the extra
noisy-domain rows outweigh the information lost with the dropped columns.

Label-free selection (done, `domain_gap.py`, no labels read): 45 columns with |AUC(copy vs real noisy) - 0.5| <= 0.05 (multivariate AUC 0.65) + `noise`.

Fixed protocol.
- Rows: Study900 dev 400 x official 3 partitions x 5 folds; evaluation rows = noise >= 12 & not failed (124 per partition). U-Net variant u1dmavgH2Z,
  feature file push-800/feat-u1dmavgH2Z-anghcor-train.npz (production 044-slot base columns, without the H9 additions).
- Copies: deg, deg2, deg3, deg4 rows with measured noise >= 12 whose source is in the outer training fold (copies of validation sources excluded).
- Model: TabPFN v3.5 default estimators, seeds 0,1,2 (100*seed + fold, as production dev_eval.py), MPS. Context = all training-fold real rows (+ copies).
- Arms: F0 all 98 columns, real only (reference) | F1 all columns + copies | A0 invariant subset, real only | A1 subset + all copies |
  A2 subset + deg4 copies only | C1 = A1 with copy labels permuted across sources (control).
- Primary metric: gated-row RMSE pooled over the 3 partitions; paired image bootstrap (4000) P(arm better than F0).
- Success: A1 (or A2) beats F0 by >= 0.15 gated RMSE with P >= 0.9, beats A0, and C1 does not beat A0. Only then a blend-level test
  (same-weight replacement of the TabPFN block on gated rows) is run. Otherwise E1 is rejected; no dose or subset tuning afterwards.
