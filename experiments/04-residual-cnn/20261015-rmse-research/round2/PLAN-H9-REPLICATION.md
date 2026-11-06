# Frozen H9 seed replication, before extra-seed scores

H9 one-seed all-500 result: gain 0.0166058; partition gains
0.0229336 / -0.0299801 / 0.0566859; conditional ID-bootstrap CI includes zero.
The physical-minus-baseline prediction change has variance 0.87861 HV², while
the observed squared-error reduction is only 0.32372 HV². The random two-column
control also changes TabPFN's stochastic feature/preprocessing realization.
This economical one-bank, one-seed prototype is much less averaged than the
production 073 block. Replicate to assess this source of uncertainty; do not
adjust features, mixing weight, noise gate, or select a seed.

Run two additional seeds, original per-fold seed +100 and +200, for EACH of
the baseline, physical-feature and identical random-feature arms. Same 15
outer folds, same training/query rows, same n_estimators=4, CPU two threads.
Average all three seeds equally; retain both the original one-seed results
and the matched three-seed benchmark. Other ensemble components stay fixed.
No branch is chosen by intermediate scores, and no failed seed is dropped.

If positive point gain remains but the interval crosses zero or a partition
still worsens, call the result inconclusive; do not promote a final model or
create an allegedly improved official submission. Preserve all fitted input
definitions, inference outputs and the negative control.
