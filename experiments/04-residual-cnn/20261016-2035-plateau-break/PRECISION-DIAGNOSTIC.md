# Locked train-only precision diagnostic

Registered after reproduction discovered a CPU-versus-default-MPS difference.
This is a reproducibility/validation diagnostic, not a fourth performance
candidate and not a score-based choice of precision.

Use one existing physical bank: H9, slot038, u1dm1sz, same nb/full noise12 gate.
Use original development folds0–4 and seeds0/100/200 plus the original fold
offset. Each fit uses exactly320 training images and predicts its80 excluded
images. Compare CPU automatic (the production policy), MPS automatic (historical
development policy), and explicitly float32 MPS. Every arm has the same inputs,
targets, pretrained checkpoint and8 estimators. No test images/predictions are
used by this diagnostic. Saved source OOF is a separate reproduction reference.

Primary outcomes: paired prediction RMS and max discrepancy, aggregate RMSE and
CPU-versus-MPS paired error differences. Compare identical-size fits; no claim
about213→320→500 transfer is possible from this test. Read all five folds only
after completion. No coefficient, gate, feature, seed or candidate is selected
from these comparisons. Complete15 fits per backend, each query group split by
its fixed existing feature gate; in practice30 inference contexts per backend.
Use existing native model cache and bounded MPS memory; schedule after incumbent
replay rather than running two sizable inference jobs at once. If the diagnostic
exceeds15minutes or memory cap, preserve completed folds and report it incomplete.

CPU/MPS float32 differences≤.001 HV are a numerical equivalence screen, not a
claim of bitwise reproducibility. The fixed threshold corresponds to a maximum
.000384-HV RMSE perturbation if an entire0.384-weight Tab block differed by that
amount; it is much smaller than the predeclared.03-RMSE experiment threshold.
