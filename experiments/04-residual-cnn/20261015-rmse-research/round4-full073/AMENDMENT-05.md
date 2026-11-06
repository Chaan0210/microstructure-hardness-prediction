# Noise-regime calibration — explicitly post-analysis exploration

Observed mean2 replacement diagnostics: image-only CNN gain versus T is
−.00046 on noise<12 and +.26491 on noise>=12. Auxiliary CNN gives −.09472
and +.44894 respectively. Auxiliary benefit is concentrated in a few large
errors, and its intervals are wide. These findings motivate this secondary
experiment; they are not an independent confirmation dataset.

Use the EXISTING073 boundary12, never search a threshold. Do not manually
zero the low-noise weight using these outer results. Instead, for each
original outer fold and each mean2 predictor, learn two bounded coefficients
from B107 only, shrinking them toward the global B coefficient. Select the
shrinkage strength by leave-one-out prediction entirely inside B. The B
predictions come from A-only models, so each calibration leave-out label
also cannot affect those base/CNN predictions.

Fixed pseudo-sample strengths: global-only,100,30,10,0, in that order to
prefer more pooling for exact ties. The penalty is strength times mean
squared correction on the calibration training rows. Refit selected weights
on all B and apply them once to V. Both addition and replacement are tested;
the same fallbacks and frozen CNN predictions remain in place. No new neural
or TabPFN training. Compare against the corresponding global-weight mean2,
not just073. Report all outcomes as exploratory and include the new tests
in the conditional simultaneous bands.
