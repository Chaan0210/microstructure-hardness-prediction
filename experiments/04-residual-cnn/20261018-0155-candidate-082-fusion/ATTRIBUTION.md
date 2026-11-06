# Additional calibration attribution control

Frozen before any fusion outer score is evaluated. Primary selection, models,
penalties, grids and acceptance rule are unchanged. The conditional fusion can
exploit noise/log-count routing even without a new predictor, so constant
oldCNN-only calibration is not a sufficient new-information control.

Also fit a control that zeros all new predictor columns and chooses, insideB
LOO, between082,constant oldCNN calibration and quality-conditioned oldCNN
calibration, using the SAME3lambdas and quality normalization/corner constraints.
No new-information claim without an improvement over this stronger control.
It is an attribution ablation, never an outer-selected replacement candidate.
