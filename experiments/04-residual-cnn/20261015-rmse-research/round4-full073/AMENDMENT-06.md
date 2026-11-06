# Matched calibration controls after the noise-regime result

Noise-regime calibration improved aux-mean2 to9.31965(addition) /
9.25879(replacement). Before attributing that benefit to image information,
apply exactly the same calibration policy to both existing control families.

Train the statistics-only and context-only controls with the already fixed
second seed shift100000. Use the same fixed equal-seed mean and internal B
recalibration as the CNNs, then the identical threshold12 and B-only
leave-one-out shrinkage choice. No new threshold, hyperparameter or model
architecture. This is a fairness control, not a search for a better CNN.
Report both seeds and both means. Preserve all primary neural fits.
