# Small CNN representation experiment under the resource constraint

The full076 teacher pilot exited137 after3of12 teachers/252of864 Tab members.
Its cause is unconfirmed; memory/swap pressure was observed. Do not report a
score or restart that large stage repeatedly. Preserve all caches.

New hypothesis: high-noise structural information is easier to learn when
the CNN receives both the native image and a fixed smoothed image. Sigma1 is
fixed from the already successful H9 measurement scale, not selected here.
Compute a normalized5x5 Gaussian kernel, reflective padding, inside each
minibatch. No separate full-image float cache or GPU allocation is needed.

Parameter-matched control: two identical raw channels. Candidate: raw plus
sigma1 Gaussian. Only the first convolution changes1->2 channels; raw input
mean/std and all downstream layer widths/pooling/context/auxiliary loss stay
unchanged. Both arms have identical initialization, batch order, augmentation,
optimizer and40epoch schedule for a given seed. Original single-channel CNN
is a separate legacy control. This is new to the residual branch; the earlier
blur augmentation of an old direct-y CNN did not test this two-stream control.

Reuse verified nested full073 teachers, as the current deployed correction
does, and compare both arms on strict076's original5fold baseline. This is
NOT a completed test of training on full076 residuals. First seed is
2026101501+100*fold; train both arms on A213 andT320,20fits total. Compute
coefficients only from B using the original noise-regime LOO selector and
base073 B predictions, exactly as the reference correction. High-only gate
and failure fallback are fixed. Also compare with original frozen coefficients
to identify gains that depend only on coefficient selection.

Advance to seed offset100000 only if the Gaussian arm improves>=.01 against
BOTH the matched one-seed legacy CNN and the duplicate-channel control,
improves>=3folds against each, and has positive frozen-weight gain over the
legacy CNN. Complete all5folds before reading comparisons. If advanced, repeat
both arms and use fixed equal seed averaging followed by B-only coefficient
selection. Promotion additionally needs improvement over the existing2seed
candidate with conditional95% gain interval above0 and>=3folds improving.

CPU1,one process, no test inputs, no external data/weights. Reuse one normalized
image array between arms, release it between stages, record peakRSS and stop
further fits if peakRSS exceeds1GiB. Retain all negative results; no new Public
evaluation or weight fitting.
