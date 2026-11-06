# Secondary controls, fixed before any new CNN or outer result

2026-10-16 00:40 KST. No round4 CNN has yet been fitted. Teacher construction
continues unchanged; PLAN.md and its original hash remain preserved.

The round3 spatial contribution beyond image mean/SD was uncertain. Therefore
run the statistics-only control unconditionally, rather than only after a
positive primary result. It uses the same head, seed, training targets,
inner weight selection and outer folds, zeroing every convolutional feature.

Also test the previously evaluated auxiliary-objective arm as a secondary
hypothesis. In round3 its pooled gain was .04398, compared with .03520 for
the image-only residual arm, but its interval crossed zero. It adds the
unchanged .2 original-y auxiliary loss and six context inputs (honest T,
TabPFN minus Ridge, noise, phase fraction, log1p object count, coverage).
This may stabilize representation learning and condition the image correction
on the strength of the tabular prediction. All context normalization is fitted
on the CNN's training subset; supervised context is cross-fitted at the same
depth as its residual target. No new architecture or hyperparameter search.

Primary arm remains image-only residual40. Run both secondary arms once,
then repeat a CNN arm with fixed seed shift100000 only if its addition gain
is positive or its replacement beats deletion-only T. No averaging or
choosing seed weights using outer labels. Report every result and label all
intervals conditional and exploratory; adding a secondary arm increases
selection uncertainty. These inexpensive CNN fits reuse the same30 full
teachers and add no TabPFN training.

Before any new CNN run, also reserve a context-only control if the auxiliary
arm improves: zero all image features, retain the same six honest context
inputs, fit only the residual objective and choose its weight internally.
This distinguishes image information from a nonlinear recalibration of T.
Do not interpret an auxiliary-arm gain as spatial evidence without this check.
