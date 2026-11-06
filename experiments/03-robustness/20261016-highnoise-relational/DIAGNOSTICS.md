# Additional diagnostic lock before comparative scoring

The primary protocol and decisions in PLAN.md are unchanged. In addition to the
global affine projection already specified, decompose each fixed OOF change into
an affine component **within the pre-existing high-noise gate** and its orthogonal
remainder. Both are zero outside that gate. Report MSE/RMSE effects, never deploy
these OOF-derived projections. A claim of new image information must not rest
only on a gated spread/intercept effect.

Report held-out neighborhood geometry for the three arms using their saved
four-view-mean embeddings, training-only target normalization, and each excluded
80-image fold. This is a descriptive mechanism check, not another optimization
target or choice of candidate. Pairwise distances are dependent; do not give
them an inflated sample count. Online training KL uses different dropout/view/
batch conditions, so do not call a raw train/test KL difference an overfit gap.

No new weight, architecture, temperature, label-kernel bandwidth or routing
choice is introduced. Primary advancement still requires the preregistered
matched comparisons, uncertainty and contributor-removal checks.
