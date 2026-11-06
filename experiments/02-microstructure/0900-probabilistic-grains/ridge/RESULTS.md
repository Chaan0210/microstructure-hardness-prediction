# Ridge confirmation results

Conditional comparison on reused 400-image development data.

| Method | Pooled RMSE | Gain vs 045 | Image-ID paired 95% interval | Better folds |
|---|---:|---:|---|---:|
| baseline045 | 9.940240 | +0.000000 | [+0.000000, +0.000000] | 0/15 |
| weight_only_084 | 9.903711 | +0.036529 | [-0.002190, +0.074374] | 12/15 |
| ridge | 10.237852 | -0.297612 | [-0.713957, +0.131224] | 2/15 |
| ridge_20 | 9.831835 | +0.108405 | [+0.008665, +0.218243] | 13/15 |
| ridge_same021share | 9.877356 | +0.062884 | [-0.013991, +0.149481] | 12/15 |
| fgls_ridge | 10.155052 | -0.214812 | [-0.643869, +0.226826] | 4/15 |
| fgls_ridge_20 | 9.834183 | +0.106057 | [+0.008751, +0.210478] | 14/15 |
| fgls_ridge_same021share | 9.880119 | +0.060121 | [-0.014895, +0.141026] | 11/15 |

Positive gain means lower RMSE. All designs and blend weights were fixed in METHOD.md before scoring.
The primary 20% blend decreases the old 021 component from 20% to 16%; the same021share comparator preserves it.
The weight-only control measures the effect of that change without adding Ridge.

All 30 fold models reproduced predictions exactly after reload. Held-out label poisoning left all fits unchanged.
Inputs were aligned explicitly by image ID and feature name; permuting stored rows and columns reproduced the same design.
Models, training/validation IDs, the weighted method's inner folds, input hashes, and all metrics are saved locally.

These folds and legacy representations were already used in model development. The intervals do not adjust for historical selection.
No public/private improvement is established; no test predictions or submission were produced.
