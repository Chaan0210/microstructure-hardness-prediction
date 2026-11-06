# Unexpected control result: one bounded seed confirmation

The pre-registered Gaussian study failed: matched one-seed Gaussian RMSE
9.236777 versus legacy9.244220 (gain.007443, CI[-.048598,.063856]) and
duplicate-channel control9.227371 (Gaussian worse.009405). Do not advance
Gaussian or claim that smoothing is beneficial.

The duplicate control unexpectedly improved the point score. Its architecture
has the same function space as the old CNN: conv([x,x],[W0,W1])+bias is
conv(x,W0+W1)+bias. However, first-layer initialization and AdamW dynamics
differ. Thus its gain is not evidence of new image information or of smoothing;
it could be first-layer optimization, initialization, or sampling variation.

Run ONLY the duplicate control at the already fixed second seed offset100000,
same five folds, selector and outer fit (ten new fits, CPU1). Preserve all
first-seed files and the Gaussian failure. No new learning rate/epoch search.
Collapse the first convolution of completed duplicate models and check their
query predictions against the two-channel models to verify function equivalence.

Evaluate fixed equal mean of two duplicate seeds, with B-only original
noise-regime calibration, on the same strict076 base. Also use the original
frozen coefficients. Require improvement>=.01 over the current two-seed CNN,
>=3 improving folds, conditional95% lower bound>0, and positive frozen-weight
gain before production. Report both seed directions and the matched old seed
controls. If this fails, do not train a full500 variant or produce a replacement
submission. This is a post-analysis confirmation on reused labels, not fresh data.
