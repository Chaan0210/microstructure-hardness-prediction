# Low-memory complementary probe while large teacher fitting is interrupted

The actual076 teacher pilot was interrupted for observed resource contention,
not for a score. One completed teacher and118 individual Tab predictions are
retained and readable. No other process was signaled. The pilot is incomplete.

New fixed hypothesis: two CNN seeds that disagree strongly about a correction
may deserve attenuation. For seed corrections r0,r1, define m=(r0+r1)/2 and
v=(r0-r1)^2/4 (variance-of-the-mean estimate under an independence assumption).
Use q=m^2/(m^2+v), with q=0 if both are zero, and replace m by q*m.
This is a fixed reliability heuristic, not a calibrated posterior; correlated
seeds share bias and disagreement can also reflect useful diversity.

Keep all original inner coefficients and the existing high-noise gate. There
is no new fitted alpha, threshold, Public-derived scaling, or image inference.
Compare on the same strict076 original5fold baseline. A matched uniform
shrinkage control uses the mean q on that fold's inner B high-noise eligible
rows only, without labels; apply that scalar to all high-noise V corrections.
This distinguishes sample-specific confidence from simply smaller corrections.

Advance only if confidence improves>=.005, improves>=3folds, and its paired-ID
95% lower bound is above0 versus BOTH the current correction and uniform
control. Otherwise keep the current candidate. Read cached predictions only,
NumPy CPU1, small bootstrap batches; do not load PyTorch/TabPFN/image arrays.
