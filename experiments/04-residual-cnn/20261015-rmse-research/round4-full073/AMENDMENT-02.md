# Inner-only adaptive021 weighting, fixed before CNN outcomes

No round4 CNN has yet been fitted. Full teacher construction is still in
fold0. The original primary additions/replacements remain unchanged.

The complete internal021 anchors permit a useful additional paired control
at essentially zero model-training cost:

* Weight-only control: (1−a)T + aG, with a in[0,1] fit only on B107.
* Joint blend: (1−a−b)T + aG + b(T+r), with a,b>=0 and a+b<=1,
  also fit only on B107. Equivalently T+a(G−T)+br.

G is complete021. T contains all original non021 components. r is the
already trained CNN's residual output. Failed rows remain the common G
fallback. Refit nothing and tune no outer coefficient. Optimize the convex
two-variable least-squares problem over the simplex, record per-fold weights,
and compare against the adaptive weight-only control. In particular, a gain
over fixed073 does not establish a CNN benefit if this control is better.

Report these as secondary explorations, including all models and seeds;
do not choose a seed or a coefficient based on outer performance. This
addresses whether021's fixed16% weight itself is a bottleneck and whether
CNN information remains useful after an honest correction of that weight.
