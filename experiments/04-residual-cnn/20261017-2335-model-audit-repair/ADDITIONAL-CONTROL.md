# Stricter attribution check before dense-model outer evaluation

Recorded18October2026KST while dense selector models are training; no dense
outer score has been computed or inspected. The original PLAN and thresholds
remain unchanged. No model, target, hyperparameter, seed or epoch changes.

H1 now demonstrates that recomputing the old CNN coefficient on076 can give a
large-looking dev gain without better learned corrections: old matched9.16971,
retarget matched9.18485,0829.25674. Both gains fail influence robustness.
Therefore H3 must additionally beat `old_matched`, with a positive98.33%
paired interval, before attributing a material improvement to the new sampling
architecture or recommending it. The original H3 decision is still reported.
This is a prospective stricter condition, never a rescue of a failing score.

Same-compute dense-pick remains the primary mechanism control. Averaging4x4
activations also changes effective receptive field and sampling phase; this
experiment cannot separate those from low-pass aggregation. Small frozen-model
1px-shift responses do not demonstrate a major aliasing failure. Learned
downsampling is tested because a frozen model probe cannot establish whether
omitted dense activations contain usable information.

The CPU probe projected6336s; the MPS probe projected3116s. The implementation
had a conservative3000s probe threshold, stricter than PLAN's90min total budget.
The original false probes and code are preserved. A budget-only check before
main fitting established3905s remaining after600s verification reserve and a
3300s hard dense-fit cap. Thus3116s fits the unchanged budget. See
dense-budget-decision.json; no accuracy result drove this resource correction.
