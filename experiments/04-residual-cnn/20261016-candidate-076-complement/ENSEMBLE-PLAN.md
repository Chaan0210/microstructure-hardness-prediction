# Fixed-family averaging, registered before second-seed evaluation

The unexpected duplicate-input control could supply optimization diversity,
even if replacing the existing CNN is not robust. In addition to the direct
two-seed confirmation, evaluate the fixed50:50 mean of the CURRENT candidate
and the independently B-calibrated two-seed duplicate candidate. No weight
search or re-calibration of this mean. Its frozen-coefficient counterpart is
also a fixed50:50 mean. Both inputs have identical original outer folds.

Report residual correlation and the exact MSE averaging identity. Include the
direct duplicate and fixed-family mean in simultaneous conditional bootstrap
bands. The mean needs gain>=.01,>=3betterfolds,a95%lower bound above0 and positive
frozen-weight gain before promotion. Report this as a post-first-seed hypothesis
with stochastic seed confirmation, not independent data confirmation.
