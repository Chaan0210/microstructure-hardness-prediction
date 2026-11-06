# Fixed two-seed mean, before seed1 outcomes

The first primary seed's complete original5-fold result is now known:
addition9.4638873 versus0739.5472452; replacement9.4006377 versus deletion
9.4934386. Both relevant conditional intervals include zero. The statistics
control improves less. No second-seed result has been observed.

The planned second-seed confirmation will also support one fixed ensemble:
average the two raw residual predictions with equal weights, then select a
single correction coefficient on B107 for that averaged predictor. Apply it
to the equal mean of the two final T320-trained CNNs on V80. Do not select
a seed or fit a mixing ratio using outer labels. Evaluate both addition and
replacement, and retain every individual-seed result in the report.

This is a secondary, explicitly post-first-seed hypothesis about optimization
variance. It adds no model training. Use the same rule for the auxiliary arm
only if its pre-registered second-seed run exists. Ensemble coefficients
remain strictly internal to each outer fold. Report conditional uncertainty,
current-round simultaneous bands and the lack of independent new data.
