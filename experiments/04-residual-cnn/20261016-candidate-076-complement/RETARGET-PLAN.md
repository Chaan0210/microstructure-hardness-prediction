# Train on actual076 residuals: bounded two-fold pilot

Registered after the cheap probes: full8-view inference gained only.00279
(interval crosses0); context refresh alone lost.00516. Neither is adopted.
The076 high-domain teacher shift has RMS.82757 and correction correlation
−.13575, so changing context alone does not answer whether consistent target
and context alignment improves learning. No new loss/backbone/seed search.

Use fixed original outer folds0 and1 as a pilot, not a fresh holdout. For each,
replace all six nested073 non021 teachers by076 teachers: all12 H9 Tab banks,
three seeds, both feature gates; high-noise linear partners use source-grouped
five-fold alpha selection inside that teacher's training rows. Augmented
sources must be in that exact training set. Reuse unchanged low linear,
GMM and grain-set predictions from matching original teachers; verify IDs
and hashes. Recompute the complete076 non021 weighted sum and failure rule.
No reduced Tab ensemble or source-external augmented labels are allowed.

Train the original auxiliary CNN on y−T076, with076 context, same40 epochs,
normalization rule, AdamW,D4,four-view inference and two existing seeds
2026101501+100*fold and2026201501+100*fold. A213/B107 selection uses deep
teachers trained on142 rows; outer T320/V80 uses three213/214-row teachers.

Compare against the same old two-seed CNN added to the same strict076 base.
For the matched end-to-end comparison, both old and retargeted corrections
have weights selected by the original noise-regime LOO rule using B only
and the same full076 B prediction; zero low correction is fixed. Also report
both corrections with the original frozen inner coefficients to isolate
the effect of learning from weight refitting. Original fixed candidate is
retained as a separate reference. Failed rows stay unchanged.

Run both pilot folds before comparing scores. Advance to remaining folds
only if retargeting gains>=.01 pooled RMSE over the matched old CNN, neither
pilot fold worsens>.05, and the frozen-weight comparison also improves.
If this gate fails, do not run remaining18 teacher bundles or full500 CNNs.
Pilot-selected results are explicitly exploratory. If it passes, complete
all five folds before considering production; require gain on>=3folds and
conditional95% gain interval above0 against the matched old CNN.

One CPU2 process; no new environment or test inputs. Twelve new teacher
bundles in the pilot, reusing unchanged components;72 Tab fits per bundle.
Use caches safely and a70min wall-clock cap for this pilot teacher stage.
Keep all completed artifacts if the budget is reached; never claim missing
folds completed. Separate evaluation from training; no outer-label early stop.
