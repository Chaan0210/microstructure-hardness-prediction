# Frozen CNN residual heads — secondary, before outer CNN scores

No new round4 outer CNN score has been viewed. Primary CNN/auxiliary/statistics
arms and their weights remain unchanged. This inexpensive secondary study uses
the already generated full073 teachers and existing native256 CNN features.

Hypothesis: learning a weak residual signal from scratch with213/320 images
may be harder than learning a small head on a frozen visual representation.
The train residual objectives in the completed folds changed only modestly
during40 epochs. A fixed visual encoder reduces supervised representation
variance; residualization may expose information complementary to T.

Prior evidence AGAINST this family is explicit: study205's frozen
ImageNet/MicroNet ResNet50 mean2048 + EDF32 Ridge had standalone RMSE15.4690 /
15.8745. Adding10% to021 worsened its RMSE; MicroNet improved0/5 folds.
Do not rerun that fixed021 experiment or claim domain pretraining is superior.
The new task is conditional correction of full073's much stronger non021 T.

Reuse both feature caches read-only. Keep the original preprocessing and
target-independent EDF32 regularization exactly. Fit two matched objectives:
direct-y (control) and cross-fitted residual y−T (hypothesis). The direct
direction is H_direct−T; the residual direction is H_residual. For both,
fit addition and replacement coefficients in B107 only, then refit the head
on T320 and evaluate on the same V80. Use the same fallback on failed rows.
There is no new encoder training, download, feature inference, alpha search,
target-dependent capacity selection or outer-label weight optimization.

Use the existing205 implementation's pure NumPy definitions after inspecting
them, not its executable training entry point. Reproduce all old direct-y
outer predictions as a control. Check independent dual Ridge solutions,
held-out label poisoning, exact feature/ID/fold hashes and coefficient reload.
Save every result, including failures. Report this as another exploratory
comparison on the already reused400 images; a small single-partition gain
does not establish generalization or automatically replace061/073.
