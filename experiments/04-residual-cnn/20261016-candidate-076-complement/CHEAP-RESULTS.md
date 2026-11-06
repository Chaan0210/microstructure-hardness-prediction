# Completed inference probes and current research scope

The corrected076 baseline uses the original400 images/five folds and restricts
each augmented copy to its actual training source. Its RMSE is9.492439957.
This replaces historical076's9.508302515 as the local comparison baseline;
the numerical change is a validation reconstruction, not a newly improved
production model. H9 Tab OOF IDs/folds/labels/feature hashes were verified and
the076 linear alpha procedure was refitted inside each training fold.

| Same strict076 base and frozen original inner coefficients | RMSE | Gain versus current |
|---|---:|---:|
| Current two-seed CNN/four views |9.256739|0|
| Eight views |9.253945|+0.002793|
| Refresh context to076 |9.261895|−0.005156|
| Both |9.258938|−0.002199|
| Seed-disagreement shrinkage |9.264470|−0.007731|
| B-only mean shrinkage control |9.277105|−0.020366|

None passed its pre-registered advancement gate. The eight-view gain's paired
conditional95% interval is[-0.003841,0.009160]; no new production candidate
was created from these probes. The original four-view predictions were
replayed exactly. Confidence-based shrinkage has no fitted new parameter;
its constant control uses only inner B predictions and no labels.

The full076 residual-teacher pilot is **incomplete**. Three of twelve teacher
bundles and252 of864 individual Tab predictions are retained. The last teacher
process exited137 while memory pressure was observed; the exact cause was not
established. Prior voluntary interruptions affected only this task's own PIDs.
No other agent's process or files were changed. No new076-target CNN was trained
or scored. Resume instructions/code remain in `retarget_teachers.py` and
`retarget_cnn.py`; missing models must not be represented as completed results.

The installed library's built-model reuse setting was checked on four seed,
training-subset and feature-gate combinations: all predictions matched exactly.
It changes allocation/reconstruction cost, not the statistical model. Its
verification is `built-cache-verification.json`.

The subsequent small two-stream CNN study is separate: it retains the current
correction's honest073 teacher targets, compares raw+Gaussian versus raw+raw
with identical parameter count and initialization, and evaluates on strict076.
Its fixed protocol is `STREAM-PLAN.md`. Read completed result JSONs before
claiming an improvement; training curves alone are not validation results.

All comparisons reuse previously explored labels; paired-ID bootstrap intervals
are conditional summaries and do not account for the complete historical search
or dependence induced by shared training sets. The recommended076+frozenCNN CSV
from the preceding experiment remains unchanged and has not been uploaded here.
