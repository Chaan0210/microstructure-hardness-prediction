# Last low-cost calibration stability check

The duplicate-input effect replicated direction across two seeds but did not
pass promotion: mean2 gain.019633,CI[-.016359,.056729]. Fixed family averaging
also failed its interval gate. No full500 variant is authorized by those gates.

New observation: separately calibrated legacy seeds score9.2442/9.2111, but
averaging their raw corrections then recalibrating scores9.2567. The operations
do not commute because the calibration includes discrete pooling selection.
Test two structural changes, with no new hyperparameter grid:

1. Calibrate each seed on B independently, then average its final corrections
   with fixed equal weights, versus calibrating the averaged raw correction.
2. Select the existing pooling strength by LOO loss only on the high-noise
   rows where the deployed correction is applied. Low-row correction is fixed0;
   low labels can still inform the shared prior inside each calibration fit.
   Every left-out high label is excluded from that prior. Retain exactly the
   original strengths(global,100,30,10,0) and bounds[0,1].

Evaluate the2x2 calibration methods on both the existing and duplicate two-seed
CNNs. All8cases use the same strict076 outer predictions and original B-only
calibration labels. Original mean/LOO is the reference; no Public fitting.
This is explicitly motivated by already-read scores, not a fresh preregistered
generalization test. Report simultaneous conditional bands for new contrasts.

Any apparent gain must survive the existing frozen-A role test: anchors0,1,4,
five107/80 assignments,seed2026101608; original073 A-only teachers/base and
A-only CNNs,all187query images excluded from each A fit. Infer only the small
CNNs; reuse existing teacher extensions. This checks calibration stability,
not full076 replication. Aggregate losses by originalID,not1200 independent
rows. Require>=.01gain,>=3outerfolds and positive conditional95% lower bound
on original5fold,plus positive increment in every frozen anchor,positive
conditional pooled lower bound,and gain remaining after removing ten largest
beneficiaries. Otherwise keep the current candidate and stop this search branch.

CPU1,small batches,no large TabPFN fitting,no test inputs or automatic upload.
