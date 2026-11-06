# Reproduction and provenance

Run from `.` with the
existing root `.venv`. No packages are installed or changed. All new training
is small CPU2 meta fitting; no image encoder is retrained in this experiment.
Every output is scoped to this directory and source artifacts are read-only.

For a clean repeat, copy the final Python/protocol files into a fresh sibling
experiment directory. Never rerun output writers over this completed record.
Paths derive from each script's location. `common.py` points to the frozen
historical base-model families; copying the scripts does not copy or modify them.

```sh
D=experiments/04-residual-cnn/20261018-0155-candidate-082-fusion
.venv/bin/python -B "$D/protect.py" freeze
.venv/bin/python -B "$D/prepare.py"
.venv/bin/python -B "$D/preflight.py"
.venv/bin/python -B "$D/fit_fusion.py"
.venv/bin/python -B "$D/calibration_control.py"
.venv/bin/python -B "$D/evaluate.py"
.venv/bin/python -B "$D/verify.py"
.venv/bin/python -B "$D/diagnostics.py"
.venv/bin/python -B "$D/reproduce.py"
.venv/bin/python -B "$D/protect.py" verify
.venv/bin/python -B "$D/report.py"
```

The first preflight ran the exact numerical fixture directly through Python
before `preflight.py` was saved. The script repeats and verifies that record.
No query accuracy was inspected at preflight. The main fit-code lock precedes
all fusion fitting. ATTRIBUTION.md adds a stricter quality-calibration control
before any fusion outer score; it changes neither primary selection nor weights.

`prepare.py` checks source scopes and writes `inputs/fold*.npz`. These containers
also include query labels for later scoring. `fit_fusion.cohorts` explicitly
loads only `yB`; query dictionaries have no `y`, and prediction rejects such a
key. `verify.py` actually uses the same loader against an alternative input file
with all V labels and query features poisoned. Entire nested selection stays
exactly identical. A separate check excludes one B label from the old082 beta,
quality normalization, family fits and every held-out prediction.

Sources are in input-provenance.json and the initial protected inventory:
- OldCNN: `rmse-research-20261015/round4-full073/aux-seed*/fold*`.
-076-targetCNN: the two completed pilot folds in `076-complement-20261016`
  and three completed folds in `model-audit-repair-20261017-2335`.
- Dense pick/pool: both seeds in `model-audit-repair-20261017-2335`.
- Topology difference: original evaluated paired physical/augmented readouts in
  `boundary-connectivity-20261017-1835`; no later loss prototype is substituted.
- A→B physical base: complete076 and073 teachers plus the G021 A-only anchor.
- T→V reference: verified source-restricted076 and082 caches.

Every family averages its two seeds before fusion. Each source is predicted onB
by models trained only onA, and onV by models trained only onT. No other-fold
OOF prediction is used as meta-training data. The inherited representations and
past model choices retain their historical limitations; these are not new labels.

The selected procedure is exactly PLAN.md: null/fixed/shared/global/conditional,
lambda1/10/100, every nuisance fitted again within B-LOO, deterministic ties,
then a fullB refit. SLSQP uses analytic gradients and fixed bounds. The physical
quality variables are estimated noise and segmentation object count, not verified
ground-truth grain measurements. Bootstrap seed2026101809. Final parameters and
all selection losses are in fold*-selection.json; source-specific residual
coefficients are in coefficients.csv. These are not whole-predictor percentages.

Verification includes an independent active-set quadratic-program solver for
15global contexts, duplicated-predictor invariance,30saved fusion replays,
outside-label/feature poisoning, query coefficient bounds, scalarRMSE and
top-contributor removal, and the exact fixed-mixture MSE decomposition.
The first verifier's reporting loop reused the variableB when naming a tested
source; the test itself used fold0 correctly. It now captures the ID explicitly.
In these folds the two reported IDs happened to be identical. Original code and
record are preserved; verification-metadata-fix.json documents this metadata-only
repair. No model, assertion, prediction or score changed.

Incumbent conservation replay reads the deployed test inputs only to reconstruct
the existing076/082 outputs. It performs no fitting or new-candidate selection.
It is saved-weight/cache inference, not a reproduction of all original training.
082SHA256:0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f.
076SHA256:b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0.
GitHEAD, original dirty state, package versions, every protected upstream file
and previous execution records are retained in starting-state/protected.json.

The primary failed its fixed screen, so equal-size expansion and full500/test
candidate construction were not run. No submission or official-best change.
