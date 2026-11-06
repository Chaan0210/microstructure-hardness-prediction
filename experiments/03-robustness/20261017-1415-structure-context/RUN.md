# Commands and provenance

Run from `.`. Use existing environments only. GPU commands require access to the local MPS device; no installation, network access or external data is needed. Each stage is sequential on the GPU. All outputs derive from the script's own directory and use exclusive creation.

```sh
D=experiments/03-robustness/20261017-1415-structure-context
.venv/bin/python -B "$D/protect.py" freeze > "$D/protection-start.log" 2>&1
.venv/bin/python -B "$D/encoder.py" probe > "$D/probe.log" 2>&1
.venv/bin/python -B "$D/encoder.py" train > "$D/training.log" 2>&1
.venv/bin/python -B "$D/verify_math.py" > "$D/mathematical-checks.log" 2>&1
.venv/bin/python -B "$D/loss_audit.py" > "$D/loss-audit.log" 2>&1
.venv/bin/python -B "$D/verify_encoder.py" > "$D/encoder-verification.log" 2>&1
.venv/bin/python -B "$D/verify_random.py" > "$D/random-control-verification.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/readout.py" > "$D/readout.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/pre_score_check.py" > "$D/pre-score-check.log" 2>&1
.venv/bin/python -B "$D/evaluate.py" > "$D/evaluation.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/verify_readout.py" > "$D/readout-verification.log" 2>&1
.venv/bin/python -B "$D/replay_encoder.py" > "$D/encoder-repeat.log" 2>&1
.venv/bin/python -B "$D/reproduce.py" > "$D/reproduction.log" 2>&1
.venv/bin/python -B "$D/protect.py" verify > "$D/protection-final.log" 2>&1
.venv/bin/python -B "$D/manifest.py"
```

Main settings and stopping criteria are frozen in PLAN.md. AMENDMENT-01.md records the timing-only90->120min compute amendment before main training/hardness fitting. The original probe reports `passed:false` for the90min estimate; numerical, memory and gradient checks passed. The main trainer checks the amended120min bound and reserves readout time. No epoch/seed reduction is hidden.

The readout uses the exact protected100-column feature file identified by `experiments/03-robustness/1300-robust-ensembles/abl/dev-h9phys-full-044-u1dmavg.npz`, rather than reconstructing its physical/H9 interactions. Both members of every augmented-minus-base direction fit the same source IDs and original targets. Every new encoder, normalization, PCA, readout and beta fits only its prescribed A/T/B subset. No production082 weight is applied to OOF. The new branch is nested conditional on historical frozen auxiliary representations.

For a clean rerun, copy code and protocol files into a fresh sibling directory under experiments, without copying output caches. Never run into this completed record or historical baseline directories. The scripts retain the project-root and immutable input references. The originals remain protected by the inherited inventory; starting-state.json captures current Git status/diffs/head and root package versions, and the readout lock records its separate environment.

Encoder seeds2026101771/2026111771 plus100*outer fold; paired view/order/dropout seed=encoder seed+41. Source permutation=RandomState271828+source index. Query-geometry masks=81173+source index. Tab seeds100*j+fold for j=0,1,2. Evaluation bootstrap2026101781 and coefficient bootstrap2026101782+fold. No new seed is selected from outer scores.

The source graphs and082/076 anchors are `experiments/03-robustness/20261016-highnoise-relational/{inputs.npz,graphs.json,anchors/fold*.npz}`. All source files, features, labels, dependencies, incumbent weights/configurations/predictions and previous exact replay evidence remain covered by protected.json. Full original incumbent training/assembly provenance is retained in the protected `experiments/03-robustness/20261016-2312-highnoise-ssl/incumbent-provenance/`.

No new test data is read by research fitting. If the fixed gate passes, final500 fitting and candidate inference are a separate, explicitly logged stage.082/076 are never replaced or submitted automatically. Run manifest.py last after every writer has exited.

`illustrate.py` is an optional CPU-only, label-blind diagnostic run on the first three noisy fold0 B images. It does not select sources by error or change training. `reproduce.py` is a separate conservation-only inference script:076 cached-component assembly and082 saved-CNN inference, with no fitting or candidate selection. It may read deployed test images solely for checking incumbent identity; research scripts block those paths. This verifies saved-artifact inference rather than the entire historical training run.

The shuffled control's physical-mask-footprint limitation is recorded in CONTROL-NOTE.md. The masked-pixel isolation check demonstrates that hidden values cannot enter model outputs, and the loss audit uses an independent NumPy formula. MPS repeatability is measured by replay_encoder.py; exactness is reported only if observed.

One additional, prospectively locked projection policy follows the closed PCA failure. `alignment-followup/PLAN.md` freezes PLS settings, nested PCA/PLS/zero selection, the unchanged total compute budget, and the stricter97.5% interval rule. `alignment.py` does not change parent predictions or decisions. Run these stages after parent verification and before incumbent reproduction/final preservation:

```sh
D=experiments/03-robustness/20261017-1415-structure-context
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/alignment.py" fit > "$D/alignment-followup/fit.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/verify_alignment.py" before > "$D/alignment-followup/pre-score-check.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/alignment.py" evaluate > "$D/alignment-followup/evaluation.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/verify_alignment.py" after > "$D/alignment-followup/selection-check.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/replay_alignment.py" > "$D/alignment-followup/numerical-replay.log" 2>&1
.venv/bin/python -B "$D/alignment_diagnostics.py" > "$D/alignment-followup/diagnostics.log" 2>&1
```

The PLS fit and numerical replay are sequential GPU stages in that block. Teacher residuals are leave-one-source-out physical Ridge(alpha100), including imputation/scaling. All180 new Tab contexts use the same original hardness targets and source counts as their30 reused physical references. The projection has16 components and both encoder seeds; no PLS dimension/seed grid is searched. Bootstrap seed2026101791 supplies the additional97.5% intervals. After successful preservation/reproduction, `close_report.py` builds the final report and completion record; run `manifest.py` after all writers have exited.

Optional completed diagnostics use `.venv/bin/python -B "$D/diagnostics.py"` and `.venv/bin/python -B "$D/illustrate.py"`. They do not fit or choose a candidate. `verify_readout_attempt1.py` preserves the original verifier: its assembly equality assertion failed only because mean(augmented)-base and mean(augmented-base) differed by2.84e-14 HV. `verification-repair.json` records the repair to require exact same-order assembly and tolerate1e-10 for independently ordered arithmetic. No trained model or prediction was changed.
