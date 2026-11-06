# Commands and provenance

Working directory: `.`.
No package changes. Root `.venv` supplies image models, numerical analysis and
sklearn. Existing `experiments/frameworks/tabpfn-v2/.venv` supplies TabPFNv3.5 with local
`experiments/frameworks/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors`.
HF/Transformers offline flags and telemetry disabling are set in common.py.
Temporary/matplotlib outputs stay in this experiment. Python runs with `-B`.

These commands are the execution recipe, not an instruction to overwrite this
completed record. For a clean repeat copy final source/protocol files into a
new sibling directory under experiments. Output locations derive from each
script's own directory. Existing historical sources/caches remain read-only.
Freeze a fresh protected inventory before running. Save each command's logs in
that new directory; scripts use exclusive creation for artifacts.

```sh
D=experiments/04-residual-cnn/20261017-2335-model-audit-repair
.venv/bin/python -B "$D/protect.py" freeze
experiments/frameworks/tabpfn-v2/.venv/bin/python -B -u "$D/finish_teachers.py"
.venv/bin/python -B -u "$D/cnn_repair.py" retarget
.venv/bin/python -B "$D/evaluate.py" retarget
.venv/bin/python -B "$D/audit.py"
.venv/bin/python -B -u "$D/cnn_repair.py" probe
.venv/bin/python -B -u "$D/log_mean.py"
.venv/bin/python -B "$D/evaluate.py" log
.venv/bin/python -B -u "$D/dense_mps.py" probe
# Inspect timings and apply PLAN's unchanged90min total/10min verification reserve.
# The recorded main-run authorization is dense-budget-decision.json.
.venv/bin/python -B "$D/budget.py"
.venv/bin/python -B -u "$D/dense_mps.py" train
.venv/bin/python -B "$D/evaluate.py" dense
.venv/bin/python -B "$D/dense_decision.py"
.venv/bin/python -B "$D/log_size_diagnostic.py"
.venv/bin/python -B "$D/calibration_diagnostic.py"
.venv/bin/python -B "$D/trace_selection.py"
.venv/bin/python -B "$D/cases.py"
.venv/bin/python -B "$D/legacy_replay.py"
.venv/bin/python -B "$D/verify.py" sources
.venv/bin/python -B "$D/verify.py" repeat
.venv/bin/python -B "$D/verify.py" models
.venv/bin/python -B "$D/verify.py" assembly
.venv/bin/python -B -u "$D/dense_verify.py"
.venv/bin/python -B "$D/reproduce.py"
.venv/bin/python -B "$D/protect.py" verify
```

MPS needs host execution in this desktop's sandbox. It uses local hardware
only, one fitting process at a time. CPU uses2threads. The model memory check
caps own RSS plus MPS driver allocation at5GiB. Stage logs preserve failed
resource/attribute probes; they are not accuracy experiments.

Teacher seeds100*j+fold,j=0,1,2; all8Tab estimators after automatic resolution.
Missing180contexts use explicitfloat32. Existing27teacher bundles and36partial
members are reused after checking source/queryIDs and exact training-label hash.
One matched142-source originalCPU context is reproduced onMPS before fitting;
tolerance.001HV is predeclared, observed0.0000742549974575HV. Original freeze
contains all caches, feature banks, source files and dependencies. The first
attempt checked `n_estimators` before fit (the library exposes`auto`); it failed
before fitting. The repaired assertion checks resolved`n_estimators_` after fit.
Both source versions and logs remain; this is not a changed ensemble size.

Retarget CNN: exact historicalCPU fitter,2seeds2026101501+100fold+100000*repeat,
40epochs,batch32,AdamW.002/weight_decay.01,cosine to.0001,D4,final5state mean,
four-view inference.8completed pilot models reused with metadata and byte-exact
saved-weight inference;12remaining models newly trained. All20 use source-honest
T076 residual targets. They still use a nonG021 teacher normalized by.84; do not
describe them as residuals of the complete076 mixture or a teacher-size repair.

Dense models: same seeds/epochs/loss/parameter tensors, original073 teacher
targets/context to isolate architecture. First conv stride1,then either sample
every fourth location or4x4average pooling. Later blocks unchanged. Both arms
use MPS after the label-freeCPU timing gate failed. CPU andMPS timing probe
weights are discarded from validation. No crop or intensity view is trained.
Dense controls share GPU random streams/initialization; they are not an exact
CPU training trajectory comparison. ADDITIONAL-CONTROL.md is a prospective
stricter test against unchangedCNN with matching coefficient selection.

Log correction:20outer plus100inner four-variant source-grouped fits, with the
old alpha grid nested again inside each inner training partition. Every
derivative remains with its original source. Smearing and offset use only
original noisy training-source OOF residuals. Stored model.joblib/context.npz
files allow exact inference and full Ridge refitting without reading labels
outside the recorded training capability. A matched256-source diagnostic is
descriptive after primary failure and cannot promote another policy.

Baseline provenance is preserved in starting-state.json/protected.json and
the historical `plateau-break-20261016-2035/{incumbent.json,environments.json,
RUN.md,model-replay.json}`, `round4-full073/final500/recipe.json`, and
`highnoise-relational-20261016/{inputs.npz,graphs.json,anchors}`.
082SHA256=`0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f`;
076SHA256=`b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.
GitHEAD, original uncommitted diff and package versions were recorded before
experimentation. No historical file is updated to reflect a new result.

reproduce.py performs conservation-only076 frozen-component assembly and082
saved-weight inference on the318existing deployed test images, after research
decisions are fixed. No fitting/selection uses those test inputs. This verifies
the frozen prediction pipeline, not a fresh run of all historical training.
