# Reproduction commands and provenance

Run from `.`. All scripts derive outputs from their own directory, and refuse to overwrite completed artifacts. Use the existing root Python for image training and the existing TabPFN Python for readouts. No dependencies are installed or upgraded. Only one MPS stage runs at a time; do not overlap with unrelated GPU work.

For a clean repeat, copy the final source/protocol files into a new sibling directory under this project's experiments, without copying output caches. Never rerun into a completed record or historical directories. Code retains immutable source references. The original failed16/32 timing records and display repair are evidence, not files to replace during a repeat.

```sh
D=experiments/03-robustness/20261017-1835-boundary-connectivity
.venv/bin/python -B "$D/protect.py" freeze > "$D/protection-start.log" 2>&1
.venv/bin/python -B "$D/targets.py" preflight > "$D/target-preflight.log" 2>&1
.venv/bin/python -B "$D/targets.py" prepare > "$D/target-prepare.log" 2>&1
.venv/bin/python -B "$D/verify_graph.py" > "$D/graph-verification.log" 2>&1
.venv/bin/python -B "$D/encoder.py" probe > "$D/resource-probe-b32.log" 2>&1
.venv/bin/python -B "$D/budget.py"
.venv/bin/python -B "$D/encoder.py" train > "$D/training.log" 2>&1
.venv/bin/python -B "$D/verify_encoder.py" > "$D/encoder-verification.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/probe_readout.py" > "$D/readout-resource-probe.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/readout.py" > "$D/readout.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/pre_score_check.py" > "$D/pre-score-check.log" 2>&1
.venv/bin/python -B "$D/evaluate.py" > "$D/evaluation.log" 2>&1
.venv/bin/python -B "$D/quality_evaluate.py" > "$D/quality-evaluation.log" 2>&1
.venv/bin/python -B "$D/verify_quality.py" after > "$D/quality-verification.log" 2>&1
.venv/bin/python -B "$D/replay_training.py" > "$D/training-replay.log" 2>&1
.venv/bin/python -B "$D/replay_warmup.py" original > "$D/warmup-replay-original.log" 2>&1
.venv/bin/python -B "$D/replay_warmup.py" poison > "$D/warmup-replay-poison.log" 2>&1
experiments/frameworks/tabpfn-v2/.venv/bin/python -B "$D/verify_final.py" > "$D/final-verification.log" 2>&1
.venv/bin/python -B "$D/probe_gradient.py" > "$D/gradient-repeatability.log" 2>&1
.venv/bin/python -B "$D/replay_deterministic.py" original > "$D/deterministic-replay-original.log" 2>&1
.venv/bin/python -B "$D/replay_deterministic.py" poison > "$D/deterministic-replay-poison.log" 2>&1
.venv/bin/python -B "$D/measurement_shift.py" > "$D/measurement-shift.log" 2>&1
.venv/bin/python -B "$D/diagnostics.py" > "$D/diagnostics.log" 2>&1
.venv/bin/python -B "$D/quality_diagnostics.py" > "$D/quality-diagnostics.log" 2>&1
.venv/bin/python -B "$D/reproduce.py" > "$D/reproduction.log" 2>&1
.venv/bin/python -B "$D/protect.py" verify > "$D/protection-final.log" 2>&1
.venv/bin/python -B "$D/close_report.py"
.venv/bin/python -B "$D/manifest.py"
```

The original run first executed the now archived `encoder_probe_batch16.py`; its budget check failed.32 is the predeclared resource-only batch adjustment. The32 projection was6265.66s including a900s readout allowance. The probe initially had an over-conservative6000s gate, while PLAN's120min total allows6600s fits plus600s verification. `resource-accounting.json` records the correction before main fitting and all hardness scoring. Encoder/features projection5365.66s is below its unchanged5700s limit; projected total with verification6865.66s is below7200s. Both originally false probe records remain. This is not an epoch/control/seed reduction or a budget extension.

The first target preflight completed its numerical gate, then its optional matplotlib panel failed because that package was unavailable. The implementation now uses already-installed PIL. `targets_preflight_attempt1.py` and `target-preflight.log` preserve the original; `target-panel-repair.json` confirms target-function ASTs were unchanged. The original run completed only the missing panel with `targets.py panel` before preparing the400 source targets. A clean repeat uses the corrected single preflight command above.

Encoder seeds2026101793/2026111793 plus100*fold; view/order/dropout seed+41.40epochs, batch32, final5epoch parameter mean. Mean/std of pixels and hardness (only the ordinary supervised control) fit on the relevant A/T rows. Graph edges are4-neighbour32x32; ties use stable edge index. Dense boundary targets128x128. Pair pool seed2026101784,512pairs per offset group1/2/4/8; balanced64-pair training samples use each encoder's fixed training RNG. Fixed signature pairs use128 per group, same seed. No hardness-based pair graph or external image weights.

Each learned descriptor is320 pooled latent values plus28 direct boundary/affinity/connectivity summaries. StandardScaler/PCA16 fit only A/T latent rows; raw28 values and16PCs append to the exact incumbent100 physical columns, giving144 inputs. TabPFNv3.5 uses installed `experiments/frameworks/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors`, float32, memory_saving_mode, three seeds100*j+fold, j0/1/2. Raw-target measurement control gets28 values and16 zeros; it has one three-seed readout, duplicated in the prediction schema only. It is not an extra independent encoder replication.

The exact physical100 feature path/hash comes from `experiments/03-robustness/1300-robust-ensembles/abl/dev-h9phys-full-044-u1dmavg.npz`. Physical-only comparator contexts/predictions reuse the verified `experiments/03-robustness/20261017-1415-structure-context/readouts/` fits after checking source IDs, original hardness, features, seeds and checksums. Learned branches use identical source counts/targets. The high-noise beta rule and fixed primary/control gates are in PLAN.md. Bootstrap seed2026101795; B-coefficient bootstrap2026101794+fold.

The original source graph/anchors are `experiments/03-robustness/20261016-highnoise-relational/{inputs.npz,graphs.json,anchors/fold*.npz}`. Existing base and D4 masks are `experiments/02-microstructure/0220-grain-cellpose/masks-v3-train.npz` and `masks-tta-train.npz`, with hashes in target-preflight.json/target-audit.json. These auxiliary predictions are not grain annotations. All their derivatives keep source identity. Their historical mask choices are not newly nested.

Protected153547-file inventory includes082/076, all upstream dependencies and the entire completed preceding experiment. `starting-state.json` records Git HEAD, original staged/unstaged diffs, untracked status, package versions and incumbent CSV hashes. The separate Tab environment is recorded at readout execution. Historical full incumbent provenance remains protected in `experiments/03-robustness/20261016-2312-highnoise-ssl/incumbent-provenance/` and the preceding reproduction records. No Git writes occur.

After numerical/source checks, run `reproduce.py` for fresh076 cached-component assembly and082 saved-CNN inference; outputs go only into this experiment. This conservation-only process may read318 deployed test images, with no training/fitting. It does not claim reproduction of all historical training. `replay_training.py` is a separate complete40epoch new-model reproduction for one topology and one ordinary-hardness selector, with excluded images/masks/labels changed before selecting the permitted inputs; it reports actual accelerator repeatability. `verify_final.py` checks independent beta/assembly/RMSE math and performs paired actual Tab replays with outside labels changed. `diagnostics.py` reads completed OOF results only for descriptive failures, weak-target metrics and explicitly labeled held-out map panels; it never changes a candidate. Run `protect.py verify` and finally `manifest.py` after every writer and the report/completion record have closed.

The prospective source-support policy is separately frozen in QUALITY-PLAN.md and quality-code-lock.json while encoders were still training, before paired Tab readouts or outer hardness scoring. Its image-local criterion is strictly evidenceAUC>0.5; chance/undefined0.5 is not support. Run quality_evaluate.py after the original evaluate.py and use delivery-decision.json for the predeclared final protocol priority. The added policy has no new GPU fits or threshold grid. Its route/zero decision is internalB LOO, including leave-one-out recalibration of the old082 coefficient. Use bootstrap2026101797 for97.5% intervals, with the original protocol subjected to the same stricter family screen for final recommendation. No outer-winning control is selected.

`verify_quality.py preflight` was run before new readouts and checks scalar all/supported/zero cases and allfive existing incumbent calibrations. `constraint_audit.py` reproduces the target-consistency diagnostic; it uses no hardness and changes no targets. `MEASUREMENT-LIMITS.md` distinguishes affinity scores from calibrated probabilities and mask consensus from physical truth. The original all-high results remain intact regardless of the added policy.

Numerical verification found the original topology training was not bit repeatable, even with the original startup and unchanged images. Its frozen OOF remains the original realized weights/predictions. `probe_gradient.py` isolated duplicate-index gather backward:40 distinct gradients in40 identical real-batch repeats versus1 with equivalent per-edge coefficient aggregation. `deterministic_loss.py` is a separate verification prototype; `encoder.py` was NOT edited to use it. `replay_deterministic.py` explicitly overrides only the objective inside its own process and records the override/helper hash. Two completeA213/40epoch same-seed runs, with original and excluded-poisoned warmup images, matched weights/features exactly. This prototype did NOT receive full nested accuracy validation and must not inherit the reported OOF score. Do not use it to regenerate the original60-model matrix and expect identical results. A future experiment must freeze its own protocol and reevaluate that implementation.

The initial `replay_training.py` omitted topology's random-feature startup and reported nonexactness. `replay_warmup.py` matched that startup in separate processes; unchanged inputs still failed bit replay, so the discrepancy was not attributed to omitted startup alone. All failure records remain. `measurement_shift.py` compares interpretable signatures of the same source across320-source models trained with/without it, without fitting a new predictor; no clear mean entropy/component shift was established. It is a descriptive diagnostic, not a new candidate.

`constraint_audit.py` and `summarize_targets.py` reproduce target diagnostics without hardness labels. `summarize_targets.py` may verify an existing identical summary; other output writers use exclusive creation. The numerical repair commands are verification stages that were introduced only after an actual reproducibility failure, with no new accuracy/parameter search. All MPS stages above remain sequential. Core stage timers and independent log spans are in compute-accounting.json; nested comparisons are in comparison.csv and quality-policy/comparison.csv. Final recommendation is controlled by delivery-decision.json, which selects neither new protocol.

`close_report.py` is a one-off narrative for the current frozen evaluated artifacts and explicitly checks its core numerical result. It is not a generic reporter for a fresh stochastic training run. Repeating the original gathered-gradient training can change weights/results; such a run needs its own newly interpreted report. Saved-weight/context inference and baseline assembly are the deterministic reproduction targets checked here. The separate aggregated-gradient prototype requires a new full validation protocol before any performance claim.
