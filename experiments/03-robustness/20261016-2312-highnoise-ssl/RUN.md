# Reproduction and provenance

Work from `.` using the existing `.venv`; nothing was installed or upgraded. Every script derives its output path from its own directory. Outputs use exclusive creation. Do not rerun into this completed directory or invoke historical builders in their original locations.

## Exact research sequence

```sh
D=experiments/03-robustness/20261016-2312-highnoise-ssl
.venv/bin/python -B "$D/protect.py" freeze > "$D/protection-start.log" 2>&1
.venv/bin/python -B "$D/study.py" selfcheck
.venv/bin/python -B "$D/study.py" safety > "$D/safety-v2.log" 2>&1
.venv/bin/python -B "$D/study.py" train > "$D/training.log" 2>&1
# Only if the original45-minute check stops before the fixed matrix completes:
.venv/bin/python -B "$D/resume.py" > "$D/resume.log" 2>&1
.venv/bin/python -B "$D/strength_audit.py" > "$D/strength-audit.log" 2>&1
.venv/bin/python -B "$D/evaluate.py"
.venv/bin/python -B "$D/diagnose.py"
.venv/bin/python -B "$D/verify.py"
.venv/bin/python -B "$D/verify.py" replay
.venv/bin/python -B "$D/reproduce.py"
.venv/bin/python -B "$D/protect.py" verify
.venv/bin/python -B "$D/manifest.py"
```

The first selfcheck and first safety invocation stopped before computation because the write guard blocked PyTorch's default temporary-file probe outside this directory. Nothing outside was written. Runtime TMPDIR, Torch and matplotlib caches were explicitly redirected here before successful checks and before any measured run. The failed `safety.log` remains as evidence. No model-selection rule changed.

## Clean rerun

Create a fresh sibling directory under experiments; copy only `study.py`, `evaluate.py`, `verify.py`, `diagnose.py`, `view_screen.py`, `strength_audit.py`, `protect.py`, `reproduce.py`, `resume.py`, `manifest.py`, `PLAN.md`, `BUDGET-AMENDMENT.md`, `DIAGNOSTIC-PROTOCOL.md`, `analysis-code-lock.json`, `budget-lock.json`, `diagnostic-lock.json` and this RUN.md. The three copied locks retain the original declaration times and assert that code and protocols are unchanged; the rerun writes a fresh code-lock and starting-state. Run the sequence with D set to that new directory. The immutable earlier study remains the input source for `inputs.npz`, `graphs.json` and five `anchors/fold*.npz`; their hashes are in protected.json. No outputs are copied into baseline paths. `BUDGET-AMENDMENT.md` discloses the one-time timing-only extension45->55 minutes, recorded before outer scores. This measured run finished without resumption, so resume.py was not executed.

The optional label-blind source-view illustration was generated during training with `.venv/bin/python -B "$D/view_screen.py"`. It performs fixed image-local operations on two A213 sources, writes only the figure/measurement record here, and does not change any training input or choice. The display quantizes float views for viewing; model training uses float values.

## Parameters

- Outer folds and A213/B107/T320/V80 partitions: `experiments/03-robustness/20261016-highnoise-relational/graphs.json`; exact IDs are retained in every training record.
- Encoder initialization: 2026101671 +100*fold. First phase RNG: initialization seed+10000. Fine-tuning RNG: initialization seed+20000. Fresh selector/refit fits share seed, not data.
- Safety-render RNG: 2026101672+global source index. Source-grouped3fold: 2026101671+fold. Safety bootstrap:2000 source draws, same seed. Primary uncertainty:10000 draws seed2026101673.
- Native256 images; no masks or pretrained encoder from the baseline are used by the new learner. Two transformed views/source, batch at most32 sources, balanced final batch. Final4 epoch states averaged per24-epoch phase.
- Frozen-feature Ridge alpha100; feature scaling fitted on its own encoder training sources. Blend fitted only on B high-noise eligible rows, bounded[0,.25]. V never selects strength, encoder, checkpoint, downstream parameter, or blend.

## Incumbents

082 CSV SHA256: `0fa34cec1626b0887d3d7971197207c135a447a8f4805096b258e77b912f0e9f`.
076 CSV SHA256: `b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0`.

The complete baseline inventory and upstream environment/code/artifact hashes are inherited and checked in `protected.json` (151006 files at start), with Git head/diffs/status and installed-package versions in `starting-state.json`. The original seeds, training commands and component formula are retained in the protected `plateau-break-20261016-2035/{incumbent.json,environments.json,RUN.md,model-replay.json}` and `rmse-research-20261015/round4-full073/final500/recipe.json`. This study's082 baseline is the prior source-restricted OOF, never a production-coefficient retrofit to OOF.

`reproduce.py` reassembles076 from frozen component predictions and reruns082's two saved CNNs on the318 deployed high-noise test images for conservation verification only. There is no fitting or test-data use for new research. Outputs go into this experiment's `reproduction/`. This is saved-artifact prediction reproduction, not a claim that the historical full training run was repeated. Prior fuller076 head/precision replays remain protected.

`manifest.py` runs last after writers exit; do not redirect its stdout into the directory being hashed.
