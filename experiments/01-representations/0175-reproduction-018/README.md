# Study175: fresh development reproduction of018

This directory contains a separate training implementation. It uses the project's existing `.venv`, CPU threads2, MPS, and local official DINOv2 weights. There are no historical study-module imports in the trainer and no package installation or network access.

The target is the frozen018 **original five-fold development OOF procedure**, not a new candidate and not a500-row production model. Every fold trains its entire13-node hierarchy from initialization. The retained recipe is selection on all current KRR training rows; study174's80% intervention is not used.

## Inputs and what is reused

- Exactly400 development IDs, targets, groups and original fold assignments are frozen in `run/feature-plan.json` and `run/plan.json`.
- Preparation reads the development-only run012 cache for IDs/targets/groups and only the `ids`, `y`, and `fold` members of the development018 archive for split metadata. Historical OOF predictions are not training inputs.
- `extract.py` reads only the named400 raw training images. It does not open the500-row `train.csv`, formerholdout images, test images, or sample submission.
- Official pretrained DINO-S/B weights are reused, frozen, from the local155 package; their hashes are recorded. No encoder is fine-tuned.
- Compact26, physics57, CNN128 inputs, S3456, B6912 and fixed spatial528 descriptors are freshly extracted. The new cache is used for every new learner.
- Old feature caches, model states and OOF predictions are opened only by feature verification or the separate comparison process. They are never initialization, fitting targets, calibration data, or selection scores for new learners.

The training process installs a file-access audit hook that blocks other project directories, including historical artifacts, reports and raw data. Only this standalone directory and installed `.venv` files are allowed within the project. Network connections are disabled. `run/execution-*.json` records observed accesses and resource samples.

## Actual commands for the current attempt

Run from the project root with its existing environment:

```sh
.venv/bin/python experiments/01-representations/0175-reproduction-018/extract.py
.venv/bin/python experiments/01-representations/0175-reproduction-018/train.py prepare
.venv/bin/python experiments/01-representations/0175-reproduction-018/train.py 0
.venv/bin/python experiments/01-representations/0175-reproduction-018/validate.py 0
.venv/bin/python experiments/01-representations/0175-reproduction-018/compare.py 0
.venv/bin/python experiments/01-representations/0175-reproduction-018/train.py 1 2 3 4
.venv/bin/python experiments/01-representations/0175-reproduction-018/validate.py 1 2 3 4
.venv/bin/python experiments/01-representations/0175-reproduction-018/compare.py 1 2 3 4
.venv/bin/python experiments/01-representations/0175-reproduction-018/replay.py
```

The completed outputs are intentionally exclusive: do not rerun extraction or preparation into an existing `run`. Training may reuse **this attempt's completed nodes** after verifying their report hashes and immutable input/code plan. It never substitutes historical models. An incomplete node is retained and causes an explicit stop; recovery must record and move that incomplete node to a separate location before restarting it. A PID lock prevents two train commands from using the attempt simultaneously. A stale lock after abrupt process termination requires checking that its PID is no longer active; it is never silently removed.

For a second independent attempt, copy only the code/vendor files into a new sibling directory under `experiments`, with an empty `run`. The current code resolves the project two directory levels above itself. This is a project-local standalone trainer, not yet a portable raw500 training distribution. Preserve this attempt and its plan for comparison.

## Exact training scope

Each outer320/80 fold fits13 base nodes: the outer320 learner,3 middle213/214 learners, and9 deep142/143 learners. Each node has3 CNNs,3 S heads,3 B heads and its own9-setting,3-fold KRR selection followed by a full-node refit. Four shortB heads are trained per outer fold. All scalers and target statistics use the node's own training rows.

Three middle OOF blocks fit the outer affine calibration. For each honest residual blockB, three deeper OOF blocks withinA fit its anchor calibration, which excludes allB labels. Five residual candidates are independently fitted insideB's three-fold selection and again on fullB. Fixed512 group bootstrap draws choose convex winner-frequency weights. Outer predictions add the mean of the three residual corrections. No global400 OOF residuals are used for training.

Main five-fold counts:195 CNNs,390 CPU long heads,20 shortB heads,65 final KRRs plus1,755 inner candidate fits,20 affine fits, and300 residual model objects (240 nonzero fits). The separate representative replay adds3 CNNs,6 long heads,1 shortB and28 KRR fits; these counts are reported separately.

## Verification and tolerances

`validate.py` reconstructs every KRR grid score and selected fit using direct numerical algebra, independently reconstructs residual ridge/RBF fits and bootstrap weights, checks every nesting/group/ID boundary, and reloads all neural models on the applicable devices. It does not use historical018 models.

`compare.py` is a separate, comparison-only reader. After a fresh fold report is committed, it compares each seed's prediction and tensor state, then base predictions, anchors, residual selection, corrections and final OOF against preserved018 references. Its source hashes identify every reference opened.

Before fitting, `run/plan.json` fixes classical/CPU-head tolerance1e-8, same-device reload1e-8, CPU/MPS reload1e-3 and fresh MPS/pipeline difference ceiling1e-3. Prior local measurements were around1e-5–1e-4, and prior same-environment CNN training replay was exact. The1e-3 ceiling is diagnostic, not permission to explain away drift. Tensor equality is reported separately. A difference is not called MPS nondeterminism without same-input/seed replay evidence. Tolerances are not widened based on OOF results.

`replay.py` changes the outer80 labels and freshly retrains the representative outer node, then checks predictions and all10 neural states. This is **not** a full13-node poison rerun. Other nodes are covered by explicit train/query scopes, nesting reconstruction, reload and comparison checks.

## Resources and later scope

The original15–25minute estimate covered one400-row root model. A full400-row OOF reproduction requires five separate320-row hierarchies. Fold0 took327.1seconds, projecting approximately27.3minutes for training alone. Fresh400 feature extraction took213.4seconds. Implementation, audits and the representative replay add time. Process RSS is capped at4GB and MPS driver allocation at2GB; their unified-memory allocations overlap and must not be added blindly.

No500-row training, test inference, new submission CSV, upload, installation, or paid service belongs to175. A later500 plan must preserve the historical top-level KRR selection on the original400 IDs and refit on500, while nestedA/C KRR selection uses all current training rows. No formerholdout subgroup scoring is permitted.
