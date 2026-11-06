# Reproduce the audit

Run from `.` with the existing root `.venv`. No installs, network access, MPS or new rendering are required. All output paths derive from the script's own sibling directory; creation is exclusive. Never rerun the original pilot script into its original work directory.

```sh
D=experiments/03-robustness/20261017-0030-degradation-pilot-audit
.venv/bin/python -B "$D/protect.py" freeze > "$D/protection-start.log" 2>&1
.venv/bin/python -B "$D/audit.py" inspect > "$D/inspection.log" 2>&1
.venv/bin/python -B "$D/audit.py" refit > "$D/refit.log" 2>&1
.venv/bin/python -B "$D/protect.py" verify > "$D/protection-final.log" 2>&1
.venv/bin/python -B "$D/manifest.py"
```

For a clean rerun, create a fresh directory directly under experiments and copy only audit.py, protect.py, manifest.py, PLAN.md and this RUN.md. Use that new directory as D. Do not copy outputs or delete an existing audit to reuse it. The scripts keep the project root and immutable input paths fixed.

## Inputs and fitting procedure

- Real labels: `data/train.csv`,500 supplied training sources. Original/noise/failed/reference metadata: `experiments/03-robustness/20261016-highnoise-relational/inputs.npz`.
-98 real feature columns: `experiments/02-microstructure/0800-shading-features/feat-u1dmavgH2Z-anghcor-train.npz`.
- Existing copies: four `experiments/03-robustness/1200-degradation-augmentation/synthrows-{deg,deg2,deg3,deg4}-u1dmavgH2Z-anghcor.npz` files.
- New copies: `experiments/03-robustness/20261016-infomatched-pilot/feat/synthrows-{v1a,v2a}-u1dmavgH2Z-anghcor.npz`.
- Original predictions and exact split/label/alpha metadata: the pilot's `work/oof-ridge.npz` and `work/oof-tab-{T1,T2,TD}.npz`.
- Real-only TabPFN reference: `experiments/04-residual-cnn/20261016-stepchange/work/e1/oof-F0.npz`, with the same98 columns, three seed contexts per fold and source partitions as audited in e1_tab.py/pilot.py. Its cached predictions are evaluated, not newly trained here.

All input files above are hashed in protected.json, directly or through the inherited inventory. starting-state.json records Git head, uncommitted state and installed package versions. The preceding protected study also contains incumbent-provenance with the full082/076 seeds, environment records, training/inference commands and assembly recipes. No dependency was changed.

Every valid replay uses source-filtered copies, GroupKFold5 alpha selection on real noisy inner-validation rows, training-only median imputation and standard scaling, Ridge and optional log/exp target transform. Grids are identical to the original pilot and locked in code-lock.json. Each of the90 valid outer fits must reproduce its selected alpha and predictions with tolerance1e-7 hardness units; the measured difference was exactly0, so tolerance was unused.

The corrected RS control uses seed2026101701+100*partition_index+10*outer_fold+inner_fold; final-refit offset9. The donor pool contains only new-copy source IDs in that actual fitting subset. No label donor crosses an inner or outer validation boundary. Its complete fold0 repeat poisons180 outside labels and blocks dataset reads during fitting. Neither this control repair nor the audit selects a predictive candidate.

Statistics use4000 RandomState0 source bootstrap draws. Squared errors are averaged across the three reused partitions per source before resampling. Primary high-noise set124; all-source component diagnostics400. Original fold names `20261031` and `20261101` identify pre-existing partition seeds, not independent future holdouts.

## Reproduction scope

This reproduces saved-statistic evaluation,90 original linear fits,30 corrected shuffled-control fits and one complete poison replay. It does not rerender copies, refit shared auxiliary representations, rerun TabPFN contexts, perform StageB or train a500-source candidate. It does not rerun incumbent inference on test images; it preserves and verifies the prior exact inference records and incumbent artifact hashes.

The resource cap was20 minutes of sequential CPU refitting and1GiB RSS. Actual refit/check time44.16s, peak105.61MiB. Run manifest.py last after all writers exit, without redirecting its output into the directory being hashed.
