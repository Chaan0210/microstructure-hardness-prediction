# Handover — 2026-10-16 ~21:55 KST (written at a usage-limit stop)

**UPDATE 23:50: pilot COMPLETED and branch CLOSED. See REPORT.md. Nothing is running. The child session is idle and has acknowledged the handover.**

## State
- **Child session** `local_07c6a848…` (Information-matched noisy copies): turn stopped twice, its sweeps were killed, and a handover message was sent. Its directory `experiments/03-robustness/20261016-infomatched-copies/` is read-only and is not used for results.
- **Still running, detached** (both mine, both write only into this directory):
  - `zsh pipe/run_set.sh v1a`
  - `zsh pipe/run_set.sh v2a`
  - Each takes about 45 min (4-view Cellpose segmentation at about 10 s/image, then U-Nets and rows).
  - Completion is marked by `DONE v1a` and `DONE v2a` in `logs/progress.txt`.
  - Outputs: `feat/synthrows-v1a-u1dmavgH2Z-anghcor.npz`, `feat/synthrows-v2a-u1dmavgH2Z-anghcor.npz`.
- **Done:** frozen protocol (`PLAN.md` + amendment A1), render sets v1a/v1b/v2a/v2b (`synth/`), D3 hardness-distribution diagnostic (`work/diag/d3.json`).
  - D3 result: no mean shift between clean sources and real noisy rows. The spread is somewhat wider on real noisy rows (SD 19.6 vs 17.1, CI of ratio 0.99–1.33).
- **Not run:** D1/D2 diagnostics, all labelled arms, the evaluation.

## To finish (from the project root, after both DONE lines appear)
```
.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py diag
.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py ridge
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py tab T1
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py tab T2
experiments/frameworks/tabpfn-v2/.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py tab TD
.venv/bin/python experiments/03-robustness/20261016-infomatched-pilot/work/pilot.py eval
```
- **Decision rule** (PLAN.md): the primary comparisons are R3 vs R1 (Ridge and logRidge) and T2 vs T0. All of the following are required:
  - dMSE < 0 with P ≥ 0.95
  - the same sign in all 3 partitions and in at least 4/5 original-partition folds
  - still < 0 after removing the top-5 helped images
  - better than the duplicate (RD/TD) and shuffled-label (RS) controls
- Otherwise close the branch.
- Stage B (fold-honest renderer recalibration) runs only if Stage A passes.

## Candidate next hypotheses if the branch closes (not yet ranked by any experiment)
- **Residual-CNN family: retarget to the 076 teacher.** Status seen read-only in `076-complement-20261016`: folds 0–1 done; folds 2–4 teacher stage stopped at its own 70-min budget. Its 2-fold pilot correlated 0.9998 with the existing correction, so expected change is small.
- **Clean-row phase measurement.** Per-grain darkness overlaps between phases in some clean images (stepchange REPORT). The moment-based H9 phase fraction is the most recent measurement that transferred to Public. A hierarchical (shared-shape) per-grain mixture is a cheap CPU test with a random-column control.

## 082 private projection — corrected
- The recorded Public improvement is a fact: −0.0304 on 300 rows.
