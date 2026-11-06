# Reproduction (run from the project root unless noted)

Environments (unchanged): `.venv` (numpy 2.5.3, scikit-learn 1.9.1, torch 2.14.0); `experiments/frameworks/tabpfn-v2/.venv` (tabpfn 9.1.0,
TabPFN v3.5 checkpoint `experiments/frameworks/tabpfn-v3/weights/tabpfn-v3.5-20260909.safetensors`).

```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/protect.py start
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/diag/diagnose.py
```
```bash
.venv/bin/python -B experiments/03-robustness/20261017-infogain/diag/cnn_probe.py
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/h1/measure.py deg4
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/h1/gates.py
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/h3/spectral.py
```
```bash
experiments/03-robustness/20261017-infogain/h3/run_screen.sh
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/h3/eval_screen.py
```
```bash
.venv/bin/python -B experiments/03-robustness/20261017-infogain/h4/spec_residual.py
```
```bash
.venv/bin/python experiments/03-robustness/20261017-infogain/protect.py end
```

Configuration:
- **H3 screen.** TabPFN v3.5, default n_estimators, `random_state = 100·sd + fold` (sd 0, 1, 2), device CPU with 3 threads per
  stream, original partition (`resolution-216/ensemble-OOF.npz` folds), features `push-1300/abl/feat-h9phys-{nb,full}-044-{variant}-train.npz`.
  +MEAN appends `h3/spec-train.npz` column 0; +RAND appends `RandomState(20261017).standard_normal(500)`.
- **H4.** Teachers, context and gating from round4 (`teacher_oof`, `raw_context`); RidgeCV(alphas = logspace(−2, 4, 25)) after
  median imputation and standardisation; coefficient on B against the anchors' p76B/p82B, clipped to [0, 1]; holdout and
  re-partitions use the mean fold coefficient. Bootstrap seed 20261017.
- **CNN probe.** RNG seed 20261017 for shuffles and phase randomisation; round4 `aux-seed{0,1}/fold{k}-final/model.pt`.
- **Not run** (no candidate advanced): `confirm/tab_confirm.py`, `confirm/eval_confirm.py`. They are ready for the full-block
  `im_mean` experiment in REPORT §5.
