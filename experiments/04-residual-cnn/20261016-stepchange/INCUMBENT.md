# Incumbent record — candidate 076 (taken 2026-10-16 20:30–20:45, before any experiment)

**Verified best submitted file:** `artifacts/submission-candidate-076.csv`, Public RMSE 8.9646534604 (#53, 2026-10-15 23:49 KST,
`reports/submissions/candidate076-public-submission.json`), leaderboard rank 5 at 20:30 (leader 8.5843). The page is in auto mode, so 076 is the current final selection.
Candidate 082 (076 + residual-CNN correction) was never submitted. Its Public reading of 8.9343 is arithmetic from 076/078/080, not measured.

## Identity
| item | value |
|---|---|
| 076 CSV SHA-256 | `b1be09b6c060de1a92191bbf31f76f1c501a126493976771f335d7ff3242e3f0` |
| git HEAD | `f8f2b3add5df3a01c4550b20587a33890a6c99e5` (`record/git-head.txt`) |
| uncommitted state | the whole 061→076 chain (push-1200, push-1300 and later) is **untracked** (`record/git-status-porcelain.txt`, 139 entries). Its code identity is therefore the SHA-256 record, not git. |
| protected files | 17,245 files SHA-256 hashed (`record/protected-sha256.json`): data/, artifacts/submission-candidate-*, reports/, experiments/{push-300,400,800,900,1200,1300}, probabilistic-grains-900, resolution-216/217, rmse-research-20261015, 078-080-update-20261016, grain-cellpose-220/features |
| whole tree | 265,950 files size+mtime (`record/tree-stat.json`). Excluded: .git, this directory, and the two directories another session was writing (076-complement-20261016, push-1500). |
| environments | Python 3.13.3. `.venv`: numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, torch 2.14.0. `tabpfn-v2/.venv`: tabpfn 9.1.0 (TabPFN v3.5 checkpoint `tabpfn-v3.5-20260909.safetensors`), torch 2.14.1. `grain-cellpose-220/.venv`: cellpose 4.2.1.1. Full lists: `record/freeze*.txt`. Nothing was installed or upgraded. |

Direct inputs of the assembly (SHA-256 prefix): 074 `03a9c71f`, 073 `6a243cfb`, 061 `259d69c4`, 053 `a6365c07`, 021 `7cf4ce3b`, 038 `e6eb78af`,
039 `bfc64992`, 044 `f9fff6f3`, `push-1300/fin-h9-tab-cpu.npz` `cc6215e6`, `fin-nb-tab-cpu.npz` `1475c71c`, `fin-srcwB.npz` `168532ba`,
`fin-nb-partners.npz` `15392369`, `push-1200/fin-d1234n12.npz` `03daef1a`, `fin-none.npz` `917b4110`, failure masks
`probabilistic-grains-900/{ridge/final500,set_model/full500}/predictions.npz` `21b02388`/`2cdb3569`. Scripts: make_061 `439e353b`, make_073 `38661c08`,
make_074 `b4e890f3`, make_076 `0c03650b`, final_h9_tab `8606a5bc`.

## Structure and seeds
076 = where(failed, CNN021, 0.16·CNN021 + 0.384·TabPFN block + 0.128·RidgeCV + 0.128·logRidge + 0.05·set NN + 0.15·gmm-Ridge).
- Non-gated rows (noise < 12): TabPFN `nb+h9` members and plain `nb` partners.
- Gated rows (noise ≥ 12 & not failed, 318 test rows): TabPFN `full+h9` members, DTA Ridge/logRidge (copies deg..deg4 with noise ≥ 12, arm-B alpha), and gmm `full`.
- TabPFN block: 3 slots (038/039/044) × 4 U-Net variants × seeds `random_state = 100·sd (+fold in dev)`, sd ∈ {0,1,2}, TabPFN v3.5, context = all 500 train rows (final) or the 320 training-fold rows (dev).

## Commands (from the project root)
Full training is not re-run here: the U-Nets, Cellpose features, TabPFN fits and CNN021 are frozen artifacts.
- Recorded assembly: `push-1200/make_061.py` → `push-1300/make_073.py` → `make_074.py` → `make_076.py`, each run from its own directory.
- Re-execution with redirected output: `.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/repro_chain.py [chain|chain-post]`
- Component-level rebuild, run from `experiments/03-robustness/1300-robust-ensembles`. This refits the deterministic linear partners and reuses the saved TabPFN, CNN021 and set outputs:
  `../tabpfn-v2/.venv/bin/python ../stepchange-20261016/repro/components/c14_partners.py .. ../push-1200 deg,deg2,deg3,deg4 ../stepchange-20261016/repro/components/c14p-orig.npz`
  `../tabpfn-v2/.venv/bin/python ../stepchange-20261016/repro/components/c14_blend.py .. abl ORIG076 fin-h9-tab-cpu.npz ../stepchange-20261016/repro/components/c14p-orig.npz ../stepchange-20261016/repro/components/c14b-orig`
  The blend script is copied from `push-1400/R/experiments/push-1300/c14_blend.py`. The only change is the `ORIG076` tag, which maps to the production file names (`dev-h9phys-*-044-*` for slot 044).
- Verification: `.venv/bin/python experiments/04-residual-cnn/20261016-stepchange/protect.py verify`

## Reproduction results
| check | before experiments | after experiments |
|---|---|---|
| chain 061/073/074/076 rebuilt vs stored | byte-identical (SHA match) | byte-identical (`repro/chain-post/`) |
| component rebuild test vs stored 076 CSV | max abs diff 4.5e-13 | 4.5e-13; bit-identical to the pre-experiment rebuild (diff 0.0) |
| component rebuild dev OOF | 9.5462 (clean 7.983 / mid 8.166 / noisy 12.265) = recorded push-1400 value | identical |
| protected SHA-256 (17,245 files) | — | 0 changed |
| tree size/mtime (265,950 files) | — | 0 modified/deleted. 123 added, all in `experiments/04-residual-cnn/20261016-2035-plateau-break/` (another session, not this one) |

Tolerance: the 4.5e-13 residual comes from float64 summation order (delta chain vs. direct component sum). It is nine orders of magnitude below the 0.01 precision that matters, and the recorded chain itself reproduces the stored bytes exactly.
This reproduces predictions from saved components. It does **not** reproduce the U-Net, Cellpose, TabPFN or CNN021 training runs.
