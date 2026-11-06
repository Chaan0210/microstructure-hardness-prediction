# Study179: DINO-L spatial replacement

Replace B spatial neighbor co-change statistics with L statistics, holding all dimensionality, physics, upstream anchors, residual training and globalL021 term fixed. No concatenation arm.

Normalized final layer tokens36x36, projected32. Half mean outerproduct over2520 horizontal/vertical nearest-neighbor differences, no wrap; uppertriangle528. Same definition as018.

Gaussian QR with seed20271201, B768x32/L1024x32, no fitted PCA or labels. Same procedure/rank, different coordinate bases due different channel widths.

Replace only B528 in physics57+B528 by L528. No concatenation/addition, layer/rank/seed/grid changes. MinMaxclip andStandardScaler trainonly; block1/sqrt(dim). Existing none/ridge1/ridge01/rbf1/rbf01, lambda*n, trainlocal residualRMS andRBF distance/kernel centering. Each honest block inner3fold;512groupbootstrap selectionfrequency seed20271204+10000*splitindex+100*fold+block,mean3honestblocks.

021_swap = .9*(unchanged015base + newly fitted L residual mean3) + .1*fixed177 LglobalMLP. Control021=.9*018+.1*sameLglobal. Also report018-equivalent swap as attribution only, never fallback candidate.

## Result

Outcome: rejected_no_new_candidate; partitions: ['original'].

{
  "original": {
    "018": 12.811945280526919,
    "021": 12.786200775750695,
    "globalL": 13.897309666752351,
    "B_replay": 12.811945280526919,
    "L_swap018": 12.813108034745927,
    "primary": 12.788492654890762
  }
}

## Validation and reuse

All B andL residual fits use the same honest blocks and inner folds. Existing certified upstream anchors and015base are reused; no training against global OOF residuals. B inner predictions/full predictions/512-bootstrap weights and018/021 controls must match exactly. All main residual objects are reloaded and checked with independent scaling and dual Ridge/eigen RBF calculations. All three fold0 L blocks are refitted after outer labels are poisoned; existing upstream exclusion certificates are reused. The scope of this replay is explicit, not a complete upstream neural retraining.

Feature checks include three cached B samples against the old projection/statistic, independent2520edge enumeration, three L CPU/MPS samples, token-grid D4 invariance, spatial shuffling with unchanged global mean/std, exact400 globalL final-layer summaries and image hashes. Token-grid invariance does not imply whole-image DINO invariance.

No concatenation arm, alternative rank/layer/seed/ratio, holdout/test access, full500 fit, new CSV, install, download, cost or upload. Existing018/021 preserved. Scores remain adaptive development evidence.

## Commands

Run from the project root using the existing .venv. Completed output paths are exclusive; preserve them. The original setup and code locks are recorded before extraction/fitting.

```sh
.venv/bin/python -B experiments/01-representations/0179-spatial-l/prepare.py
.venv/bin/python -B experiments/01-representations/0179-spatial-l/extract.py
.venv/bin/python -B experiments/01-representations/0179-spatial-l/readout.py original
.venv/bin/python -B experiments/01-representations/0179-spatial-l/finalize.py
```

Evidence: plan.json, code locks, feature-validation.json, each partition report.json/predictions.npz and all block models/reports, resource traces; project reports/spatial-L-179-summary.json.
