# Candidate021 reproduction (study178)

Fixed recipe: `0.90 * unchanged candidate018 + 0.10 * mean(3 fresh full500 DINOv2-L MLPs)`. No upload is performed. Existing `.venv` only; CPU two threads, MPS encoder. All paths below are relative to this project root.

## Selection and scope

`selection-audit.json` records plan time 2026-10-05 23:13:33 UTC, before L download/extraction and original readout results. Ratio, relative layers, head settings and seeds were fixed then; readout code was hashed before fitting. Two predefined readouts were compared on original development400: MLP and nested PCA/SVR. Only the unchanged MLP blend advanced to both existing repartitions. There was no weight/layer/seed/epoch sweep in177. Local timestamps are evidence, not a trusted timestamp service.

Original OOF RMSE: 018 12.811945280527 -> fixed blend 12.786200775751. Repartitions: 12.645872618240 -> 12.621902419709; 13.048624416802 -> 13.032449212753. Pooled gain 0.021916501039. These reuse the same400 and follow adaptive research; the conditional bootstrap interval includes zero. They are not independent unbiased validation or a Public improvement claim. No additional holdout score or selection was computed during178; all500 labels were used only for the already-fixed final training.

## Explicit reuse

- Existing official frozen L weights: `experiments/01-representations/0177-dinol/dinov2_vitl14_pretrain.pth`; provenance and Apache-2.0 license copied here. No new download.
- Existing018 CSV,176 fitted models and176 test feature cache are the fixed90% component. They are deliberately reused, not newly trained in178. Both CPU/MPS saved-model replays must reproduce018 (MPS CSV byte exact).
- L raw-image features for all500 training and all1000 test examples are newly extracted in178. All three L heads are newly trained on500; each full100epoch fit is repeated with the same seed and tensor states/predictions must match exactly.
- Development400 old L features are used only for an equality check after fresh500 extraction, never as a substitute for that extraction.

## Fixed L model

Original non-register DINOv2 ViT-L/14, FP32, 504px PIL bicubic grayscale replicated toRGB, ImageNet normalization. Normalized layer indices11,17,23, each CLS/patch mean/population standard deviation:9216 dimensions. Train-only standardization; hidden64/dropout0.25,100epochs, batch64, AdamW lr0.0003 cosine minimum0.00001, weight decay0.01, gradient clip1, SmoothL1 beta1, effective auxiliary weight0. Seeds20261128,20261205,20261219; final500 offset0. No PCA or clipping. See `plan.json`, `mlp_head.py` and `train.py` for exact behavior; the inherited base CONFIG auxiliary field is overridden to0 at fit time.

## Commands executed in order

```sh
.venv/bin/python -B experiments/01-representations/0178-candidate021/prepare.py
.venv/bin/python -B experiments/01-representations/0178-candidate021/extract.py train
.venv/bin/python -B experiments/01-representations/0178-candidate021/train.py
.venv/bin/python -B experiments/01-representations/0178-candidate021/extract.py test
.venv/bin/python -B experiments/01-representations/0178-candidate021/infer.py --output experiments/01-representations/0178-candidate021/final-inference --candidate
.venv/bin/python -B experiments/01-representations/0178-candidate021/infer.py --output experiments/01-representations/0178-candidate021/reproduction-check
```

Creation paths are exclusive: do not rerun completed stages over their outputs. For another saved-model CSV replay, invoke the last command with a new output directory. It verifies locked inputs/model hashes, replays018 on both devices, reloads all L heads, and creates a CSV without fitting. Expected CSV hash is in `../../reports/candidate021-178-summary.json` and the candidate manifest.

For a fresh full L retraining reproduction, create a new sibling directory under `experiments/`, copy only the scripts, plan/code locks, runtime-origin, selection audit, provenance and licenses (not models/features/results), then execute its `extract.py train`, `train.py`, `extract.py test`, and `infer.py --output <new-output-directory>` without `--candidate`. Scripts derive the project root from that sibling location. Retain the locked177/176 inputs and vendor code. `prepare.py` is an original setup record and should not be rerun in this copied workspace. A second separate-directory raw500/full training run was not executed; all three full seed replays in the first training process were executed.

## Verification evidence

`train-feature-validation.json`, `training-validation.json`, `models/manifest.json`, `test-feature-validation.json`, `final-inference/validation.json`, `reproduction-check/validation.json`, and code locks contain raw hashes, exact reload/training checks and device differences. Test CPU/MPS regression checks cover1000 predictions; L CPU encoder checks cover the three predefined IDs000001/000500/001000, not all1000 CPU encoder extractions. The018 encoder features are reused from176.

Final CSV checks require exactly ID/hardness columns, canonical unique ordered1000 IDs, all finite values, exact float/CSV reload, unchanged018 SHA and all15 protected inputs. The broad predefined range guard is a sanity check only; no prediction clipping is applied. Prediction range is not an accuracy estimate. Network/fit/backward guards operate during new test extraction/inference; no test labels are parsed for modeling.

Resource files record peak RSS, MPS driver memory, conservative RSS+MPS, swap and memory pressure. Limits:6GB, swap growth64MiB, free memory15%, each main phase900seconds. CPU/MPS prediction tolerance0.001; encoder feature tolerance0.002. No installation, new weights, paid compute, upload or leaderboard query in178. Other environments have not been tested.
