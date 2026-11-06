# 표현·백본·학습 정책

초기 앙상블 재현, CNN/DINO 표현, 학습 정책과 검증 설계.

[전체 실험 분류](../README.md)

| 현재 폴더 | 기존 실험 ID | 먼저 볼 자료 | 최종 패키지 |
|---|---|---|---|
| [0175-reproduction-018](0175-reproduction-018/) | `reproduction-018-175` | [README.md](0175-reproduction-018/README.md) |  |
| [0176-reproduction-018](0176-reproduction-018/) | `reproduction-018-176` | [파일 보기](0176-reproduction-018/) | ★ |
| [0176-reproduction-018-isolated](0176-reproduction-018-isolated/) | `reproduction-018-176-isolated` | [파일 보기](0176-reproduction-018-isolated/) |  |
| [0177-dinol](0177-dinol/) | `dinol-177` | [파일 보기](0177-dinol/) | ★ |
| [0178-candidate021](0178-candidate021/) | `candidate021-178` | [README.md](0178-candidate021/README.md) | ★ |
| [0179-spatial-l](0179-spatial-l/) | `spatial-L-179` | [README.md](0179-spatial-l/README.md) |  |
| [0180-joint-readout](0180-joint-readout/) | `joint-readout-180` | [README.md](0180-joint-readout/README.md) |  |
| [0181-midlayer-l](0181-midlayer-l/) | `midlayer-L-181` | [README.md](0181-midlayer-l/README.md) |  |
| [0182-wph](0182-wph/) | `wph-182` | [README.md](0182-wph/README.md) |  |
| [0183-tessellation-diagnostic](0183-tessellation-diagnostic/) | `tessellation-diagnostic-183` | [파일 보기](0183-tessellation-diagnostic/) |  |
| [0185-pretrained-cnn](0185-pretrained-cnn/) | `pretrained-cnn-185` | [summary.json](0185-pretrained-cnn/summary.json) |  |
| [0186-cnn-optimizer](0186-cnn-optimizer/) | `cnn-optimizer-186` | `summary.json` |  |
| [0187-cnn-scale-floor](0187-cnn-scale-floor/) | `cnn-scale-floor-187` | [summary.json](0187-cnn-scale-floor/summary.json) |  |
| [0188-cnn-resolution](0188-cnn-resolution/) | `cnn-resolution-188` | [summary.json](0188-cnn-resolution/summary.json) |  |
| [0189-cnn-unfreeze](0189-cnn-unfreeze/) | `cnn-unfreeze-189` | [summary.json](0189-cnn-unfreeze/summary.json) |  |
| [0190-modern-backbones](0190-modern-backbones/) | `modern-backbones-190` | [summary.json](0190-modern-backbones/summary.json) |  |
| [0191-spatial-moments](0191-spatial-moments/) | `spatial-moments-191` | [summary.json](0191-spatial-moments/summary.json) |  |
| [0192-pca-ridge](0192-pca-ridge/) | `pca-ridge-192` | [summary.json](0192-pca-ridge/summary.json) |  |
| [0193-auxiliary-cnn](0193-auxiliary-cnn/) | `auxiliary-cnn-193` | `summary.json` |  |
| [0194-matched-replacement](0194-matched-replacement/) | `matched-replacement-194` | `summary.json` |  |
| [0195-pooling-extrema](0195-pooling-extrema/) | `pooling-extrema-195` | `summary.json` |  |
| [0196-multihead-decoder](0196-multihead-decoder/) | `multihead-decoder-196` | `summary.json` |  |
| [0197-query-normalization](0197-query-normalization/) | `query-normalization-197` | `summary.json` |  |
| [0198-candidate022](0198-candidate022/) | `candidate022-198` | `summary.json` |  |
| [0199-submission022-diagnostic](0199-submission022-diagnostic/) | `submission022-diagnostic-199` | [summary.json](0199-submission022-diagnostic/summary.json) |  |
| [0200-attention-counterfactual](0200-attention-counterfactual/) | `attention-counterfactual-200` | [summary.json](0200-attention-counterfactual/summary.json) |  |
| [0201-auxiliary-repartition](0201-auxiliary-repartition/) | `auxiliary-repartition-201` | `summary.json` |  |
| [0202-simplification](0202-simplification/) | `simplification-202` | [summary.json](0202-simplification/summary.json) |  |
| [0203-cnn-seed-expand](0203-cnn-seed-expand/) | `cnn-seed-expand-203` | [summary.json](0203-cnn-seed-expand/summary.json) |  |
| [0204-cnn-nested](0204-cnn-nested/) | `cnn-nested-204` | [HANDOFF.md](0204-cnn-nested/HANDOFF.md) · [summary.json](0204-cnn-nested/summary.json) |  |
| [0205-micronet-frozen](0205-micronet-frozen/) | `micronet-frozen-205` | [REPORT.md](0205-micronet-frozen/REPORT.md) · `summary.json` |  |
| [0206-contrast-augmentation](0206-contrast-augmentation/) | `contrast-augmentation-206` | [REPORT.md](0206-contrast-augmentation/REPORT.md) · `summary.json` |  |
| [0207-checkpoint-averaging](0207-checkpoint-averaging/) | `checkpoint-averaging-207` | [REPORT.md](0207-checkpoint-averaging/REPORT.md) · `summary.json` |  |
| [0208-update-ema](0208-update-ema/) | `update-ema-208` | [REPORT.md](0208-update-ema/REPORT.md) · `summary.json` |  |
| [0209-ema-repartition](0209-ema-repartition/) | `ema-repartition-209` | [REPORT.md](0209-ema-repartition/REPORT.md) · [summary.json](0209-ema-repartition/summary.json) |  |
| [0210-convnext-llrd](0210-convnext-llrd/) | `convnext-llrd-210` | [REPORT.md](0210-convnext-llrd/REPORT.md) · [summary.json](0210-convnext-llrd/summary.json) |  |
| [0211-huber-loss](0211-huber-loss/) | `huber-loss-211` | [REPORT.md](0211-huber-loss/REPORT.md) · `summary.json` |  |
| [0212-huber-repartition](0212-huber-repartition/) | `huber-repartition-212` | [REPORT.md](0212-huber-repartition/REPORT.md) · [summary.json](0212-huber-repartition/summary.json) |  |
| [0213-brightness-augmentation](0213-brightness-augmentation/) | `brightness-augmentation-213` | [REPORT.md](0213-brightness-augmentation/REPORT.md) · `summary.json` |  |
| [0214-gamma-augmentation](0214-gamma-augmentation/) | `gamma-augmentation-214` | [REPORT.md](0214-gamma-augmentation/REPORT.md) · `summary.json` |  |
| [0215-blur-augmentation](0215-blur-augmentation/) | `blur-augmentation-215` | [REPORT.md](0215-blur-augmentation/REPORT.md) · `summary.json` |  |
| [0216-resolution](0216-resolution/) | `resolution-216` | [REPORT.md](0216-resolution/REPORT.md) · `summary.json` |  |
| [0217-resolution-repartition](0217-resolution-repartition/) | `resolution-repartition-217` | [REPORT.md](0217-resolution-repartition/REPORT.md) · `summary.json` |  |
| [0218-grain-validity](0218-grain-validity/) | `grain-validity-218` | [파일 보기](0218-grain-validity/) |  |
| [0219-residual-cnn](0219-residual-cnn/) | `residual-cnn-219` | [파일 보기](0219-residual-cnn/) |  |
