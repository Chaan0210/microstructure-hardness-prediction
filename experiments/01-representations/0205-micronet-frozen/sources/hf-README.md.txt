---
license: mit
tags:
  - computer-vision
  - microscopy
  - materials-science
  - encoder
  - segmentation
  - pytorch
  - pretrained
library_name: pretrained-microscopy-models
---

# Pretrained Microscopy Encoder - resnet50 (micronet)

This is a `resnet50` encoder pretrained on `micronet` microscopy datasets, prepared for use with [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).

Models originally pretrained for [pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models)

## Model Details

- **Architecture**: resnet50
- **Pretrained on**: micronet
- **Input shape**: RGB images
- **Framework**: PyTorch
- **Use case**: Feature extraction, segmentation backbone

## Files

- `encoder_weights.pth` - PyTorch `state_dict()` of the encoder
- `README.md` - This model card
- `encoder.py` - Sample code to use the encoder within UNet.

## License

mit
