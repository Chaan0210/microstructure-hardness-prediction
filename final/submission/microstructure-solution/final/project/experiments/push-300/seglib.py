import numpy as np
from pathlib import Path
from PIL import Image
R220 = Path(__file__).resolve().parents[1] / 'grain-cellpose-220'
_model = None

def model():
    global _model
    if _model is None:
        from cellpose import models
        _model = models.CellposeModel(gpu=True, pretrained_model=str(R220 / 'weights/cpsam'))
    return _model

def seg(im):
    m, _, _ = model().eval(im.copy(), flow_threshold=0.8, cellprob_threshold=-2.0)
    if (m > 0).mean() < 0.95:
        m2, _, _ = model().eval(im.copy(), flow_threshold=1.0, cellprob_threshold=-3.0)
        if (m2 > 0).mean() > (m > 0).mean(): m = m2
    return m.astype(np.uint16)

def seg_up2(im):
    """segment a 2x bicubic-upsampled image, return labels at native resolution (nearest)."""
    up = np.asarray(Image.fromarray(im).resize((512, 512), Image.BICUBIC))
    m = seg(up)
    return m[::2, ::2].copy()

def seg_down2(im):
    """segment a 2x box-downsampled image (noise sigma halves), return labels at native resolution (nearest)."""
    d = im.astype(np.float32).reshape(128, 2, 128, 2).mean((1, 3))
    m = seg(np.clip(np.round(d), 0, 255).astype(np.uint8))
    return np.kron(m, np.ones((2, 2), m.dtype))
