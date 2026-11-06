"""Label-preserving synthetic degradation matched to the noisy half of train (blur ~1, noise 10-21 gray levels, contrast 0.75-1)."""
import numpy as np
from scipy import ndimage as ndi

def degrade(im_u8, rng, blur=None, noise=None, contrast=None):
    f = im_u8.astype(np.float32)
    sb = rng.uniform(0.6, 1.4) if blur is None else blur
    sn = rng.uniform(10, 21) if noise is None else noise
    c = rng.uniform(0.75, 1.0) if contrast is None else contrast
    g = ndi.gaussian_filter(f, sb) if sb > 0 else f
    m = g.mean(); g = m + c * (g - m)
    # existing noise is also blurred: top up so total noise ~ sn
    g = g + rng.normal(0, sn, f.shape)
    return np.clip(np.round(g), 0, 255).astype(np.uint8)
