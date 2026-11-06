"""Absolute-intensity summary features (label-free)."""
import numpy as np
from scipy import ndimage as ndi
KEYS = ['br_mean', 'br_median', 'br_p75s', 'br_p95s', 'br_p1s', 'br_range_s']
def features(im_u8):
    a = im_u8.astype(np.float64); s = ndi.gaussian_filter(a, 2)
    return dict(zip(KEYS, [a.mean(), np.median(a), np.percentile(s, 75), np.percentile(s, 95), np.percentile(s, 1), np.percentile(s, 99) - np.percentile(s, 1)]))
def matrix(images):
    return np.array([[features(i)[k] for k in KEYS] for i in images], float)
