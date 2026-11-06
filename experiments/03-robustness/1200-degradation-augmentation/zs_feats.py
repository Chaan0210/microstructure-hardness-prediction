"""Image-level features from ZS-net zone predictions R [n, folds, 16, 4] (b, lam, ph, lcnt).
Cross-fitted use: for clean train images seen by one fold model, the other fold's prediction is used (pick='cross'); otherwise mean over folds."""
import numpy as np
KF = ['zs_het_b', 'zs_het_lam', 'zs_ph_std', 'zs_rng_b', 'zs_b_mean', 'zs_lam_mean', 'zs_ph_mean', 'zs_lcnt_mean', 'zs_lcnt_std', 'zs_hp_b']
def feats_from_zones(Z):
    b, lam, ph, lc = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    return np.array([b.std(), lam.std(), ph.std(), b.max() - b.min(), b.mean(), lam.mean(), ph.mean(), lc.mean(), lc.std(), -2 * np.log(np.mean(np.exp(-0.5 * b)))])
def image_feats(R, ids, folds, pick='mean'):
    """pick: 'mean' (average folds) or dict id -> fold index to use."""
    out = []
    for i, iid in enumerate(ids):
        Z = R[i].mean(0) if not isinstance(pick, dict) or iid not in pick else R[i, pick[iid]]
        out.append(feats_from_zones(Z))
    return np.array(out)
