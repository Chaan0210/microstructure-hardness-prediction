"""Identical auxiliary arrays with one decompression of the original mask bank.

An NPZ lookup returns a fresh full array. Keeping one selected view from each
lookup can retain hundreds of full arrays. Materialize the mask bank once.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import numpy as np
from phase_joint_v3 import ROOT,targets

def load_masks_once():
    with np.load(_repo_path(ROOT, 'experiments/grain-cellpose-220/masks-v3-train.npz')) as z:
        return {'ids':z['ids'],'masks':z['masks']}

def prepare_safe(d, px, tr, out):
    path = out / 'auxiliary.npz'
    if path.exists():
        a = dict(np.load(_repo_io(path)))
        assert np.array_equal(a['train'], tr)
        mp = out / 'auxiliary-masks.npy'
        if not mp.exists():
            masks = load_masks_once()
            mi = {str(v): j for j, v in enumerate(masks['ids'])}
            np.save(_repo_io(mp), np.array([masks['masks'][mi[str(d['ids'][j])]] for j in a['indices']], dtype=np.uint16))
        a['masks'] = np.load(_repo_io(mp), mmap_mode='r')
        return a
    masks = load_masks_once()
    mi = {str(v): j for j, v in enumerate(masks['ids'])}
    ix = []
    classes = []
    reg = []
    valid = []
    for j in tr:
        if d['noise'][j] >= 9.5:
            continue
        m = masks['masks'][mi[str(d['ids'][j])]]
        if m.max() < 20:
            continue
        t, r, v = targets(px[j, 0], m)
        if not v.any():
            continue
        ix.append(j)
        classes.append(t.astype(np.uint8))
        reg.append(r)
        valid.append(v)
    rr = np.array(reg)
    vv = np.array(valid)
    la_mu = float(np.mean([r[0][v].mean() for r, v in zip(rr, vv)]))
    la_sd = float(np.mean([r[0][v].std() for r, v in zip(rr, vv)]) + 0.5)
    a = dict(train=tr, indices=np.array(ix), classes=np.array(classes), reg=rr, valid=vv, mean=np.array([la_mu, 0, 0, 0.15], np.float32), scale=np.array([la_sd, 0.3, 0.3, 0.15], np.float32))
    np.savez_compressed(_repo_io(path), **a)
    np.save(_repo_io(out / 'auxiliary-masks.npy'), np.array([masks['masks'][mi[str(d['ids'][j])]] for j in a['indices']], dtype=np.uint16))
    a['masks'] = np.load(_repo_io(out / 'auxiliary-masks.npy'), mmap_mode='r')
    return a
