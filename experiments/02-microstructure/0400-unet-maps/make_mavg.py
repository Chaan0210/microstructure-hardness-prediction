"""Average multi-task U-Net feature columns over several nets (reduces U-Net training noise in the features).
usage: make_mavg.py OUT_TAG NET[,NET...] [phase_too]
  writes unet2-<OUT_TAG>-{train,test}.npz with the 10 size/elong/phase cols averaged over NETs (unet2-<NET>-<split>.npz).
  phase_too: also write feat-u1d<OUT_TAG>-{split}.npz = feat-u1phdrop with u_ph/u_ph_hard/u_ph_f32 replaced by the mean of
             u1 and the nets' m_ph/m_ph_hard/m_ph_f32, plus the averaged 10 cols."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, numpy as np
out, nets = sys.argv[1], sys.argv[2].split(','); phase = len(sys.argv) > 3 and sys.argv[3] == 'phase_too'
cols = ['m_el_glob', 'm_el_loc16', 'm_el_pix', 'm_el_coh', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_tile8', 'm_la_mean', 'm_ph_hard']
for s in ('train', 'test'):
    D = [np.load(_repo_io(f'unet2-{t}-{s}.npz'), allow_pickle=True) for t in nets]; k = [str(x) for x in D[0]['keys']]
    assert all((d['ids'] == D[0]['ids']).all() for d in D)
    X = np.mean([d['X'][:, [k.index(c) for c in cols]] for d in D], 0)
    np.savez(_repo_io(f'unet2-{out}-{s}.npz'), ids=D[0]['ids'], X=X, keys=np.array(cols))
    F = np.load(_repo_io(f'feat-u1phdrop-{s}.npz'), allow_pickle=True); fk = [str(x) for x in F['keys']]; FX = F['X'].astype(float).copy()
    assert (F['ids'] == D[0]['ids']).all()
    if phase:
        for uc, mc in (('u_ph', 'm_ph'), ('u_ph_hard', 'm_ph_hard'), ('u_ph_f32', 'm_ph_f32')):
            FX[:, fk.index(uc)] = np.mean([FX[:, fk.index(uc)]] + [d['X'][:, k.index(mc)] for d in D], 0)
    tag = f'u1d{out}' + ('p' if phase else '')
    np.savez(_repo_io(f'feat-{tag}-{s}.npz'), X=np.hstack([FX, X]), keys=np.array(fk + cols), ids=F['ids'])
    print(s, tag, np.hstack([FX, X]).shape)
