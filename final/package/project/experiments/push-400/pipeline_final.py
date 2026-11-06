import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'push-300'))
NETS = {'m1': None, 'm2': (5.6343302726745605, 0.9692867994308472), 'h1': None, 'm3': (5.6343302726745605, 0.9692867994308472),
        'm4': (5.6343302726745605, 0.9692867994308472)}   # (LA_MU, LA_SD) of the training set; None = default (219-image set)
U1COLS = ['u_ph', 'u_ph_hard', 'u_ph_f32']
DROP = ['f_dark_20', 'f_dark_35', 'f_dark_50', 'sp_frac_grains', 'sp_count_frac', 'z_q50', 'z_q75', 'z_q90', 'vstd_sp_frac_grains', 'vstd_sp_count_frac',
        'vstd_f_dark_20', 'vstd_f_dark_35']
MCOLS = ['m_el_glob', 'm_el_loc16', 'm_el_pix', 'm_el_coh', 'm_la_std', 'm_la_f16', 'm_la_f32', 'm_la_tile8', 'm_la_mean', 'm_ph_hard']
AVGS = {'mavg': ['m1', 'm2', 'h1'], 'mavg4': ['m1', 'm2', 'h1', 'm3'], 'mavg5': ['m1', 'm2', 'h1', 'm3', 'm4']}

def ld(p): return np.load(p, allow_pickle=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['unet', 'assemble']); ap.add_argument('arg', nargs='?')
    ap.add_argument('--out', required=True); ap.add_argument('--unet-dir', default=None, help='read U-Net feature npz from here instead of --out (assemble)')
    ap.add_argument('--base-pattern', default=str(HERE.parent / 'push-300/feat-a1s250n4vstd-stcoh-{split}.npz'))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    py = str(P / '.venv/bin/python')
    if a.stage == 'unet':
        env = dict(os.environ, OUTDIR=str(out), NOMAPS='1')
        subprocess.run([py, str(HERE / 'unet_feats.py'), 'u1', a.arg], env=env, check=True)
        for n, la in NETS.items():
            e = dict(env)
            if la: e.update(LA_MU=repr(la[0]), LA_SD=repr(la[1]))
            subprocess.run([py, str(HERE / 'unet2_feats.py'), n, a.arg, n], env=e, check=True)
    elif a.stage == 'assemble':
        s = a.arg; ud = Path(a.unet_dir) if a.unet_dir else out
        F = ld(a.base_pattern.format(split=s)); keys = [str(k) for k in F['keys']]; ids = F['ids']
        U = ld(ud / f'unet-u1-{s}.npz'); uk = [str(k) for k in U['keys']]; assert (U['ids'] == ids).all()
        X = np.hstack([F['X'].astype(float), U['X'][:, [uk.index(c) for c in U1COLS]].astype(float)]); k1 = keys + U1COLS   # = feat-u1ph
        keep = [i for i, c in enumerate(k1) if c not in DROP]; XD = X[:, keep]; kd = [k1[i] for i in keep]                  # = feat-u1phdrop
        np.savez(out / f'feat-u1phdrop-{s}.npz', X=XD, keys=np.array(kd), ids=ids)
        M = {}
        for n in NETS:
            D = ld(ud / f'unet2-{n}-{s}.npz'); dk = [str(k) for k in D['keys']]; assert (D['ids'] == ids).all(); M[n] = (D, dk)
            np.savez(out / f'feat-u1d{n}sz-{s}.npz', X=np.hstack([XD, D['X'][:, [dk.index(c) for c in MCOLS]].astype(float)]), keys=np.array(kd + MCOLS), ids=ids)
        for tag, nets in AVGS.items():
            A = np.mean([M[n][0]['X'][:, [M[n][1].index(c) for c in MCOLS]] for n in nets], 0)
            np.savez(out / f'feat-u1d{tag}-{s}.npz', X=np.hstack([XD, A]), keys=np.array(kd + MCOLS), ids=ids)
        print(json.dumps(dict(stage='assemble', split=s, n_cols_u1phdrop=len(kd))))
    else:
        raise SystemExit(f'unknown stage {a.stage}')

if __name__ == '__main__':
    main()
