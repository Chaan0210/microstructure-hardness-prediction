import sys, numpy as np
net, cols, sin, sout = sys.argv[1], sys.argv[2].split(','), sys.argv[3], sys.argv[4]
for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4'):
    for s in __import__('os').environ.get('REPRO062_SPLITS','train,test').split(','):
        F = np.load(f'feat-{v}{sin}-{s}.npz', allow_pickle=True); ids = [str(i) for i in F['ids']]
        Z = np.load(f'../push-400/unet4-{net}-{s}.npz', allow_pickle=True); zk = [str(k) for k in Z['keys']]; assert [str(i) for i in Z['ids']] == ids
        add = Z['X'][:, [zk.index(c) for c in cols]].astype(float)
        np.savez(f'feat-{v}{sout}-{s}.npz', X=np.hstack([F['X'].astype(float), add]), keys=np.array([str(k) for k in F['keys']] + cols), ids=np.array(ids))
        print(v + sout, s, F['X'].shape[1] + len(cols))
