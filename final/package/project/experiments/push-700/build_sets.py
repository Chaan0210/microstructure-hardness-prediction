import sys, numpy as np
spec = sys.argv[1]
for v in sys.argv[2:]:
    for split in __import__('os').environ.get('REPRO062_SPLITS','train,test').split(','):
        F = np.load(f'../push-500/feat-{v}-{split}.npz', allow_pickle=True); keys = [str(k) for k in F['keys']]; X = F['X'].astype(float)
        A = np.load(f'angle-{split}.npz', allow_pickle=True); assert [str(i) for i in A['ids']] == [str(i) for i in F['ids']]
        new, names = [], []
        if 'ang' in spec or spec == 'angh':
            new += [A['X'][:, 0], A['X'][:, 1]]; names += ['ang_c4', 'ang_s4']
        if spec == 'angh':
            for a in ('het4_u', 'sizefield_std_s32'):
                new.append(X[:, keys.index(a)] * X[:, keys.index('m_el_pix')]); names.append(f'{a}_x_m_el_pix')
        np.savez(f'feat-{v}-{spec}-{split}.npz', X=np.c_[X, np.array(new).T], keys=np.array(keys + names), ids=F['ids'])
    print('built', v, spec, names)
