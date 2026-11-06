"""[h9s2: + sigma 2,4 p34 and contrast] Test predictions of the H9 TabPFN block (nb+2 cumulant cols for non-gated rows, full+2 cols for gated rows): per slot, mean over 4 U-Net variants x
seeds (random_state 100*sd, sd 0,1,2), fit on all 500 train rows (production finalize procedure). -> fin-h9s2-tab-cpu.npz"""
import sys, csv, numpy as np, warnings
warnings.filterwarnings('ignore'); sys.path.insert(0, '../push-300'); import harness
lab = {r['ID']: float(r['hardness']) for r in csv.DictReader(open('../../data/train.csv'))}; order = [r['ID'] for r in csv.DictReader(open('../../data/sample_submission.csv'))]
def member(ftr, fte):
    Tr = np.load(ftr, allow_pickle=True); Te = np.load(fte, allow_pickle=True); assert [str(i) for i in Te['ids']] == order and list(Tr['keys']) == list(Te['keys'])
    X, y, Xe = Tr['X'].astype(float), np.array([lab[str(i)] for i in Tr['ids']]), Te['X'].astype(float); ps = []
    for sd in (0, 1, 2): m = harness.tabpfn_new('v3.5', 100 * sd, 'cpu'); m.fit(X, y); ps.append(m.predict(Xe))
    return np.mean(ps, 0)
out = {}
for s in ('038', '039', '044'):
    for fs in ('nb', 'full'):
        out[f'{fs}_{s}'] = np.mean([member(f'abl/feat-h9s2-{fs}-{s}-{v}-train.npz', f'abl/feat-h9s2-{fs}-{s}-{v}-test.npz') for v in ('u1dm1sz', 'u1dm2sz', 'u1dmavg', 'u1dmavg4')], 0); print(fs, s, flush=True)
        np.savez('fin-h9s2-tab-cpu.npz', ids=np.array(order), **out)
