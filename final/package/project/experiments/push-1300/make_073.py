"""Candidate 073 = 061 with, on non-gated rows (noise < 12 & ~failed), the TabPFN block (3 slots x 4 U-Net variants x 3 seeds) and the three plain
linear partners (RidgeCV, logRidge x4, gmm-Ridge) refitted without the 13 columns of the 'het' (size-field / tile heterogeneity) and 'intensity'
(object/image intensity statistics) groups; same weights, gated rows unchanged. Found by the push-1300 feature-group ablation (TabPFN and
Ridge screens agree; dropping these 13 columns beats 200 random 13-column drops, P < 0.005); holdout-100 consistent in sign. Nothing fitted on test.
usage (from experiments/push-1300): python make_073.py -> ../../artifacts/submission-candidate-073.csv (never overwrites)"""
import csv, json, hashlib, numpy as np
from pathlib import Path
A = Path('../../artifacts'); out = A / 'submission-candidate-073.csv'; assert not out.exists()
order = [r['ID'] for r in csv.DictReader(open('../../data/sample_submission.csv'))]
rd = lambda c: np.array([float(r['hardness']) for r in csv.DictReader(open(A / f'submission-candidate-{c}.csv'))])
p, c61 = rd('021'), rd('061'); G = np.load('../push-1200/fin-d1234n12.npz'); assert [str(i) for i in G['ids']] == order
R = np.load('../probabilistic-grains-900/ridge/final500/predictions.npz'); S = np.load('../probabilistic-grains-900/set_model/full500/predictions.npz', allow_pickle=True)
failed = R['failed'] | S['failed']; ng = (np.nan_to_num(G['noise'], nan=0) < 12) & ~failed
T = np.load('fin-nb-tab-cpu.npz'); Pn = np.load('fin-nb-partners.npz'); assert [str(i) for i in T['ids']] == order and [str(i) for i in Pn['ids']] == order
full = {c: (rd(c) - 0.25 * p) / 0.75 for c in ('038', '039', '044')}
chk = float(np.sqrt(((T['full_044'] - full['044'])[~failed] ** 2).mean())); assert chk < 0.1, chk   # rerun reproduces the production 044 slot
tab_d = np.mean([T[f'nb_{c}'] - full[c] for c in full], 0)
delta = 0.384 * tab_d + 0.128 * (Pn['ridge_nb'] - Pn['ridge_full']) + 0.128 * (Pn['logridge_nb'] - Pn['logridge_full']) + 0.15 * (Pn['gmm_nb'] - Pn['gmm_full'])
cand = np.where(ng, c61 + delta, c61); assert len(cand) == 1000 and np.isfinite(cand).all()
with out.open('x', newline='') as f:
    w = csv.writer(f); w.writerow(['ID', 'hardness']); [w.writerow([i, repr(float(v))]) for i, v in zip(order, cand)]
sha = hashlib.sha256(out.read_bytes()).hexdigest(); d = cand - c61
info = dict(candidate='073', formula='061; on noise<12 & ~failed rows TabPFN block and plain Ridge/logRidge/gmm partners without het+intensity columns (same weights)',
            n_changed_rows=int(ng.sum()), slot044_rerun_rms_vs_production=chk, change_rms_all=float(np.sqrt((d ** 2).mean())), change_rms_changed=float(np.sqrt((d[ng] ** 2).mean())),
            change_mean_changed=float(d[ng].mean()), CSV_sha256=sha, uploaded=False)
json.dump(info, open(A / 'submission-candidate-073.manifest.json', 'x'), indent=1); print(json.dumps(info, indent=1))
