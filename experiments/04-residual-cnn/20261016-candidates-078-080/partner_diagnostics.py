"""Descriptive audit of078; no fitting or test-label inference."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
import numpy as np
import study as s


def main():
    h=_repo_path(s.ROOT, 'experiments/push-1300')
    d=dict(np.load(_repo_io(s.R4.parent/'inputs.npz')));dev=d['dev'];ids=d['ids'][dev]
    gate=d['noise'][dev]<12
    old=np.load(_repo_io(h/'dev-combo.npy'));new=np.load(_repo_io(h/'dev-078.npy'));y=d['y'][dev]
    cache=np.load(_repo_io(h/'h9-partners.npz'))
    partners={}
    for key,weight in [('ridge',.128),('logridge',.128),('gmm',.15)]:
        delta=weight*(cache['phys_'+key]-cache['none_'+key])
        change=np.where(gate&~d['fail'],delta,0.)
        partners[key]=dict(weight=weight,dev_change_rms=float(np.sqrt(np.mean(change**2))),
                           pooled_rmse_before=float(np.sqrt(np.mean((old-y)**2))),
                           pooled_rmse_after=float(np.sqrt(np.mean((old+change-y)**2))),
                           partition_gains=[float(np.sqrt(np.mean((old[p]-y)**2))-np.sqrt(np.mean((old[p]+change[p]-y)**2))) for p in range(3)])
    train=np.load(_repo_io(h/'cumul-train.npz'));test=np.load(_repo_io(h/'cumul-test.npz'))
    tx=train['X'][:,[1,4]];ex=test['X'][:,[1,4]]
    saved=np.load(_repo_io(s.R4/'final500/test-predictions.npz'));test_ids=saved['ids'].tolist()
    assert test['ids'].tolist()==test_ids
    pos={str(i):j for j,i in enumerate(train['ids'])}
    tx=tx[[pos[str(i)] for i in d['ids']]]
    testgate=(saved['noise']<12)&~saved['failed']
    clipped=(ex<tx.min(0))|(ex>tx.max(0))
    trgate=d['noise']<12
    # Feature-only transport diagnostics; never learn coefficients from test.
    columns={}
    for j,key in enumerate(('p34_sigma1','contrast_sigma1')):
        columns[key]=dict(train_low_quantiles=np.quantile(tx[trgate,j],[0,.01,.5,.99,1]).tolist(),
                          test_low_quantiles=np.quantile(ex[testgate,j],[0,.01,.5,.99,1]).tolist(),
                          clipped_low_ids=np.array(test_ids)[clipped[:,j]&testgate].tolist())
    p76=s.read_csv(_repo_path(s.ROOT, 'artifacts/submission-candidate-076.csv'),test_ids)
    p78=s.read_csv(_repo_path(s.ROOT, 'artifacts/submission-candidate-078.csv'),test_ids)
    delta=p78-p76;top=np.argsort(delta**2)[-10:][::-1]
    out=dict(partners=partners,feature_transport=columns,
             cv_clipping_mismatch='h9_partners.py did not clip outer-validation features to each training fold range; final_h9c_partners.py clips test to full500 train range.',
             test_delta_rms=float(np.sqrt(np.mean(delta**2))),
             dev_delta_rms=float(np.sqrt(np.mean((new-old)**2))),
             low_rows_clipped_any=int((clipped.any(1)&testgate).sum()),
             correction_energy_share_clipped=float(np.sum(delta[clipped.any(1)]**2)/np.sum(delta**2)),
             largest10_test_corrections=[dict(ID=test_ids[i],correction=float(delta[i]),clipped=bool(clipped[i].any())) for i in top],
             conclusions=['Public identifies aggregate low-partner deterioration, not which partner or rows caused it.',
                          'Train/test feature summaries and change magnitude are descriptive, not causal failure attribution.',
                          'The CV/production clipping mismatch is real; its effect requires train-only refits to quantify.',
                          'Do not optimize test outliers, thresholds, or weights from these diagnostics.'])
    s.save(_repo_io('partner-diagnostics.json'),out)
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
