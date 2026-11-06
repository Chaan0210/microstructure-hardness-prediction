"""Parameter-free seed-disagreement shrinkage; NumPy only, no new training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3]
R4=_repo_path(ROOT, 'experiments/rmse-research-20261015/round4-full073')


def reliability(predictions):
    mean=predictions.mean(0);variance=(predictions[0]-predictions[1])**2/4
    q=np.divide(mean**2,mean**2+variance,out=np.zeros_like(mean),where=(mean**2+variance)>0)
    assert np.isfinite(q).all() and np.all((q>=0)&(q<=1))
    return mean,q


def compare(a,b,y,fold):
    ae=(a-y)**2;be=(b-y)**2;rng=np.random.default_rng(2026101631);samples=[]
    for _ in range(100):
        draw=rng.integers(0,len(y),(100,len(y)))
        samples.extend(np.sqrt(ae[draw].mean(1))-np.sqrt(be[draw].mean(1)))
    order=np.argsort(ae-be)[::-1];remain=order[10:]
    return dict(gain=float(np.sqrt(ae.mean())-np.sqrt(be.mean())),
        ci95=np.quantile(samples,[.025,.975]).tolist(),
        fold_gains=[float(np.sqrt(ae[fold==k].mean())-np.sqrt(be[fold==k].mean())) for k in range(5)],
        gain_without_largest10=float(np.sqrt(ae[remain].mean())-np.sqrt(be[remain].mean())))


def main():
    z=dict(np.load(_repo_path(HERE, 'baseline076.npz')));d=dict(np.load(_repo_path(R4.parent, 'inputs.npz')));y=d['y'][d['dev']]
    old_weights=json.loads((_repo_path(R4, 'noise-calibration/summary.json')).read_text())['aux']['selection']
    out={k:z['p76'].copy() for k in ('current','confidence','uniform')};qq=np.full(400,np.nan);records=[]
    for k in range(5):
        inner=[np.load(_repo_path(R4, f'aux-seed{s}/fold{k}-selector/predictions.npz')) for s in (0,1)]
        outer=[np.load(_repo_path(R4, f'aux-seed{s}/fold{k}-final/predictions.npz')) for s in (0,1)]
        B=inner[0]['validation_index'];V=outer[0]['validation_index'];ix=np.flatnonzero(z['fold']==k)
        assert np.array_equal(z['dev'][ix],V) and set(B).isdisjoint(V)
        for j in (0,1):
            assert np.array_equal(inner[j]['validation_index'],B) and np.array_equal(outer[j]['validation_index'],V)
            assert set(inner[j]['train_index']).isdisjoint(B) and set(inner[j]['train_index']).isdisjoint(V)
            assert set(outer[j]['train_index']).isdisjoint(V)
        _,qb=reliability(np.array([q['validation_prediction'] for q in inner]))
        mean,qv=reliability(np.array([q['validation_prediction'] for q in outer]));qq[ix]=qv
        cal=np.load(_repo_path(R4, f'aux-mean2/fold{k}-selection.npz'));assert np.array_equal(cal['B'],B)
        eligible=cal['eligible']&(d['noise'][B]>=12);uniform=float(qb[eligible].mean())
        good=(z['noise'][ix]>=12)&~z['failed'][ix];weight=old_weights[k]['addition']['weights'][1]
        for name,q in [('current',np.ones(len(V))),('confidence',qv),('uniform',np.full(len(V),uniform))]:
            out[name][ix[good]]+=weight*(q*mean)[good]
        records.append(dict(fold=k,original_coefficient=weight,uniform_from_B=uniform,n_high_calibration=int(eligible.sum()),
                            mean_outer_q_diagnostic=float(qv[good].mean()),calibration_ids=d['ids'][B].tolist()))
    assert np.max(np.abs(out['current']-np.load(_repo_path(HERE, 'probe-oof.npz'))['old4']))<1e-10
    comparisons={name:compare(out['current'],v,y,z['fold']) for name,v in out.items()}
    incremental=compare(out['uniform'],out['confidence'],y,z['fold'])
    effect=comparisons['confidence'];advance=effect['gain']>=.005 and sum(g>1e-10 for g in effect['fold_gains'])>=3 and effect['ci95'][0]>0 and incremental['ci95'][0]>0
    scores={name:dict(rmse=float(np.sqrt(np.mean((v-y)**2))),folds=[float(np.sqrt(np.mean((v[z['fold']==k]-y[z['fold']==k])**2))) for k in range(5)]) for name,v in out.items()}
    result=dict(scores=scores,vs_current=comparisons,confidence_vs_uniform=incremental,advance=bool(advance),
                calibration=records,parameters_fitted=False,Public_used=False,
                limitation='Heuristic variance from two correlated seeds. Conditional reused-ID intervals; not independently calibrated uncertainty.')
    (_repo_path(HERE, 'confidence-results.json')).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(_repo_path(HERE, 'confidence-oof.npz'),ids=z['ids'],fold=z['fold'],y=y,q=qq,**out)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
