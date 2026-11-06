"""Fixed 3-seed CNN averaging on original development folds only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,csv,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import frozen_validation as fv

NEW_SEEDS=(20261023,20261043)
def read(p):
    with p.open() as f:return {r['ID']:r for r in csv.DictReader(f)}
def main():
    out=_repo_path(ROOT, 'artifacts/run-018');out.mkdir(exist_ok=False)
    original=read(_repo_path(ROOT, 'artifacts/run-012/development_oof.csv'));ids=sorted(original)
    split=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
    assert len(ids)==400 and set(ids)==set(split['development_folds']) and not set(ids)&set(split['holdout'])
    y=np.array([float(original[s]['hardness']) for s in ids]);fid=np.array([split['development_folds'][s] for s in ids]);groups=np.array([split['group_by_id'][s] for s in ids])
    c=read(_repo_path(ROOT, 'artifacts/run-009/development_oof.csv'))
    baseline=np.array([float(original[s]['baseline']) for s in ids]);kernel=np.array([float(original[s]['nested_kernel']) for s in ids]);oldcnn=np.array([float(c[s]['cnn']) for s in ids]);current=.5*baseline+.25*kernel+.25*oldcnn
    np.testing.assert_allclose(current,[float(original[s]['nested_kernel_cnnblend']) for s in ids],rtol=0,atol=1e-10)
    protected={p:fv.sha(_repo_path(ROOT, p)) for p in ['artifacts/submission-candidate-003.csv','artifacts/final-model-017/cnn.pt','artifacts/final-model-017/kernel.joblib','artifacts/final-model-017/baseline.joblib']}
    plan={'hypothesis':'Reduce stochastic CNN variance by equal prediction averaging three prespecified seeds; no seed selection, no weight averaging across seeds','seeds':[20261003,*NEW_SEEDS],'endpoint_epochs':80,'cnn_weight':.25,'kernel_weight':.25,'baseline_weight':.5,'folds':'original development400 five groups','future_confirmation_seeds':[20261031,20261101],'no_test_holdout_reads':True,'protected':protected,'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['cnn_multiseed.py','frozen_validation.py','cnn_controls.py']}}
    fv.write(out/'plan.json',plan);start=time.perf_counter();images,hashes=fv.load_development_images(ids)
    torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';predictions=[oldcnn];logs=[]
    for seed in NEW_SEEDS:
        fv.CNN_SEED=seed;p=np.full(400,np.nan)
        for k in range(5):
            tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va])
            p[va],summary=fv.cnn_fold(images,y,tr,va,ids,k,device,out/f'cnn-seed{seed}-fold{k}.pt');logs.append({'seed':seed,'fold':k,**summary})
        predictions.append(p);print(json.dumps({'seed':seed,'rmse':fv.rmse(y,p)}),flush=True)
    mean=np.mean(predictions,axis=0);new=.5*baseline+.25*kernel+.25*mean
    values={'candidate003':current,'cnn_seed20261003':oldcnn,**{f'cnn_seed{s}':p for s,p in zip(NEW_SEEDS,predictions[1:])},'cnn_mean3':mean,'candidate_mean3':new}
    scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in values.items()}
    with (out/'development_oof.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*values]);w.writerows([s,y[i],*[p[i] for p in values.values()]] for i,s in enumerate(ids))
    for p,h in protected.items():assert fv.sha(_repo_path(ROOT, p))==h
    result={**plan,'scores':scores,'seed_prediction_correlation':np.corrcoef(predictions).tolist(),'fold_improvements':[fv.rmse(y[fid==k],current[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)],'elapsed_seconds':time.perf_counter()-start,'device':device,'folds':logs,'image_sha256':hashes,'interpretation':'Same reused development set; original seed itself historically chosen. Requires frozen repartition comparison before promotion.'}
    fv.write(out/'report.json',result);print(json.dumps({'scores':scores,'seconds':result['elapsed_seconds'],'fold_improvements':result['fold_improvements']}),flush=True)
if __name__=='__main__':main()
