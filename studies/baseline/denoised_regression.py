"""Preplanned denoised grain segmentation ablation on development400 only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
if Path(sys.prefix).resolve()!=(_repo_path(ROOT, '.venv')).resolve():raise SystemExit('Use project .venv')
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
import frozen_validation as fv
from denoised_grain_features import extract_denoised

def extract_one(ident):
    p=_repo_path(ROOT, 'data/train', ident+'.png')
    with Image.open(_repo_io(p)) as im:a=np.asarray(im,dtype=np.uint8).copy()
    values,names,blocks=extract_denoised(a)
    return values,names,blocks,fv.sha(p)
def rows(p):
    with p.open() as f:return {r['ID']:r for r in csv.DictReader(f)}
def main():
    out=_repo_path(ROOT, 'artifacts/run-022');out.mkdir(exist_ok=False);start=time.perf_counter()
    ids,y,groups,_,_,_,panels,provenance=fv.read_inputs(['run-012'])
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());fid=np.array([sp['development_folds'][s] for s in ids])
    protected={p:fv.sha(_repo_path(ROOT, p)) for p in ['artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json']}
    plan={'hypothesis':'Adaptive TV denoising and gradient-minima watershed may reduce visually observed noise-driven oversegmentation; no sign assumptions','panels':['denoised grains','compact26+denoised grains'],'models':['denoised nested KRR','combined nested KRR','combined fixed ET300 minleaf3 maxfeatures0.8','combined fixed HGB200 leaves7 minleaf20 l2=10 lr0.05'],'kernel_inner_grid':{'bandwidth':[.25,1.,4.],'alpha':[.01,.1,1.]},'ensemble_weights':{'baseline':.5,'cnn':.25,'new_feature_model':.25},'cnn_choices':['original seed20261003','prespecified equal3seeds andD4 run020'],'scope':'development400 original folds only; no holdout/test access','protected':protected,'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['denoised_regression.py','denoised_grain_features.py','frozen_validation.py','compact_regression.py']}}
    fv.write(out/'plan.json',plan)
    records=joblib.Parallel(n_jobs=2)(joblib.delayed(extract_one)(s) for s in ids)
    N=np.stack([r[0] for r in records]);nn,nb=records[0][1:3];assert N.shape==(400,len(nn)) and all(r[1]==nn and r[2]==nb for r in records)
    p=panels['run-012'];X=np.column_stack([p['X'],N]);names=p['names']+nn;blocks=p['blocks']+nb
    np.savez_compressed(_repo_io(out/'development_features.npz'),X=X,ids=np.array(ids),groups=groups)
    fv.write(out/'input-provenance.json',{'old_panel':provenance['panels'],'new_image_sha256':dict(zip(ids,[r[3] for r in records]))})
    feature_seconds=time.perf_counter()-start
    predictions={n:np.full(400,np.nan) for n in ['denoised_kernel','combined_kernel','combined_et','combined_hgb']};folds=[]
    for k in range(5):
        tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k)
        assert not set(groups[tr])&set(groups[va])
        log={'fold':k}
        for name,z,b in [('denoised_kernel',N,nb),('combined_kernel',X,blocks)]:
            predictions[name][va],model,detail=fv.kernel_fold(z,y,groups,b,tr,va,k);log[name]=detail
        et=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,max_features=.8,n_jobs=2,random_state=20261024).fit(X[tr],y[tr]);predictions['combined_et'][va]=et.predict(X[va])
        hgb=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20261024).fit(X[tr],y[tr]);predictions['combined_hgb'][va]=hgb.predict(X[va]);folds.append(log)
        print(json.dumps({'fold':k,'scores':{n:fv.rmse(y[va],z[va]) for n,z in predictions.items()}}),flush=True)
    prior=rows(_repo_path(ROOT, 'artifacts/run-012/development_oof.csv'));multi=rows(_repo_path(ROOT, 'artifacts/run-020/development_oof.csv')); original_cnn=rows(_repo_path(ROOT, 'artifacts/run-018/development_oof.csv')); assert set(prior)==set(multi)==set(original_cnn)==set(ids); assert all(float(prior[s]['hardness'])==float(multi[s]['hardness'])==float(original_cnn[s]['hardness']) for s in ids)
    b=np.array([float(prior[s]['baseline']) for s in ids]);c=np.array([float(original_cnn[s]['cnn_seed20261003']) for s in ids]);m=np.array([float(multi[s]['cnn_mean3_d4']) for s in ids])
    values={'candidate003':np.array([float(prior[s]['nested_kernel_cnnblend']) for s in ids]),'candidate_multiseed_d4':np.array([float(multi[s]['candidate_mean3_d4']) for s in ids]),**predictions}
    for name,z in predictions.items():values[name+'_blend']=.5*b+.25*c+.25*z;values[name+'_multiseed_blend']=.5*b+.25*m+.25*z
    scores={n:{'rmse':fv.rmse(y,z),'fold_rmse':[fv.rmse(y[fid==k],z[fid==k]) for k in range(5)]} for n,z in values.items()}
    with (out/'development_oof.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*values]);w.writerows([s,y[i],*[z[i] for z in values.values()]] for i,s in enumerate(ids))
    for p,h in protected.items():assert fv.sha(_repo_path(ROOT, p))==h
    report={**plan,'features':names,'blocks':blocks,'new_feature_count':len(nn),'scores':scores,'feature_seconds':feature_seconds,'elapsed_seconds':time.perf_counter()-start,'folds':folds,'limits':'Reused development OOF; fixed model comparisons still cause selection optimism. No fulltrain/test fit.'}
    fv.write(out/'report.json',report);print(json.dumps({'scores':scores,'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
