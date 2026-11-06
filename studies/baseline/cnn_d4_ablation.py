"""Prespecified inference-invariance comparison; saved development-fold models only."""
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
from cnn_ensemble_inference import predict_views

def read(p):
    with p.open() as f:r=list(csv.DictReader(f))
    assert len(r)==400 and len({v['ID'] for v in r})==400
    return {v['ID']:v for v in r}
def main():
    out=_repo_path(ROOT, 'artifacts/run-020');out.mkdir(exist_ok=False);start=time.perf_counter()
    a=read(_repo_path(ROOT, 'artifacts/run-018/development_oof.csv'));b=read(_repo_path(ROOT, 'artifacts/run-012/development_oof.csv'));ids=sorted(a)
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());assert set(ids)==set(b)==set(sp['development_folds'])
    y=np.array([float(a[s]['hardness']) for s in ids]);np.testing.assert_array_equal(y,[float(b[s]['hardness']) for s in ids]);fid=np.array([sp['development_folds'][s] for s in ids])
    fv.write(out/'plan.json',{'hypothesis':'D4 training augmentation does not ensure invariant single-view inference; average all8 fixed D4 views to reduce orientation variation','seeds':[20261003,20261023,20261043],'weights':{'baseline':.5,'cnn':.25,'kernel':.25},'training':False,'scope':'Only original development400 validation-fold inference; checkpoints verify train IDs exclude scored IDs','source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in ['cnn_d4_ablation.py','cnn_ensemble_inference.py']}})
    images,hashes=fv.load_development_images(ids);torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';means=[];diagnostics=[]
    for seed in (20261003,20261023,20261043):
        p=np.full(400,np.nan)
        for k in range(5):
            va=np.flatnonzero(fid==k);tr=np.flatnonzero(fid!=k)
            ck=_repo_path(ROOT, 'artifacts', 'run-009' if seed==20261003 else 'run-018', f'fold-{k}.pt' if seed==20261003 else f'cnn-seed{seed}-fold{k}.pt')
            meta=torch.load(_repo_io(ck),map_location='cpu',weights_only=True);assert set(meta['train_ids'])=={ids[i] for i in tr} and not set(meta['train_ids'])&{ids[i] for i in va}
            views=predict_views(images[va],ck,device);expected=np.array([float(a[ids[i]][f'cnn_seed{seed}']) for i in va]);np.testing.assert_allclose(views[0],expected,rtol=0,atol=1e-3)
            p[va]=views.mean(axis=0);diagnostics.append({'seed':seed,'fold':k,'mean_view_prediction_std':float(views.std(axis=0).mean()),'checkpoint_sha256':fv.sha(ck),'identity_reload_max_difference':float(np.max(abs(views[0]-expected)))})
        means.append(p);print(json.dumps({'seed':seed,'d4_rmse':fv.rmse(y,p)}),flush=True)
    base=np.array([float(b[s]['baseline']) for s in ids]);kernel=np.array([float(b[s]['nested_kernel']) for s in ids]);cnn=np.mean(means,axis=0)
    values={'candidate003':np.array([float(a[s]['candidate003']) for s in ids]),'candidate_mean3':np.array([float(a[s]['candidate_mean3']) for s in ids]),'cnn_mean3_d4':cnn,'candidate_mean3_d4':.5*base+.25*kernel+.25*cnn}
    scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in values.items()}
    with (out/'development_oof.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness',*values]);w.writerows([s,y[i],*[p[i] for p in values.values()]] for i,s in enumerate(ids))
    report={'scores':scores,'diagnostics':diagnostics,'elapsed_seconds':time.perf_counter()-start,'device':device,'test_or_holdout_used':False,'interpretation':'Additional hypothesis on reused development; independent repartition refits required for promotion.'}
    fv.write(out/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
