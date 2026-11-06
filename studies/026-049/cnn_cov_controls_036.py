"""Two predeclared controls; screen originalCV then at most one in two repartitions."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,csv,time,gc,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(_repo_path(ROOT, '.venv')).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch,joblib
from torch import nn
from sklearn.base import clone
import cnn_structure_experiment as core
import frozen_validation as fv
from cnn_cov_regularized import SmallCovarianceCNN,regularized_descriptor
NAMES=('small_trace','small_regularized')
OUT=_repo_path(ROOT, 'artifacts/cnn-cov-controls-036')
SOURCES=('cnn_cov_controls_036.py','cnn_cov_regularized.py','cnn_structure_experiment.py','cnn_texture_fusion.py','cnn_controls.py','frozen_validation.py','compact_regression.py')
def factory(name):
    if name not in NAMES:raise ValueError(name)
    return SmallCovarianceCNN(name=='small_regularized')
# Reuse the exact reviewed035 trainer, changing only the constructor in memory.
core.make_model=factory

def reference(split,ids,y):
    d=_repo_path(ROOT, 'artifacts/cnn-structure-035', split)
    with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
    assert [r['ID'] for r in rows]==ids
    np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows])
    return {n:np.array([float(r[n]) for r in rows]) for n in ('baseline','compact_kernel','tiny128','tiny128_blend','cov128','cov128_blend')}

def benchmark():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2);assert torch.backends.mps.is_available()
    torch.manual_seed(20261117);z=torch.randn(1,3,3,3,dtype=torch.double,requires_grad=True);idx=torch.triu_indices(3,3);w=torch.where(idx[0]==idx[1],1.,2**.5).double()
    assert torch.autograd.gradcheck(lambda a:regularized_descriptor(a,idx[0],idx[1],w),(z,),eps=1e-6,atol=1e-4)
    mats=[]
    for magnitude in (0.,1e-5,1.,1e3):
        a=torch.randn(2,3,3,3,dtype=torch.double)*magnitude;a.requires_grad_(True)
        val=regularized_descriptor(a,idx[0],idx[1],w);val.square().sum().backward();assert torch.isfinite(val).all() and torch.isfinite(a.grad).all()
        v=val.detach()*3**.5/w;mat=torch.zeros(2,3,3,dtype=torch.double);mat[:,idx[0],idx[1]]=v;mat=mat+mat.transpose(1,2)-torch.diag_embed(mat.diagonal(dim1=1,dim2=2));assert torch.linalg.eigvalsh(mat).min()>-1e-10;assert mat.abs().max()<=1+1e-10
    states=[]
    for name in NAMES:
        torch.manual_seed(core.SEED);m=factory(name);states.append({k:v.clone() for k,v in m.state_dict().items()});v=m(torch.zeros(2,1,128,128));(v-torch.tensor([-.3,.7])).square().mean().backward();assert torch.isfinite(v).all() and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
    ids,y,groups,Xbase,recipes,weights,panel,images,provenance=core.load_data();X=images[128];del images
    sp=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text());tr=np.array([j for j,i in enumerate(ids) if sp['development_folds'][i]!=0]);ix=tr[:32];mu=X[tr].mean();sd=X[tr].std();ym=y[tr].mean();ys=y[tr].std();results={}
    for name in NAMES:
        torch.manual_seed(core.SEED);m=factory(name).to('mps');opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);x=torch.from_numpy((X[ix]-mu)/sd).to('mps');t=torch.tensor((y[ix]-ym)/ys,dtype=torch.float32,device='mps');times=[]
        for step in range(13):
            core.engine.sync('mps');start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(m(x),t);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();core.engine.sync('mps');assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            if step>=3:times.append(time.perf_counter()-start)
        results[name]={'parameters':sum(p.numel() for p in m.parameters()),'median_step_seconds':float(np.median(times)),'memory':core.engine.memory('mps'),'finite':True};del m,opt,x,t;gc.collect();torch.mps.empty_cache()
    plan={'arms':{'small_trace':'rank6,head32; same trace/meanvariance normalization as035','small_regularized':'same rank6/head32 and identical initialization;0.5diagonal shrinkage then per-channel divide by sqrt(var+0.1meanvar+1e-4)'},'trainer':'Unmodified035 train_fold reused with explicit constructor binding','schedule':'same128BOX,seed20261003+fold,80epochs,800steps,batch32,AdamWlr.002wd.01,cosine.0001,clip5,last5weightaverage,train-onlyD4;noepochselection','blend':{'baseline':.5,'compact_kernel':.25,'cnn':.25},'screening_rule':'Eligible only if original fixed-blend RMSE improves over003 and at least3of5folds improve; among eligible choose lowest blend RMSE, ties by arm order. If none eligible stop. No arbitrary minimum RMSE gain. At most one arm evaluated on both predefined repartitions.','scope':fv.CAVEAT,'source_sha256':{n:fv.sha(_repo_path(ROOT, n)) for n in SOURCES},'feature_cache_sha256':fv.sha(_repo_path(ROOT, 'artifacts/run-012/development_features.npz')),'protected':{n:fv.sha(_repo_path(ROOT, n)) for n in ('artifacts/submission-candidate-003.csv','artifacts/final-model-017/report.json')},'attribution':'Original project implementation; no external code/weights/data.'}
    fv.write(_repo_path(OUT, 'plan.json'),plan);fv.write(_repo_path(OUT, 'input-provenance.json'),provenance);report={'variants':results,'gradcheck':True,'zero_and_extreme_input_finite':True,'regularized_matrices_psd_and_bounded':True,'arms_identical_initialization':True,'thermal':core.engine.thermal()};fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)

def run(split):
    torch.set_num_threads(2);assert torch.backends.mps.is_available();plan=json.loads((_repo_path(OUT, 'plan.json')).read_text());b=json.loads((_repo_path(OUT, 'benchmark.json')).read_text())
    for n,h in plan['source_sha256'].items():assert fv.sha(_repo_path(ROOT, n))==h
    assert fv.sha(_repo_path(ROOT, 'artifacts/run-012/development_features.npz'))==plan['feature_cache_sha256']
    for info in b['variants'].values():assert info['median_step_seconds']<.2 and info['memory']['process_peak_rss_bytes']<4_000_000_000
    names=NAMES if split=='original' else (json.loads((_repo_path(OUT, 'selection.json')).read_text())['selected'],)
    assert all(n in NAMES for n in names),'No eligible arm; no repeat training allowed'
    dest=_repo_path(OUT, split);dest.mkdir(exist_ok=False);ids,y,groups,Xbase,recipes,weights,panel,images,provenance=core.load_data();X=images[128];del images;Z=panel['X'];prior=reference(split,ids,y)
    sp=json.loads((_repo_path(ROOT, 'artifacts/cnn-structure-035', split, 'split_manifest.json')).read_text());fid=np.array([sp['development_folds'][i] for i in ids]);fv.write(dest/'split_manifest.json',sp);fv.write(dest/'plan.json',{**plan,'split':split,'trained_arms':names})
    pred={n:np.full(400,np.nan) for n in ('baseline','compact_kernel',*names,*[n+'_blend' for n in names])};logs=[];start=time.perf_counter()
    for k in range(5):
        tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert len(tr)==320 and len(va)==80 and not set(groups[tr])&set(groups[va]);d=dest/f'fold-{k}';d.mkdir()
        fitted={n:clone(m).fit(Xbase[tr],y[tr]) for n,m in recipes.items()};base=sum(weights[n]*m.predict(Xbase[va]) for n,m in fitted.items());pred['baseline'][va]=base;joblib.dump({'models':fitted,'weights':weights,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(d/'baseline.joblib'));del fitted
        kp,km,kd=fv.kernel_fold(Z,y,groups,panel['blocks'],tr,va,k);pred['compact_kernel'][va]=kp;joblib.dump({'model':km,'features':panel['names'],'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(d/'kernel.joblib'));del km
        np.testing.assert_allclose(base,prior['baseline'][va],atol=1e-10,rtol=0);np.testing.assert_allclose(kp,prior['compact_kernel'][va],atol=1e-10,rtol=0)
        log={'fold':k,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va],'kernel_selection':kd,'models':{}}
        for name in names:
            p,record=core.train_fold(name,X,Z,y,tr,va,ids,k,'mps',d);pred[name][va]=p;pred[name+'_blend'][va]=.5*base+.25*kp+.25*p;log['models'][name]=record;print(json.dumps({'split':split,'fold':k,'arm':name,'cnn_rmse':record['validation_rmse'],'blend_rmse':fv.rmse(y[va],pred[name+'_blend'][va]),'seconds':record['seconds']}),flush=True)
        log['checkpoint_sha256']={p.name:fv.sha(p) for p in d.iterdir()};fv.write(dest/f'fold-{k}-report.json',log);logs.append(log)
    pred['reference003']=prior['tiny128_blend'];pred['reference_cov035']=prior['cov128_blend'];scores={n:{'rmse':fv.rmse(y,p),'fold_rmse':[fv.rmse(y[fid==k],p[fid==k]) for k in range(5)]} for n,p in pred.items()};ix=np.random.default_rng(20261118).integers(0,400,(2000,400))
    for name in names:
        p=pred[name+'_blend'];ref=pred['reference003'];g=np.sqrt(np.mean((y-ref)[ix]**2,axis=1))-np.sqrt(np.mean((y-p)[ix]**2,axis=1));scores[name+'_blend']['comparison']={'gain':fv.rmse(y,ref)-fv.rmse(y,p),'fold_gains':[fv.rmse(y[fid==k],ref[fid==k])-fv.rmse(y[fid==k],p[fid==k]) for k in range(5)],'conditional_ci95':np.percentile(g,[2.5,97.5]).tolist()}
    with (dest/'development_oof.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness','fold',*pred]);w.writerows([i,y[j],fid[j],*[a[j] for a in pred.values()]] for j,i in enumerate(ids))
    for n,h in {**plan['source_sha256'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
    report={'scores':scores,'folds':logs,'elapsed_seconds':time.perf_counter()-start,'limits':fv.CAVEAT,'all_baseline_kernel_components_refit':True,'fixed_035_controls_reused_for_reporting_only':True};fv.write(dest/'report.json',report)
    if split=='original':
        eligible=[n for n in names if scores[n+'_blend']['comparison']['gain']>0 and sum(x>0 for x in scores[n+'_blend']['comparison']['fold_gains'])>=3];selected=min(eligible,key=lambda n:scores[n+'_blend']['rmse']) if eligible else None;fv.write(_repo_path(OUT, 'selection.json'),{'selected':selected,'eligible':eligible,'rule':plan['screening_rule']})
    print(json.dumps({'split':split,'scores':scores,'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('benchmark','original','20261031','20261101'));a=p.parse_args();benchmark() if a.mode=='benchmark' else run(a.mode)
