"""Approved original DINOv2 ViT-B14 frozen capacity comparison, development400."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,csv,time,resource,argparse,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
import dino_spatial_039 as ds
import frozen_validation as fv
import earlystop_experiment_057 as baseline
from dino_regression_037 import FrozenRegressor
from dinov2.hub.backbones import dinov2_vitb14
OUT=_repo_path(ROOT, 'artifacts/dinob-062');VENDOR=_repo_path(ROOT, 'vendor/dinov2-base-062');LAMBDAS=(.1,1.,10.)
def streamsha(path):
 h=hashlib.sha256()
 with path.open('rb') as file:
  for chunk in iter(lambda:file.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def model():
 p=json.loads((_repo_path(VENDOR, 'provenance.json')).read_text());w=_repo_path(VENDOR, 'dinov2_vitb14_pretrain.pth');assert w.stat().st_size==p['bytes'] and streamsha(w)==p['sha256'];m=dinov2_vitb14(pretrained=False);m.load_state_dict(torch.load(_repo_io(w),map_location='cpu',weights_only=True),strict=True);m.requires_grad_(False);m.eval();assert m.embed_dim==768 and m.num_register_tokens==0 and m.patch_size==14 and len(m.blocks)==12;return m

def feature(m,x):
 assert not m.training and not any(p.requires_grad for p in m.parameters())
 with torch.inference_mode():
  layers=m.get_intermediate_layers(x,n=[5,8,11],return_class_token=True,norm=True);a=[torch.cat((c,p.mean(1),p.std(1,correction=0)),1) for p,c in layers];v=torch.cat(a,1);assert v.shape==(len(x),6912) and torch.isfinite(v).all();return v

def prepare():
 OUT.mkdir(exist_ok=False);p={'hypothesis':'Largerofficialpretrainedbackbone may offer complementary morphology/texture information beyondsmallViT-S,notmerelychangeheadregularization. Noqualityguarantee.','weights':json.loads((_repo_path(VENDOR, 'provenance.json')).read_text()),'rules_check':{'url':'https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai','checked_utc':'2026-10-02','summary':'Officialpublicpretrainedweights withpermitteduse/modification/redistribution licenseallowed; localmodels only; noexternaltrainingdata/testtraining. OriginalDINOv2Apache2.0 confirmed. NoCellDINO/Xray/registervariant.'},'features':'FrozenViT-B14 full504bicubicgrayscaleRGB/ImageNetnorm;blocks6/9/12 normalizedCLS768+patchmean768+populationstd768 =6912. Noaugmentation orbackboneadaptation.','control':'SamefoldViT-S0443456features, existingmatched006Ridge. Refitmatchedcontrol andchecksame006OOF exactly. Primarycomparisonis007 bestMLPensemble.','regression':'StandardScalerX+standardtargetRidge alpha=lambda*p where pfeaturecount;lambda.1/1/10 innerGroupKFold3shuffle seed20261120+outerfold, pooledinnerRMSE. Trainallpreprocessing onlyinnertrain/outertrain.','primary':'.75samefold003+.25B_Ridge versus007; matchedrepresentationdiagnostic006=.75same003+.25S_Ridge. OnlyBcandidatecanadvance; no weights/grid/layer search.','screen':'Original Bblend improves007 withgain>pairedbootstrapSD and>=3/5folds; then bothknownresplits unchangedrecipe. Gateallocatescompute,notsignificance.','scope':'Development400 only duringselection; userseparatelyauthorizedfull500/testinferenceonlyifimprovementvalidated. No testfit/holdoutscore. NoCSV/uploadinthisscript;best007preserved.','resource':'CPU/MPSlocalexistinguvpackages;benchmarkCPU/MPS3rows,projected400<300s,RSS<3GB,MPSdriver<3GB beforeextract. No paidservice/newpackage.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['dinob_062.py','dino_regression_037.py','dino_spatial_039.py','dino_frozen_037.py']},'input_hashes':{n:fv.sha(_repo_path(ROOT, n)) for n in ['artifacts/dino-multilayer-044/development_features.npz','artifacts/run-012/development_features.npz']},'protected':json.loads((_repo_path(ROOT, 'artifacts/token-cov-056/plan.json')).read_text())['protected'],'bias':fv.CAVEAT};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'weight_sha256':p['weights']['sha256']}),flush=True)
def check():
 p=json.loads((_repo_path(OUT, 'plan.json')).read_text())
 for n,h in {**p['sources'],**p['input_hashes'],**p['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 assert streamsha(_repo_path(VENDOR, 'dinov2_vitb14_pretrain.pth'))==p['weights']['sha256'];official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
 for n,h in official['official_source_sha256'].items():assert fv.sha(ds.base.SOURCE/n)==h,n
 return p

def benchmark():
 check();torch.set_num_threads(2);ids,_=ds.base.scope();ids=ids[:3];m=model();x=torch.from_numpy(ds.inputs(ids[0],'full504'));cpu=feature(m,x).numpy()[0];m=m.to('mps');values=[];t=time.perf_counter()
 for ident in ids:values.append(feature(m,torch.from_numpy(ds.inputs(ident,'full504')).to('mps')).cpu().numpy()[0])
 elapsed=time.perf_counter()-t;X=np.array(values);np.testing.assert_allclose(cpu,X[0],rtol=2e-3,atol=2e-3);rp=feature(m,x.to('mps')).cpu().numpy()[0];np.testing.assert_array_equal(rp,X[0]);assert all(p.grad is None for p in m.parameters())
 with (_repo_path(OUT, 'benchmark_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X)
 report={'strict_weight_load':True,'parameters':sum(p.numel() for p in m.parameters()),'feature_width':6912,'seconds3':elapsed,'projected400_seconds':elapsed*400/3,'cpu_mps_max_difference':float(abs(cpu-X[0]).max()),'mps_repeat_error':float(abs(rp-X[0]).max()),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};report['resource_pass']=bool(report['projected400_seconds']<300 and report['peak_rss_bytes']<3e9 and report['mps_driver_bytes']<3e9);fv.write(_repo_path(OUT, 'benchmark.json'),report);print(json.dumps(report),flush=True)
def extract():
 check();assert json.loads((_repo_path(OUT, 'benchmark.json')).read_text())['resource_pass'];torch.set_num_threads(2);ids,_=ds.base.scope();m=model().to('mps');before={n:p.detach().cpu().clone() for n,p in m.state_dict().items()};A=[];t=time.perf_counter()
 for j,ident in enumerate(ids):
  A.append(feature(m,torch.from_numpy(ds.inputs(ident,'full504')).to('mps')).cpu().numpy()[0])
  if (j+1)%80==0:print(json.dumps({'features_done':j+1,'seconds':time.perf_counter()-t}),flush=True)
 X=np.array(A);assert X.shape==(400,6912) and np.isfinite(X).all();assert all(torch.equal(before[n],p.detach().cpu()) for n,p in m.state_dict().items()) and all(p.grad is None for p in m.parameters())
 with np.load(_repo_path(OUT, 'benchmark_features.npz')) as z:assert z['ids'].tolist()==ids[:3];np.testing.assert_array_equal(z['X'],X[:3])
 with (_repo_path(OUT, 'development_features.npz')).open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),X=X)
 report={'shape':list(X.shape),'seconds':time.perf_counter()-t,'feature_sha256':fv.sha(_repo_path(OUT, 'development_features.npz')),'image_sha256':{i:fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png')) for i in ids},'weights_unchanged':True,'benchmark_repeat_exact':True,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_driver_bytes':torch.mps.driver_allocated_memory()};fv.write(_repo_path(OUT, 'extraction.json'),report);check();print(json.dumps({k:v for k,v in report.items() if k!='image_sha256'}),flush=True)
def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'selection.json')).read_text())['advance']
 ids,S,y,g,fid,r003,r007,_=baseline.inputs(split);ex=json.loads((_repo_path(OUT, 'extraction.json')).read_text());assert fv.sha(_repo_path(OUT, 'development_features.npz'))==ex['feature_sha256']
 with np.load(_repo_path(OUT, 'development_features.npz')) as z:assert z['ids'].tolist()==ids;B=z['X'].astype(float)
 src=_repo_path(ROOT, 'artifacts/dino-multilayer-044', split, 'development_oof.csv')
 with src.open() as file:rows=list(csv.DictReader(file))
 r006=np.array([float(r['multi3_blend']) for r in rows]);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);ps={};logs=[];t=time.perf_counter()
 for name,A in [('controlS',S),('baseB',B)]:
  pred=np.full(400,np.nan)
  for k in range(5):
   tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(g[tr])&set(g[va]);inner=list(GroupKFold(n_splits=3,shuffle=True,random_state=20261120+k).split(A[tr],y[tr],g[tr]));scores=[]
   for lam in LAMBDAS:
    ip=np.full(len(tr),np.nan)
    for it,iv in inner:
     assert not set(g[tr[it]])&set(g[tr[iv]]);m=FrozenRegressor('all_ridge',lam*A.shape[1]).fit(A[tr[it]],y[tr[it]]);ip[iv]=m.predict(A[tr[iv]])
    scores.append({'lambda':lam,'rmse':fv.rmse(y[tr],ip)})
   chosen=min(scores,key=lambda a:a['rmse'])['lambda'];m=FrozenRegressor('all_ridge',chosen*A.shape[1]).fit(A[tr],y[tr]);pred[va]=m.predict(A[va]);path=out/f'{name}-fold-{k}.joblib';joblib.dump({'model':m,'train_ids':[ids[j] for j in tr],'validation_ids':[ids[j] for j in va]},_repo_io(path));logs.append({'arm':name,'fold':k,'chosen_lambda':chosen,'inner_scores':scores,'inner_splits':[[tr[a].tolist(),tr[b].tolist()] for a,b in inner],'train_rmse':fv.rmse(y[tr],m.predict(A[tr])),'validation_rmse':fv.rmse(y[va],pred[va]),'model_sha256':fv.sha(path)})
  ps[name]=pred;ps[name+'_blend']=.75*r003+.25*pred
 np.testing.assert_allclose(ps['controlS_blend'],r006,atol=1e-10,rtol=0);ps.update(reference007=r007,reference006=r006,reference003=r003);scores={n:{'rmse':fv.rmse(y,v),'fold_rmse':[fv.rmse(y[fid==k],v[fid==k]) for k in range(5)]} for n,v in ps.items()};new=ps['baseB_blend'];gain=fv.rmse(y,r007)-fv.rmse(y,new);fg=[fv.rmse(y[fid==k],r007[fid==k])-fv.rmse(y[fid==k],new[fid==k]) for k in range(5)];ix=np.random.default_rng(20270103).integers(0,400,(3000,400));boot=np.sqrt(np.mean((y-r007)[ix]**2,1))-np.sqrt(np.mean((y-new)[ix]**2,1));comp={'gain_vs007':gain,'gain_vs006':fv.rmse(y,r006)-fv.rmse(y,new),'fold_gains_vs007':fg,'positive_folds_vs007':sum(a>0 for a in fg),'bootstrap_sd':float(boot.std()),'conditional_ci95':np.percentile(boot,[2.5,97.5]).tolist(),'B_S_ridge_residual_correlation':float(np.corrcoef(y-ps['baseB'],y-ps['controlS'])[0,1])}
 with (out/'development_predictions.npz').open('xb') as file:np.savez_compressed(_repo_io(file),ids=np.array(ids),y=y,fold=fid,**ps)
 report={'split':split,'scores':scores,'comparison':comp,'folds':logs,'seconds':time.perf_counter()-t,'matched006_max_error':float(abs(ps['controlS_blend']-r006).max()),'predictions_sha256':fv.sha(out/'development_predictions.npz'),'new_candidate':None};fv.write(out/'report.json',report)
 if split=='original':fv.write(_repo_path(OUT, 'selection.json'),{'advance':bool(gain>boot.std() and sum(a>0 for a in fg)>=3),'primary':'baseB_blend','comparison':comp})
 check();print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','benchmark','extract','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'benchmark':benchmark,'extract':extract}.get(a.mode,lambda:run(a.mode))()
