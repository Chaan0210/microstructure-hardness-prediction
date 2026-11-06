"""One matched head-width test:32 to64, retain011 short regularized recipe."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib,datetime,argparse,inspect
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import frozen_seed_088 as old
import dropout_control_089 as previous
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
e=old.e;fv=old.fv;OUT=_repo_path(ROOT, 'artifacts/width-control-090');CFG={**e.CFG,'hidden':64};SEED=20270215;statistics=previous.statistics
class WidthHead(torch.nn.Module):
 def __init__(self,width):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(6912,width),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(width,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def source_control():
 a=inspect.getsource(old.fit).replace('def fit(d,m,k,seed,path,full=False):','def fit(d,m,k,seed,path,full=False,width=64):').replace('path.mkdir(parents=True,exist_ok=False);start=',"cfg={**CFG,'hidden':width};path.mkdir(parents=True,exist_ok=False);start=").replace("model=e.Model(m,sc,actual,False).to('mps');initial=","model=e.Model(m,sc,actual,False).to('mps');torch.manual_seed(actual);model.head=WidthHead(width).to('mps');initial=").replace("'config':CFG","'config':cfg")
 assert inspect.getsource(fit)==a

def prepare():
 old.check();previous.check();source_control();OUT.mkdir(exist_ok=False);pub=old.diag.check_best();p={'study':'090 width32 versus64 with011 short schedule','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Separate head capacity from strongdropout/shorttraining: use oldMLP width64 but retain011 12epochs/dropout.5/weightdecay.1/fixedseed. Tests whether smallwidth restricts useful nonlinear feature interactions under the retained regularization.','factor':'Hiddenwidth only32->64; sameB6912 features,train-onlyscalers/targets,seed20270215+fold,permutation/RNGseedstream,12epochcosine,AdamW,micro2/effective16,SmoothL1,clip1,.9/.1blend. No dropout changes or newseeds. Widthchanges parametercount221249->442497 and output fan-in/initial tensor shapes; identicalwhole initialization is impossible. Same initialization algorithm andseed, first32 first-layer weightrows match; controlwidth32 replay must match011.','config':CFG,'duplicate_audit':'Old063/064 width64 used100epochs/dropout.25/wd.01/batch64.084/085 usedwidth32/12epochs/dropout.5/wd.1/micro2eff16;088 variedseed and089 varieddropout only. No exactwidth-only64with011recipe found.','original_screen':'If original pooled011 gain<=0, stop without either resplit. Validate/replay the completed original even onfailure. If gain>0 run both existingresplits unchanged.','candidate_gate':'After positiveoriginalscreen, same089 materialcriterion inall3partitions:gain>=.03,>=3/5folds,conditional95%gainCI lower>0,andgainpositiveafterdropping5largestpositiveerrorcontributors; strictvalidationpass. No alternativewidth/epoch/dropout/seed/ratio.','scope_after_experiment':'Stop headhyperparameter exploration. Summarize success/failure families and information gaps; inspect prior audit evidence/provenance without rerunning fullaudits. Any nextmajorhypothesis needs specific rationale/resource plan.','sources':{n:fv.sha(_repo_path(ROOT, n)) for n in ['width_control_090.py','validate_width_control_090.py','frozen_seed_088.py','dropout_control_089.py','dinob_joint_084.py']},'evidence':{n:fv.sha(_repo_path(ROOT, n)) for n in ['reports/dropout-control-089-summary.json','reports/frozen-head-087-seed-088-summary.json','reports/frozen-complement-085-validation.json','reports/candidate011-public-submission.json']},'protected011':pub['protected_sha256'],'limits':'Same400 historicallyreused observations; correlatedresplits are notindependentconfirmation. Fixedseed conditional effect. Widthincludes unavoidable initialtensor-shape/fan-in changes. ConditionalCI omits refitting/selection uncertainty; strictoperationalgate is not proof.','resources':'Existinguv CPU2/MPS officialfrozenweights/cache.5fitsoriginal~30seconds; atmost15newfits plus2replaysperpartition. Noinstall/download/cost.','scope':'Projectonly,development400selectiononly,best011protected,010unsubmitted,noholdout/testselection,no upload.'};fv.write(_repo_path(OUT, 'plan.json'),p);print(json.dumps({'prepared':True,'factor':p['factor'],'original_screen':p['original_screen']}),flush=True)

def check():
 old.check();previous.check();source_control();p=json.loads((_repo_path(OUT, 'plan.json')).read_text());assert p['config']==CFG
 for n,h in {**p['sources'],**p['evidence']}.items():assert fv.sha(_repo_path(ROOT, n))==h,n
 return p

def fit(d,m,k,seed,path,full=False,width=64):
 cfg={**CFG,'hidden':width};path.mkdir(parents=True,exist_ok=False);start=time.perf_counter();tr=np.arange(len(d['y'])) if full else np.flatnonzero(d['fid']!=k);va=np.array([],dtype=int) if full else np.flatnonzero(d['fid']==k)
 if not full:assert not set(d['g'][tr])&set(d['g'][va])
 sc=StandardScaler().fit(d['B'][tr]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std());actual=seed+k;model=e.Model(m,sc,actual,False).to('mps');torch.manual_seed(actual);model.head=WidthHead(width).to('mps');initial=e.h.state_hash(model.state_dict());fixed=e.fixed_hash(model);opt=e.optimizer(model);orders=np.random.default_rng(actual).permuted(np.tile(np.arange(len(tr)),(12,1)),axis=1);curve=[];seeds=[]
 for ep,order in enumerate(orders):
  e.schedule(opt,ep);model.training_mode();losses=[]
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for offset in range(0,len(group),2):
    ii=group[offset:offset+2];rng=actual+10000*(ep+1)+first+offset;torch.manual_seed(rng);seeds.append(rng);p=model(e.tensor(d['B'][ii]));loss=F.smooth_l1_loss(p,e.tensor((d['y'][ii]-ym)/ys),beta=1.);assert torch.isfinite(loss);(loss*(len(ii)/len(group))).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.);assert torch.isfinite(gn);opt.step()
  tp=e.predict(model,d,None,tr)*ys+ym;curve.append({'epoch':ep+1,'online_train_loss':float(np.mean(losses)),'train_RMSE':fv.rmse(d['y'][tr],tp),'head_lr':opt.param_groups[0]['lr']})
 p=e.predict(model,d,None,va)*ys+ym if len(va) else np.array([]);assert e.fixed_hash(model)==fixed;state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'config':cfg,'base_seed':seed,'seed':actual,'target_mean':ym,'target_std':ys,'scaler_mean':torch.tensor(sc.mean_),'scaler_scale':torch.tensor(sc.scale_),'scaler_var':torch.tensor(sc.var_),'initial_state_sha256':initial,'fixed_state_sha256':fixed},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,train_predictions=tp,predictions=p)
 r={'base_seed':seed,'seed':actual,'config':cfg,'initial_state_sha256':initial,'fixed_state_sha256':fixed,'final_state_sha256':e.h.state_hash(state),'batch_order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_seeds_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'checkpoint_sha256':fv.sha(path/'model.pt'),'predictions_sha256':fv.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};fv.write(path/'report.json',r);print(json.dumps({'fold':k,'seed':seed,'full500':full,'seconds':r['seconds']}),flush=True);return p,r

def run(split):
 check()
 if split!='original':assert json.loads((_repo_path(OUT, 'original/report.json')).read_text())['comparison']['gain']>0
 torch.set_num_threads(2);d=old.inputs(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);m=e.backbone.model();pred=np.full(400,np.nan);records={};train=[];start=time.perf_counter()
 for k in range(5):
  va=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}';p,r=fit(d,m,k,SEED,path);pred[va]=p;train.append(r['curve'][-1]['train_RMSE']);records[f'fold-{k}/report.json']=fv.sha(path/'report.json');torch.mps.empty_cache()
 primary=.9*d['reference009']+.1*pred;comp,diag,advance,sq,ab,change=statistics(d,primary)
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],new_head=pred,control_head=d['basehead'],primary=primary,reference009=d['reference009'],reference011=d['reference011'],squared_error_gain=sq,absolute_error_gain=ab,prediction_change=change)
 r={'scores':{'reference011':fv.rmse(d['y'],d['reference011']),'primary':fv.rmse(d['y'],primary),'head':fv.rmse(d['y'],pred),'control_head':fv.rmse(d['y'],d['basehead'])},'comparison':comp,'sample_diagnostics':diag,'advance':advance,'pooled_train_RMSE':float(np.sqrt(np.mean(np.array(train)**2))),'fold_report_sha256':records,'predictions_sha256':fv.sha(out/'development_predictions.npz'),'seconds':time.perf_counter()-start};fv.write(out/'report.json',r);check();print(json.dumps({'split':split,**r}),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*old.s.SPLITS]);a=ap.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
