"""One exploratory train-only nested affine fusion study after selection-rule audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,math,argparse,resource,datetime,importlib.metadata
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import patch_attention_092 as pa
import patch_replication_093 as pr
import frozen_seed_088 as old
import nested_calibration_074 as cal
OUT=_repo_path(ROOT, 'artifacts/nested-fusion-112');SPLITS=('original','20261031','20261101');ARMS={'mean':1,'attention':2};COLS=['reference011','mean','attention'];FIXED5=['TRAIN_000438','TRAIN_000331','TRAIN_000227','TRAIN_000071','TRAIN_000168']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def check():
 p=read(_repo_path(OUT, 'plan.json'))
 for n,h in {**p['sources'],**p['inputs'],**p['protected011']}.items():assert sha(_repo_path(ROOT, n))==h,n
 return p

def audit():
 result={};source={}
 for split in SPLITS:
  path=pr.result_path(split)/'development_predictions.npz';source[str(path.relative_to(ROOT))]=sha(path)
  with np.load(_repo_io(path),allow_pickle=False) as z:
   y=z['y'];ref=z['reference011'];branch=z['attention'];ids=z['ids'].tolist();r=y-ref;d=branch-ref;A=float(np.mean(r*d));B=float(np.mean(d*d));delta=-.2*A+.01*B
   np.testing.assert_allclose(delta,np.mean((y-(ref+.1*d))**2)-np.mean(r*r),atol=1e-12)
   result[split]={'attention':{'reference011':rmse(y,ref),'branch_RMSE':rmse(y,branch),'mean_control_RMSE':rmse(y,z['mean']),'mean_residual_times_delta':A,'mean_delta_squared':B,'MSE_change_at_w01':delta,'positive_gain_weight_interval_intersect_0_1':[0.,min(1.,2*A/B)] if A>0 else None,'stationary_weight_diagnostic_only':float(np.clip(A/B,0,1)),'no_optimized_RMSE_reported':True}}
  if split=='original':
   file=_repo_path(ROOT, 'artifacts/frame-features-101/original/development_predictions.npz');source[str(file.relative_to(ROOT))]=sha(file)
   with np.load(_repo_io(file),allow_pickle=False) as z:
    assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['y'],y);np.testing.assert_array_equal(z['reference011'],ref);d=z['control57']-ref;A=float(np.mean(r*d));B=float(np.mean(d*d));delta=-.2*A+.01*B
    result[split]['independent_physics57']={'branch_RMSE':rmse(y,z['control57']),'mean_residual_times_delta':A,'mean_delta_squared':B,'MSE_change_at_w01':delta,'positive_gain_weight_interval_intersect_0_1':[0.,min(1.,2*A/B)] if A>0 else None,'stationary_weight_diagnostic_only':float(np.clip(A/B,0,1)),'no_optimized_RMSE_reported':True,'scope':'One existing original prediction vector; no physics weight scan or new fitting.'}
 curves=[]
 for split in SPLITS:
  for k in range(5):
   for arm in ['mean','attention']:
    path=pr.result_path(split)/f'fold-{k}'/arm/'report.json';r=read(path);c=r['curve'];curves.append({'split':split,'fold':k,'arm':arm,'first_train_online_loss':c[0]['train_loss_online'],'last_train_online_loss':c[-1]['train_loss_online'],'last5_train_online_loss_mean':float(np.mean([v['train_loss_online'] for v in c[-5:]])),'last_lr':c[-1]['lr'],'train_RMSE':r['train_RMSE']});source[str(path.relative_to(ROOT))]=sha(path)
 early=read(_repo_path(ROOT, 'artifacts/earlystop-057/original/report.json'))
 return {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'limited_candidates':['092 coordinate-aware attention','independent compact26+TV31 physics57'],'blend_quadratic':result,'learning_curves':curves,'convergence_assessment':'092 online training loss falls but includes dropout and a decaying learning rate; training fit remains much better than heldout. This does not establish that longer training improves generalization.057 already inner-selected5/15/40/100 epochs and selected100/100/100/5/40, with worse007 ensemble. Do not rerun duration search merely because loss decreases.','057_selected_epochs':early['selected_epochs'],'unverified_axis':'Physics target regression is tested; properly crossfit residual learning against full011 remains untested in this audit. Naively learning residuals from existingglobalOOF within an outerfold leaks outer labels through its source fits. No claim that all residual methods failed.','gate_audit':'092 original fold-count rule failed despite mean superiority.093 fixed10% attention improves pooled RMSE in all3 partitions but original-selected5 IDs dominate; fixedremaining395 loses in2resplits. These are fragility diagnostics, not proof of no useful representation. Removing biggest gains, requiring everyfold win, or positive conditionalCI is not a universally valid necessity. Same400 repeatedselection means all this evidence is exploratory.','selection_bias':'Current study is explicitly selected after seeing092/093 and many other results. Nested fusion protects present coefficient fitting, not historical candidate selection. Three repartitions reuse same400 with highly correlated residuals and are not three independent studies. No Public-based tuning.','selected_axis':'Strict nested convex combination ofincumbent011 withattention, withmatchedmeancontrol andalpha0. Samearchitecture/epochs; calibrationwithin011 is itself nested again.','source_sha256':source}

def prepare():
 OUT.mkdir(exist_ok=False);a=audit();write(_repo_path(OUT, 'selection-audit.json'),a)
 old.diag.check_best();cal.check();pr.check();v=read(_repo_path(ROOT, 'reports/nested-calibration-074-validation.json'));assert v['passed']
 inputs={}
 for split in SPLITS:
  for base in [cal.OUT/split,pr.result_path(split)]:
   r=read(base/'report.json');assert sha(base/'development_predictions.npz')==r['predictions_sha256'];inputs[str((base/'report.json').relative_to(ROOT))]=sha(base/'report.json');inputs[str((base/'development_predictions.npz').relative_to(ROOT))]=sha(base/'development_predictions.npz')
  for k in range(5):
   f=cal.OUT/split/f'fold-{k}';fr=read(f/'report.json');assert sha(f/'calibration.npz')==fr['calibration_sha256'];inputs[str((f/'calibration.npz').relative_to(ROOT))]=sha(f/'calibration.npz')
   for j in range(3):
    cell=f/f'inner-{j}';r=read(cell/'report.json');assert sha(cell/'report.json')==fr['inner_report_sha256'][f'inner-{j}/report.json'];assert sha(cell/'predictions.npz')==r['files_sha256']['predictions.npz']
    for file in ['report.json','predictions.npz']:inputs[str((cell/file).relative_to(ROOT))]=sha(cell/file)
 sources={n:sha(_repo_path(ROOT, n)) for n in ['nested_fusion_112.py','validate_nested_fusion_112.py','patch_attention_092.py','patch_replication_093.py','frozen_seed_088.py','dinob_joint_084.py','nested_calibration_074.py']}
 p={'study':'112 strict nested011 attention mixture','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_sha256':sha(_repo_path(OUT, 'selection-audit.json')),
 'single_primary':'Fit onealpha in[0,1] minimizing untrimmed squared error of innerOOF011+alpha*(attention-innerOOF011), separately insideeachoutertrain. Applytoouter011/attention. alpha=0 isincumbent exactly. alpha=clip(mean((y-p011)*(pbranch-p011))/mean((pbranch-p011)^2),0,1); zero denominator->0. No additionalcalibration orweightgrid.',
 'controls':'Exactlyunchangedincumbent011, attention mixture, matchedparametermeanpool mixture withowntrain-onlyalpha. Standaloneattention/mean remainreported. No alternatecandidate selected fromcontrols.',
 'strict_nesting':'Useverified074008inner models trained213/214 onlyinsideouter320. For EACH innertraining213/214, generatefresh3fold008 OOF onthatset (~142train), withallupstreambaseline/KRR/CNN/SMLP/BMLP refittedandallscalers/kernelselectiontraining-only. Fit009affinecalibrator onthese213/214OOF pairs. Applytoverified008innerheldout predictions; combineshortB trainedonlysame213/214 withfixed.9/.1 toproduceinner011. Fitattention/mean onlysame213/214. Outer80 labels neverenteranynestedfit. Global400OOF predictionsareONLYoutervalidationreferences,nevermetatraining.',
 'seeds':'Reuse074middlelevelbase seeds;new deepestfit uses074fit_cell withk=outerk+100*(innerj+1),j=deepj. GroupKFold3 seed20270110+outerk+100*(innerj+1) fordeepcalibration. NewshortB/pool seed=base+outerk+100*(innerj+1). Outer092/093/084/085fitsreusedexactly.',
 'fixed_models':{'patch':pa.CFG,'shortB':old.e.CFG},'convergence':'057 nestedepochselectionalreadytested;092lossdeclinewithdropoutanddecayingLRdoesnotestablishinsufficientgeneralizationoptimization. No learningrate/checkpointsearch inthiscomparison.',
 'protocol':'Runall3existingpartitions asonefixedstudy withoutfirstsplitperformance stopping. PrimarypooleduntrimmedMSE/RMSE. Folds,conditionalCI,fixed092top5exclusion,andweightstabilitydiagnosticonly. CandidateeligibleifattentionmixturehasstrictlylowerpooledMSEthan011ineachpartitionandindependentvalidationpasses. No allfoldwins,.03cutoff,positiveCIortrimminggate. Nofallbackmeancandidate.',
 'resource':'135newdeep008fits(135CNN+810MLP+baseline/KRR) plus135innerheads(45shortB+90pool);alreadyverifiedmiddle008fitsreused. Firstcompletecellbenchmarkreused;project45cells<=3600s,RSS<5GB,MPS<2GB. Estimate40–60minincludingvalidation;CPU2/MPS,existingpackages,cost0. Stoponresourcefailure,no paidalternative.',
 'if_pass':'Lockrecipe andchoosefinalalphaonlyfromoriginaldevelopment400OOF011/attention. Full500attentiontraining then1000testinference;retainfixed011production. Noadditionalholdoutscore/testselection. ID/order/finite/reload/CSVhashverification;uploadcoordinatedbyparent.',
 'if_fail':'Preservebest011andfailedprocedure. Reportremainingunverifiedaxesandnextconcreteexecution/informationblocker;notclaimallattentionorallresidualmethodsfailed.',
 'sources':sources,'inputs':inputs,'protected011':read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))['protected_sha256'],'environment':{d.metadata['Name'].lower().replace('_','-'):d.version for d in importlib.metadata.distributions()},'scope':'Development400onlybeforelock;noinstall/payment/upload;failedoriginalsimmutable.','bias':a['selection_bias'],
 'methodology_sources':[{'url':'https://www.jmlr.org/beta/papers/v11/cawley10a.html','accessed':'2026-10-03','claim':'Optimizing a noisy model-selection criterion can itself overfit; nestedcurrentfitting doesnotundohistoricselection.'},{'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873/','accessed':'2026-10-03','claim':'All training and tuning steps must beinside CV loops when estimating the complete procedure.'}]}

 write(_repo_path(OUT, 'plan.json'),p);emit({'prepared':True,'diagnostic':a['blend_quadratic'],'plan_sha256':sha(_repo_path(OUT, 'plan.json'))})

class Small(nn.Module):
 def __init__(self,mean,scale,seed):
  super().__init__();torch.manual_seed(seed);self.head=old.e.Head();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
 def forward(self,x):return self.head((x-self.mean)/self.scale)
def small_predict(m,B,ix,device='mps'):
 m.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(ix),2):out.extend(m(torch.tensor(B[ix[j:j+2]],dtype=torch.float32,device=device)).cpu().numpy().astype(float))
 return np.array(out)
def fit(d,tokens,mom,tr,va,k,j,arm,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();offset=k+100*(j+1);patch=arm!='shortB';seed=(pa.CFG['seed'] if patch else old.e.CFG['seed'])+offset;epochs=20 if patch else 12
 assert not set(tr)&set(va) and not set(d['g'][tr])&set(d['g'][va]);ym=float(d['y'][tr].mean());ys=float(d['y'][tr].std())
 if patch:mean,scale,var=pa.normalize(mom,tr);model=pa.Pool(mean,scale,seed,arm).to('mps')
 else:sc=StandardScaler().fit(d['B'][tr]);mean,scale,var=sc.mean_,sc.scale_,sc.var_;model=Small(mean,scale,seed).to('mps')
 initial=old.e.h.state_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.001 if patch else .0003,weight_decay=.1);orders=np.random.default_rng(seed).permuted(np.tile(np.arange(len(tr)),(epochs,1)),axis=1);curve=[];seeds=[]
 for ep,order in enumerate(orders):
  lr=1e-5+.5*((.001 if patch else .0003)-1e-5)*(1+math.cos(math.pi*ep/epochs));opt.param_groups[0]['lr']=lr;model.train();losses=[]
  for first in range(0,len(tr),16):
   group=tr[order[first:first+16]];opt.zero_grad(set_to_none=True)
   for off in range(0,len(group),2):
    ii=group[off:off+2];rng=seed+10000*(ep+1)+first+off;torch.manual_seed(rng);seeds.append(rng);x=pa.tensor(tokens,ii) if patch else torch.tensor(d['B'][ii],dtype=torch.float32,device='mps');yy=torch.tensor((d['y'][ii]-ym)/ys,dtype=torch.float32,device='mps');loss=F.smooth_l1_loss(model(x),yy,beta=1.);assert torch.isfinite(loss);(loss*len(ii)/len(group)).backward();losses.append(float(loss.detach()))
   gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gn);opt.step()
  curve.append({'epoch':ep+1,'online_loss':float(np.mean(losses)),'lr':lr})
 pred=pa.predict(model,tokens,va)[0] if patch else small_predict(model,d['B'],va);pred=pred*ys+ym
 state={n:v.detach().cpu() for n,v in model.state_dict().items()};torch.save({'state':state,'arm':arm,'seed':seed,'mean':torch.tensor(mean),'scale':torch.tensor(scale),'var':torch.tensor(var),'target_mean':ym,'target_std':ys,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'train_ids':[d['ids'][i] for i in tr],'validation_ids':[d['ids'][i] for i in va],'epochs':epochs},_repo_io(path/'model.pt'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,predictions=pred)
 r={'arm':arm,'seed':seed,'initial_sha256':initial,'state_sha256':old.e.h.state_hash(state),'order_sha256':hashlib.sha256(orders.tobytes()).hexdigest(),'dropout_sha256':hashlib.sha256(np.array(seeds,dtype=np.int64).tobytes()).hexdigest(),'curve':curve,'model_sha256':sha(path/'model.pt'),'predictions_sha256':sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};write(path/'report.json',r);del model,opt;torch.mps.empty_cache();return pred,r

def meta_fit(P,y):
 ref,branch=P[:,0],P[:,1];delta=branch-ref;r=y-ref;A=float(np.mean(r*delta));B=float(np.mean(delta*delta));alpha=float(np.clip(A/B,0,1)) if B>0 else 0.
 return {'alpha':alpha,'mean_residual_times_delta':A,'mean_delta_squared':B,'objective_MSE':float(np.mean((r-alpha*delta)**2)),'incumbent_MSE':float(np.mean(r*r))}
def meta_predict(m,P):return P[:,0]+float(m['alpha'])*(P[:,1]-P[:,0])

def data(split):
 d=old.inputs(split);extra=cal.inputs(split)
 for key in ['S','Xbase','images','K','blocks','recipes','weights']:d[key]=extra[key]
 with np.load(_repo_io(cal.OUT/split/'development_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);d['reference008']=z['reference008'].copy();np.testing.assert_allclose(z['primary'],d['reference009'],atol=1e-10,rtol=0)
 with np.load(_repo_io(pr.result_path(split)/'development_predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['fold'],d['fid']);d['outerP']=np.column_stack([d['reference011'],z['mean'],z['attention']])
 return d

def inner_cell(d,tokens,mom,it,iv,outerva,k,j,source,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();offset=k+100*(j+1);nested=np.full(len(it),np.nan);deeprecords=[]
 deep=list(GroupKFold(3,shuffle=True,random_state=cal.INNER_SEED+offset).split(d['S'][it],d['y'][it],d['g'][it]))
 for ell,(a,b) in enumerate(deep):
  excluded=np.setdiff1d(np.arange(400),it)
  pred,rec=cal.fit_cell(d,it[a],it[b],excluded,offset,ell,path/f'deep-{ell}')
  nested[b]=pred;deeprecords.append({'deep':ell,'report_sha256':sha(path/f'deep-{ell}'/'report.json')})
 assert np.isfinite(nested).all();center=float(nested.mean());ym=float(d['y'][it].mean());z=nested-center;slope=float(z@(d['y'][it]-ym)/(z@z))
 with np.load(_repo_io(source/'predictions.npz'),allow_pickle=False) as src:np.testing.assert_array_equal(src['train_indices'],it);np.testing.assert_array_equal(src['validation_indices'],iv);p008=src['ensemble008'].copy()
 p009=ym+slope*(p008-center);models={};predictions={}
 for arm in ['shortB','mean','attention']:predictions[arm],models[arm]=fit(d,tokens,mom,it,iv,k,j,arm,path/arm)
 for key in ['initial_sha256','order_sha256','dropout_sha256']:assert models['mean'][key]==models['attention'][key]
 p011=.9*p009+.1*predictions['shortB']
 with (path/'calibration.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=it,validation_indices=iv,inner_oof008=nested,p008=p008,p009=p009,p011=p011,**predictions)
 r={'outer':k,'inner':j,'calibrator':{'prediction_center':center,'target_center':ym,'slope':slope},'deep':deeprecords,'calibration_sha256':sha(path/'calibration.npz'),'model_reports':{a:sha(path/a/'report.json') for a in models},'seconds':time.perf_counter()-start};write(path/'report.json',r)
 return np.column_stack([p011,predictions['mean'],predictions['attention']]),r

def run(split):
 check();torch.set_num_threads(2);d=data(split);tokens,mom=pa.get_cache(d);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();ps={a:np.full(400,np.nan) for a in ARMS};folds=[]
 for k in range(5):
  fd=out/f'fold-{k}';fd.mkdir();tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);ip=np.full((len(tr),3),np.nan);lookup={int(ix):j for j,ix in enumerate(tr)};cells=[]
  for j in range(3):
   source=cal.OUT/split/f'fold-{k}'/f'inner-{j}'
   with np.load(_repo_io(source/'predictions.npz'),allow_pickle=False) as z:it=z['train_indices'];iv=z['validation_indices']
   assert set(it)|set(iv)==set(tr) and not (set(it)|set(iv))&set(va)
   predictions,rec=inner_cell(d,tokens,mom,it,iv,va,k,j,source,fd/f'inner-{j}');ip[[lookup[int(i)] for i in iv]]=predictions;cells.append(rec);emit({'split':split,'outer':k,'inner':j,'cell_seconds':rec['seconds'],'elapsed':time.perf_counter()-start})
   if split=='original' and k==0 and j==0:
    bench={'first_cell_reused':True,'seconds':rec['seconds'],'projected45cells_seconds':45*rec['seconds'],'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'mps_bytes':torch.mps.driver_allocated_memory()};bench['resource_pass']=bench['projected45cells_seconds']<=3600 and bench['rss_bytes']<5e9 and bench['mps_bytes']<2e9;write(_repo_path(OUT, 'benchmark.json'),bench);assert bench['resource_pass']
  assert np.isfinite(ip).all();metas={}
  for arm,col in ARMS.items():metas[arm]=meta_fit(ip[:,[0,col]],d['y'][tr]);ps[arm][va]=meta_predict(metas[arm],d['outerP'][va][:,[0,col]])
  with (fd/'meta.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train_indices=tr,validation_indices=va,innerP=ip,outerP=d['outerP'][va],**{a:ps[a][va] for a in ARMS})
  r={'fold':k,'metas':metas,'cell_report_sha256':{f'inner-{j}/report.json':sha(fd/f'inner-{j}'/'report.json') for j in range(3)},'meta_sha256':sha(fd/'meta.npz')};write(fd/'report.json',r);folds.append(r);emit({'split':split,'outer_complete':k,'alphas':{a:m['alpha'] for a,m in metas.items()}})
 ps['reference011']=d['reference011'];scores={a:{'mse':float(np.mean((d['y']-p)**2)),'rmse':rmse(d['y'],p),'fold_rmse':[rmse(d['y'][d['fid']==k],p[d['fid']==k]) for k in range(5)]} for a,p in ps.items()};comp={a:{'MSE_gain':scores[a]['mse']-scores['attention']['mse'],'RMSE_gain':scores[a]['rmse']-scores['attention']['rmse'],'fold_RMSE_gains':(np.array(scores[a]['fold_rmse'])-np.array(scores['attention']['fold_rmse'])).tolist()} for a in ['reference011','mean']}
 keep=np.array([i not in FIXED5 for i in d['ids']]);ix=np.random.default_rng(20271012).integers(0,400,(3000,400));rr=(d['y']-ps['reference011'])**2;nn=(d['y']-ps['attention'])**2;ci=np.quantile(np.sqrt(rr[ix].mean(1))-np.sqrt(nn[ix].mean(1)),[.025,.975]).tolist()
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],outerP=d['outerP'],**ps)
 r={'split':split,'scores':scores,'comparisons':comp,'conditional95CI':ci,'fixed092_other395_gain_diagnostic':rmse(d['y'][keep],ps['reference011'][keep])-rmse(d['y'][keep],ps['attention'][keep]),'fold_alphas':[{a:f['metas'][a]['alpha'] for a in ARMS} for f in folds],'prediction_sha256':sha(out/'predictions.npz'),'fold_report_sha256':{f'fold-{k}/report.json':sha(out/f'fold-{k}'/'report.json') for k in range(5)},'seconds':time.perf_counter()-start};write(out/'report.json',r);check();emit(r)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare',*SPLITS]);args=a.parse_args();prepare() if args.mode=='prepare' else run(args.mode)
