"""Frozen018: KRR hyperparameter selection on all U versus a label-blind 80% of U."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,time,json,hashlib,datetime,resource
from pathlib import Path
import numpy as np,joblib,torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import quality_gate_165 as old
s=old.s;a=old.a;e=old.e;x=old.x;n=old.n;fv=x.fv;R=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(R, 'artifacts/krr-scope-174');ARMS=('all','subset80');SALT='174-fixed-group-selection:'
threadpool_limits(2);torch.set_num_threads(2)
def write(p,z):
 with p.open('x') as f:json.dump(z,f,indent=2,ensure_ascii=False,allow_nan=False)
def subset(d,U):
 groups=sorted(set(d['g'][U].tolist()),key=lambda g:hashlib.sha256((SALT+str(g)).encode()).digest());chosen=set(groups[:int(np.floor(.8*len(groups)))])
 return U[np.array([g in chosen for g in d['g'][U]])]
def load(split):
 d=s.data(split);path=s.OUT/split/'report.json';r=e.read(path);z=a.readnp(s.OUT/split/'predictions.npz',r['predictions_sha256']);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);d['reference018']=z['primary'];p=_repo_path(R, 'artifacts/run-012/development_features.npz');z=a.readnp(p);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);d['K']=z['X'];d['Kblocks']=e.read(_repo_path(R, 'artifacts/run-012/report.json'))['blocks'];assert d['K'].shape==(400,26);return d

def nodes(d,split,k):
 T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);result=[{'name':'outer','U':T,'Q':V,'offset':k,'expected':d['P'][V,1]}]
 for j in range(3):
  A,B,P,short,ref=d['cells'][k,j];result.append({'name':f'middle-{j}','U':A,'Q':B,'offset':k+100*(j+1),'expected':P[:,1]})
  for ell in range(3):
   C,D,DP=a.bundle(e.OUT/split/f'fold-{k}/inner-{j}/deep-{ell}');result.append({'name':f'deep-{j}-{ell}','U':C,'Q':D,'offset':k+100*(j+1)+100*(ell+1),'expected':DP[:,1]})
 assert len(result)==13
 return result

def prepare():
 OUT.mkdir(exist_ok=False);names=['krr_scope_174.py','validate_krr_scope_174.py','quality_gate_165.py','validate_quality_gate_165.py','bootstrap_selector_144.py','cnn_seed_mean_133.py','component_ablation_121.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py','compact_regression.py','frozen_validation.py','nested_calibration_074.py','reports/research-priority-173-review.json','reports/nested-selector-146-audit.json','reports/reproducibility-audit-135-summary.json','reports/reproduction-package-155-summary.json','artifacts/run-012/development_features.npz','artifacts/run-012/report.json','artifacts/spatial-joint-140/features.npz','artifacts/spatial-joint-140/extraction.json']
 plan={'study':174,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose_ko':'018의 CV와 최종학습에서 KRR 설정 선택 자료 비율이 다른 효과를 측정한다. 성능 향상 또는 새 제출 후보를 목적으로 하지 않는다.','arms':list(ARMS),'fixed_subset':'For each current KRR training set U, hash SALT+groupID with SHA256, select firstfloor(.8*n_uniquegroups), retainoriginalroworder. FixedglobalSALT; no labels/features/errors/stratification; no alternate subset/seed. Allgroups keptintact. Currently400uniquegroups.','subset_salt':SALT,'scope':'ApplytoEVERYouterT320,middleA213/214,deepC142/143 KRR node. Onlyselection usesS80%; finalKRR scaler/kernel/target stats/refit useALLU. all-arm usesU forbothselectionandrefit. No global400selectedhyperparameters fixedintoCV.','inner_selection':'Existing9settings bandwidth(.25,1,4) × alpha(.01,.1,1), sameorder/tie rule,3GroupKFold seed20261005+existingnodeoffset. Everyinner scaler/bounds/gamma/ymeanstd fit ONLY respectiveinnertrain. Same seed butsubsetmembership andinnerfoldsnecessarilychange.','downstream':'RetainCNNmean3/Smean3/Bmean3/shortB and585features. ReplaceKRR predictions atall13nodes perouterfold; refitouteraffine onmiddleOOF and3anchor-affines ondeepOOF; reconstruct .9affine+.1shortB anchors; refitall5residualkinds insideeachhonestBinner3fold,512bootstrapweights,all5fullBmodels. Preserve .3KRR/.3CNN/.2S/.2B andmean3residualblocks.','reuse':'Allunchangedneuralpredictions musthaveexacttrain/validationIDs andhashchecked caches; prior112/133 fullnestedtargetexclusioncertificates referenced. Nonewneuraltraining orbackboneextraction.','execution':'Runoriginal5fold pairedall/subset80 (65nodes perarm) asoneinitialstudy. ControlallmustreproduceeveryoldKRRprediction<1e-8 andcomplete018<1e-8. No validation-basedepochorparameteralteration.','expansion_gate':'Afteroriginalall5andindependentaudit, runBOTHexistingresplits unchanged onlyif RMSEgain(all-subset80)>.01 AND >=3/5foldgains>0 AND fullpredictionRMSdifference>.05. Otherwise stop original. Thresholds are resource/interpretation rules, not significance or promotion gates.','practical_negligibility':'RMSdifference<=.05 AND absoluteRMSEdifference<=.005. Also report rawdifferences andselectionchanges regardlessofgate. Clearharmdiagnostic=RMSEworsening>=.01 with>=4/5folds worse;anyfailureofexpansiongate stops,includingmixedsmall effects.','validation':'IndependentrawMinMax/standardization/blockweight/RBFlinearalgebra for all9x3innerfits andfullKRR eachnode; reconstructselection andIDs; allnewaffines andresidualmodels independent; completeouterV80labelpoison recomputeKRRselection/refits/downstream; anchorB labelpoison4KRRnodes+anchor; KRRselectionScomplementlabelpoison andresidualinnerH exclusion. Fullneuralretrainingnotclaimed.','resources':'Existingvenv CPU2; noMPSfits. Firstpairedfoldcap120s/4GBRSS,originalcap900s beforeaudits. Atmost3partitions basedonfrozengate. No packages/network/download/payment/rawimages/holdout/test/full500/newCSV/upload.','retraining_request':'Prepareonly isolated018fulltrainingreproductionplan, gapinventory andsource-backedtime/RAM estimates. DoNOTexecute fullretraining in174. Futurefirststage dev400only; any500stage includingformerholdoutrequiresseparateauthorization.','limitations':['Onefixed80%subset tests combinedselection-size/composition effect,notallpossible400/500splits.','Alllevels80% iscontrolleduniformanalogue; actualexistingproductionuses400/500 onlytop-levelKRR andallA/allC selectioninsidecalibration. This experimentdoesnotexactlyclonehistoricalproduction.','Same400adaptivehistory andoverlappingresplits; no independentperformanceclaim.','FrozenNNupstreamcertificates reused; changedKRRanddownstream freshlytested.','Moreorlessfitting data can changeeffective regularization andselectorvariability; no claimeffectsolelyfromsamplecount.'],'sources':{f:e.sha(_repo_path(R, f)) for f in names},'protected':e.read(_repo_path(R, 'artifacts/cubical-h1-168/plan.json'))['protected']};write(_repo_path(OUT, 'plan.json'),plan);print('PLAN_LOCKED',e.sha(_repo_path(OUT, 'plan.json')),flush=True)

def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in {**p['sources'],**p['protected']}.items():assert e.sha(_repo_path(R, f))==h,f
 return p

def select(d,U,offset,arm):
 S=U if arm=='all' else subset(d,U);assert set(S)<=set(U);folds=list(GroupKFold(3,shuffle=True,random_state=fv.INNER_SEED+offset).split(d['K'][S],d['y'][S],d['g'][S]));ip=np.empty((9,len(S)));records=[];ci=0
 for width in fv.WIDTHS:
  for alpha in fv.ALPHAS:
   for I,H in folds:
    assert not set(d['g'][S[I]])&set(d['g'][S[H]]);model=fv.BalancedKernel(d['Kblocks'],width,alpha).fit(d['K'][S[I]],d['y'][S[I]]);ip[ci,H]=model.predict(d['K'][S[H]])
   records.append({'bandwidth':width,'alpha':alpha,'inner_rmse':e.rmse(d['y'][S],ip[ci])});ci+=1
 index=int(np.argmin([z['inner_rmse'] for z in records]));return S,folds,ip,records,index

def fit_node(d,node,arm,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();U=node['U'];Q=node['Q'];assert not set(U)&set(Q) and not set(d['g'][U])&set(d['g'][Q]);S,folds,ip,scores,chosen=select(d,U,node['offset'],arm);rec=scores[chosen];model=fv.BalancedKernel(d['Kblocks'],rec['bandwidth'],rec['alpha']).fit(d['K'][U],d['y'][U]);pred=model.predict(d['K'][Q]);joblib.dump({'model':model,'train':U,'selection':S,'query':Q},_repo_io(path/'model.joblib'));np.savez_compressed(_repo_io(path/'predictions.npz'),U=U,S=S,Q=Q,inner=ip,predictions=pred);result={'name':node['name'],'arm':arm,'offset':node['offset'],'inner_seed':fv.INNER_SEED+node['offset'],'n_train':len(U),'n_selection':len(S),'scores':scores,'chosen':rec,'chosen_index':chosen,'splits':[{'train':S[I].tolist(),'validation':S[H].tolist()} for I,H in folds],'source_expected_max_error':float(abs(pred-node['expected']).max()),'model_sha256':e.sha(path/'model.joblib'),'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start}
 if arm=='all':assert result['source_expected_max_error']<1e-8,result
 write(path/'report.json',result);return pred,result

def replace_krr(d,split,k,arm,path):
 path.mkdir(exist_ok=False);dd=dict(d);dd['P']=d['P'].copy();dd['middleP']=dict(d['middleP']);dd['middleP'][k]=d['middleP'][k].copy();dd['cells']=dict(d['cells']);dd['deepP']=dict(d['deepP']);T=np.flatnonzero(d['fid']!=k);look={int(v):i for i,v in enumerate(T)};results={}
 for j in range(3):
  A,B,P,short,ref=d['cells'][k,j];dd['cells'][k,j]=(A,B,P.copy(),short,ref);dd['deepP'][k,j]=d['deepP'][k,j].copy()
 for node in nodes(d,split,k):
  p,r=fit_node(d,node,arm,path/node['name']);results[node['name']]=r;Q=node['Q']
  if node['name']=='outer':dd['P'][Q,1]=p
  elif node['name'].startswith('middle'):
   j=int(node['name'].split('-')[1]);dd['cells'][k,j][2][:,1]=p;dd['middleP'][k][[look[int(v)] for v in Q],1]=p
  else:
   _,j,ell=node['name'].split('-');j=int(j);A=d['cells'][k,j][0];la={int(v):i for i,v in enumerate(A)};dd['deepP'][k,j][[la[int(v)] for v in Q],1]=p
 return dd,results

def downstream(d,split,k,path):
 path.mkdir(exist_ok=False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);cp,cal=a.affine(a.mixture(d['middleP'][k],x.ARM),d['y'][T],a.mixture(d['P'][V],x.ARM));base=.9*cp+.1*d['short'][V];cor=[];calibrators=[]
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];pp,m=a.affine(a.mixture(d['deepP'][k,j],x.ARM),d['y'][A],a.mixture(P,x.ARM));anchor=.9*pp+.1*short;cor.append(old.system(d,B,anchor,V,k,j,split,path/f'block-{j}'));calibrators.append(m)
 pred=base+np.mean(cor,0);np.savez_compressed(_repo_io(path/'predictions.npz'),V=V,base=base,corrections=np.stack(cor),primary=pred);report={'outer_calibrator':cal,'anchor_calibrators':calibrators,'predictions_sha256':e.sha(path/'predictions.npz'),'block_reports':{f'block-{j}/report.json':e.sha(path/f'block-{j}/report.json') for j in range(3)}};write(path/'report.json',report);return pred,report

def pairfold(d,split,k,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time();dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k);pred={};recs={}
 for arm in ARMS:
  dest=path/arm;dest.mkdir();D,nodesrec=replace_krr(dd,split,k,arm,dest/'kernels');pred[arm],rr=downstream(D,split,k,dest/'pipeline');recs[arm]={'nodes':nodesrec,'pipeline_report_sha256':e.sha(dest/'pipeline/report.json')}
 np.testing.assert_allclose(pred['all'],d['reference018'][V],rtol=0,atol=1e-8);np.savez_compressed(_repo_io(path/'predictions.npz'),V=V,**pred);report={'split':split,'fold':k,'RMSE':{arm:e.rmse(d['y'][V],p) for arm,p in pred.items()},'reference018_RMSE':e.rmse(d['y'][V],d['reference018'][V]),'reference018_max_error':float(abs(pred['all']-d['reference018'][V]).max()),'prediction_difference_RMS':float(np.sqrt(np.mean((pred['all']-pred['subset80'])**2))),'selection_changed_nodes':sum((recs['all']['nodes'][name]['chosen']['bandwidth'],recs['all']['nodes'][name]['chosen']['alpha'])!=(recs['subset80']['nodes'][name]['chosen']['bandwidth'],recs['subset80']['nodes'][name]['chosen']['alpha']) for name in recs['all']['nodes']),'arms':recs,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start,'CPU_process_seconds':time.process_time()-cpu,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write(path/'report.json',report);assert report['seconds']<120 and report['peak_RSS_bytes']<4e9;print('PAIRED_FOLD_COMPLETE',split,k,json.dumps({key:report[key] for key in ['RMSE','prediction_difference_RMS','selection_changed_nodes','seconds','peak_RSS_bytes']}),flush=True);return pred,report

def run(split):
 check()
 if split!='original':assert e.read(_repo_path(OUT, 'original/report.json'))['expand'] and e.read(_repo_path(OUT, 'validation-original.json'))['passed']
 start=time.perf_counter();d=load(split);dest=_repo_path(OUT, split);dest.mkdir(exist_ok=True);pred={a:np.full(400,np.nan) for a in ARMS};reports=[]
 for k in range(5):
  path=dest/f'fold-{k}';V=np.flatnonzero(d['fid']==k)
  if (path/'report.json').exists():r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);P={arm:z[arm] for arm in ARMS}
  else:P,r=pairfold(d,split,k,path)
  for arm in ARMS:pred[arm][V]=P[arm]
  reports.append(r);assert time.perf_counter()-start<900
 scores={arm:e.rmse(d['y'],p) for arm,p in pred.items()};gain=scores['all']-scores['subset80'];fg=[r['RMSE']['all']-r['RMSE']['subset80'] for r in reports];rms=float(np.sqrt(np.mean((pred['all']-pred['subset80'])**2)));expand=gain>.01 and sum(v>0 for v in fg)>=3 and rms>.05;np.savez_compressed(_repo_io(dest/'predictions.npz'),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],**pred)
 result={'split':split,'RMSE':scores,'gain_all_minus_subset80':gain,'fold_gains':fg,'prediction_difference_RMS':rms,'selection_changed_nodes':sum(r['selection_changed_nodes'] for r in reports),'nodes_per_arm':65,'practically_negligible':rms<=.05 and abs(gain)<=.005,'clear_harm':gain<=-.01 and sum(v<0 for v in fg)>=4,'expand':expand,'no_promotion_decision':True,'predictions_sha256':e.sha(dest/'predictions.npz'),'fold_reports':{f'fold-{k}/report.json':e.sha(dest/f'fold-{k}/report.json') for k in range(5)},'input_hashes':dict(a.SOURCES),'seconds':time.perf_counter()-start};write(dest/'report.json',result);check();print('PARTITION_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','fold_reports']}),flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode=='prepare':prepare()
 else:run(mode)
