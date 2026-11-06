from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,time,json,hashlib,datetime,resource
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
P=_RepoPath(__file__).resolve().parents[3];R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(P)];assert Path(sys.prefix).resolve()==_repo_path(P, '.venv')
from policy import PolicyKernel
threadpool_limits(2)
ARMS=('A_current','B_validity','C_valid_mean');SPLITS=('original','20261031','20261101');GRID=[(w,a) for w in (.25,1.,4.) for a in (.01,.1,1.)]
def read(p):return json.loads(Path(p).read_text())
def write(p,z):
 with Path(p).open('x') as f:json.dump(z,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rmse(y,p):return float(np.sqrt(np.mean((y-p)**2)))
def emit(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def protect():
 q=read(_repo_path(R, 'plan.json'))
 for p,h in q['protected'].items():assert sha(_repo_path(P, p))==h,p
 for p,h in q['sources'].items():assert sha(_repo_path(R, p))==h,p
 return q
def guard():
 def hook(ev,args):
  if ev=='socket.connect':raise RuntimeError('Network forbidden')
  if ev=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).absolute()
   if p.is_relative_to(_repo_path(P, 'data')):raise RuntimeError('Rawdata/holdout/test forbidden; dev caches only')
   mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
   if writing and not p.is_relative_to(R) and str(p)!='/dev/null':raise RuntimeError('Only new218writes allowed')
 sys.addaudithook(hook)
def features():
 z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'));g=load(_repo_path(P, 'artifacts/run-006/development_features.npz'));np.testing.assert_array_equal(z['ids'],g['ids']);names=read(_repo_path(P, 'artifacts/run-006/report.json'))['feature_names'];hp=np.column_stack([g['X'][:,names.index(f'grain_s{s}_interior_hall_petch')] for s in [1,2]]);counts=np.column_stack([g['X'][:,names.index(f'grain_s{s}_interior_count')] for s in [1,2]]);valid=counts>0;assert np.all(hp[valid]>0) and np.all(hp[~valid]==0);np.testing.assert_array_equal(z['X'][:,0],hp.mean(1));flags=(~valid).astype(float);A=np.column_stack([z['X'],np.zeros_like(flags)]);B=np.column_stack([z['X'],flags]);C=B.copy();num=valid.sum(1);C[:,0]=np.divide((hp*valid).sum(1),num,out=np.full(400,np.nan),where=num>0);blocks=read(_repo_path(P, 'artifacts/run-012/report.json'))['blocks']+['grain_validity']*2;assert A.shape==B.shape==C.shape==(400,28)
 return z,{'A_current':A,'B_validity':B,'C_valid_mean':C},blocks,valid,hp

def source(split):
 z=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'));assert sha(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'))==read(_repo_path(P, f'experiments/dinol-177/{split}/report.json'))['predictions_sha256'];old=load(_repo_path(P, f'artifacts/component-ablation-121/{split}/predictions.npz'));base=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
 for k in ['ids','y']:np.testing.assert_array_equal(z[k],base[k]);np.testing.assert_array_equal(old[k],z[k])
 np.testing.assert_array_equal(old['fold'],z['fold']);np.testing.assert_array_equal(z['g'],base['groups']);assert len(set(z['ids']))==400
 coeff=np.empty(400)
 for k in range(5):
  c=read(_repo_path(P, f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/report.json'))['outer_calibrator'];assert c==read(_repo_path(P, f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json'))['outer_calibrator'];coeff[z['fold']==k]=.243*c['slope']
 return z,old['components'][:,1],coeff

def prepare():
 z,xx,blocks,valid,hp=features();protected={}
 for f in ['artifacts/submission-candidate-021.csv','artifacts/run-012/development_features.npz','artifacts/run-012/report.json','artifacts/run-006/development_features.npz','artifacts/run-006/report.json','compact_regression.py','docs/records/diagnostics/l1-l4-20261007/summary.json','docs/records/diagnostics/l1-l4-20261007/L4-legacy-measurements.json','docs/records/diagnostics/l1-l4-20261007/L4.json']:
  protected[f]=sha(_repo_path(P, f))
 for split in SPLITS:
  for f in [f'experiments/dinol-177/{split}/predictions.npz',f'experiments/dinol-177/{split}/report.json',f'artifacts/component-ablation-121/{split}/predictions.npz']+[f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/report.json' for k in range(5)]+[f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json' for k in range(5)]:protected[f]=sha(_repo_path(P, f))
 for f,h in read(_repo_path(P, 'experiments/candidate021-178/models/manifest.json'))['files'].items():f='experiments/candidate021-178/models/'+f;assert sha(_repo_path(P, f))==h;protected[f]=h
 p={'study':218,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Fixed validity encoding control for actualcompact26 KRR; no segmentation tuning or newgeometrycorrection','IDs':z['ids'].tolist(),'arms':{'A_current':'Original26 + two zero columns (controlpad); same existing blockweights for26.','B_validity':'Original26 + missing flags sigma1/sigma2 as separate two-column grain_validity block, scale1/sqrt2.','C_valid_mean':'B plus inverse-sqrt-grain-diameter column averages only scales with interior_count>0; all-invalid -> NaN then foldtrain median of observed HP, fallback0 if none observed. Indicator remains. No other zero values changed.'},'model':'Existing BalancedKernel MinMaxclip/StandardScaler/blockscale median-distance RBF, targetmeanstd, summedkernel alpha; identical28dimensions/blocks/grid/seeds. Train-onlyimputer/scaler/gamma/y stats at each inner andouter fit.','grid':GRID,'inner':'GroupKFold3shuffle random_state20261005+outerfold; same IDs allarms; pooledinnerRMSEmin, fixedlistorder ties; noearlystop/additionaltuning','outer':list(SPLITS),'fits_planned':{'outer':45,'inner':1215,'label_poison_outer_paths':9,'selected_model_formula_checks':45},'segmentation_geometry':'Existingsigma1/2 area>=16 strictbbox interior only; count0 denotesunobservedinteriors, not measuredHP0. Validgrain inverse-sqrtHPpositive. Other legitimatezeros (CVmonodisperse, absentphase) leftuntouched. Diskcomplete-object sampling favours smallgrains; this study correctsencoding only, no claimofunbiasedphysicalsize or inverseprobabilitycorrection.','known_empty_IDs':z['ids'][~valid.all(1)].tolist(),'ensemble':'Conditional frozen replacement P021 + .243*fold_affine_slope*(KRR_arm-KRR_old). .243=.9Lbranch*.9calibratedbranch*.3KRRraw. KRR control must match historicalcomponent1<1e-8; P021 reappears exactlywithinthat tolerance. Allcalibration/anchors/residualmodelsheldfixed; not fullrefittedpipeline or deployablecandidate. No changes to residualphysics57 in thisstudy.','statistics':'Prespecifiedpaired B-A,C-A,C-B; primaryC-A. 10000fixedID bootstrap seed202610218; pooledSEaverage across3partitions keep IDrepeatsclustered. Report whole400, known3, remaining397;3-row uncertainty descriptiveonly, no narrowCI/pvalueclaim. PerIDdeltaSSE and actualup/downcounts. No outcome-definedtrimming.','decision':'No automaticpromotion. Consistent-proxy signal if C-A pooledfixedensemblegain>0, at least2of3splitgains>0 including arepartition, remaining397 pooledgain>=0. Always report pairedCI;CIincluding0meansuncertain, notproof. Otherwise weak/mixedorharmful and no tuning torescue. Sameclassificationforstandalone;fixedproxycannotestablishfullpipelineeffect.','scope':'Only existingdev400 caches; CPU2 existinguv; no torchtraining/MPS/installation/API/rawdata/holdout/test/full500/CSV/submission','resources':'Firstfold benchreport; stop ifprocessRSS>4GB or wholetraining>900s, noresourcechangingrescue','protected':protected,'sources':{f:sha(_repo_path(R, f)) for f in ['run.py','policy.py']},'limitations':['Allsame400 adaptivelyreused;conditionalbootstrap omits retraining/selectionuncertainty.','3empties preidentified withoutlabels, too few for stable subgroupinference.','Flagging inducesmetric changes and may change hyperparameterselection affectingother397, measuredseparately.']};write(_repo_path(R, 'plan.json'),p)
 # Validate zero/missing semantics and image-local invariant without any modelscore.
 assert np.array_equal(xx['A_current'][:,:26],z['X']);assert np.array_equal(xx['B_validity'][:,:26],z['X']);assert np.sum(xx['C_valid_mean'][:,0]!=xx['A_current'][:,0])==3
 for s in range(2):assert np.array_equal(hp[:,s]==0,~valid[:,s])
 # No allmissing row exists in data. Unit fixture checks fold-local fallback and legitimatezero.
 dummy=xx['C_valid_mean'][:6].copy();dummy[0,0]=np.nan;dummy[1,1]=0;dummy[2,0]=0;ys=np.arange(6,dtype=float);m=PolicyKernel(blocks,1,.1).fit(dummy,ys);assert m.impute_==float(np.median(dummy[1:,0])) and m.transform(dummy)[1,1]==0 and m.transform(dummy)[2,0]==0
 query=dummy[:1].copy();query[0,0]=np.nan;assert m.transform(query)[0,0]==m.impute_
 write(_repo_path(R, 'preflight.json'),{'passed':True,'known_empty_IDs':p['known_empty_IDs'],'C_changes_exactly3_HPentries':True,'all_other_original26entries_unchanged':True,'normal_measured_zero_fixture_preserved':True,'foldtrain_only_imputation_fixture':True,'unit_fixture_tiny_KRR_fits':1,'geometry_diagnostic_source':'docs/records/diagnostics/l1-l4-20261007/L4-legacy.json','physical_segmentation_groundtruth':False});emit({'plan_locked':True,'arms':list(ARMS),'changed_IDs':p['known_empty_IDs']})

def fit(X,y,g,T,V,blocks,k,store=None):
 inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(X[T],y[T],g[T]));ip=np.zeros((9,len(T)));scores=[];means=[]
 for ci,(w,a) in enumerate(GRID):
  for I,H in inner:
   assert not set(T[I])&set(T[H]);m=PolicyKernel(blocks,w,a).fit(X[T[I]],y[T[I]]);ip[ci,H]=m.predict(X[T[H]])
  scores.append(rmse(y[T],ip[ci]))
 choice=int(np.argmin(scores));w,a=GRID[choice];m=PolicyKernel(blocks,w,a).fit(X[T],y[T]);pred=m.predict(X[V]);info={'inner_scores':scores,'chosen_index':choice,'chosen':{'width':w,'alpha':a},'train_indices':T.tolist(),'validation_indices':V.tolist(),'inner_splits':[{'train':T[I].tolist(),'validation':T[H].tolist()} for I,H in inner],'imputer':m.impute_}
 if store is not None:
  store.mkdir();joblib.dump(m,_repo_io(store/'model.joblib'));np.savez_compressed(_repo_io(store/'predictions.npz'),inner=ip,prediction=pred);write(store/'report.json',{**info,'model_sha256':sha(store/'model.joblib'),'prediction_sha256':sha(store/'predictions.npz')})
 return pred,m,info,ip

def run():
 protect();guard();start=time.monotonic();z,xx,blocks,valid,hp=features();y=z['y'];g=z['groups'];nfits=0
 for split in SPLITS:
  dest=_repo_path(R, split);dest.mkdir();ref,expected,coeff=source(split);out={a:np.zeros(400) for a in ARMS};folds=[]
  for k in range(5):
   fs=time.monotonic();T=np.flatnonzero(ref['fold']!=k);V=np.flatnonzero(ref['fold']==k);assert len(T)==320 and len(V)==80 and not set(g[T])&set(g[V]);froot=dest/f'fold-{k}';froot.mkdir();log={'fold':k,'known_empty_train':int((~valid[T].all(1)).sum()),'known_empty_validation':int((~valid[V].all(1)).sum())}
   for arm in ARMS:
    pr,m,info,ip=fit(xx[arm],y,g,T,V,blocks,k,froot/arm);out[arm][V]=pr;nfits+=28;log[arm]={'chosen':info['chosen'],'RMSE':rmse(y[V],pr)}
    if arm=='A_current':assert np.max(abs(pr-expected[V]))<1e-8,(split,k,float(np.max(abs(pr-expected[V]))))
   log['seconds']=time.monotonic()-fs;log['RSS']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;folds.append(log);write(froot/'completion.json',log);emit({'split':split,**log});assert log['RSS']<4e9 and time.monotonic()-start<900
  ens={arm:ref['blend_L_MLP']+coeff*(p-expected) for arm,p in out.items()};assert np.max(abs(ens['A_current']-ref['blend_L_MLP']))<1e-8
  np.savez_compressed(_repo_io(dest/'OOF.npz'),ids=z['ids'],y=y,fold=ref['fold'],groups=g,empty=~valid.all(1),reference021=ref['blend_L_MLP'],old_KRR=expected,coefficient=coeff,**out,**{'ensemble_'+a:p for a,p in ens.items()});write(dest/'completion.json',{'split':split,'folds':folds,'OOF_sha256':sha(dest/'OOF.npz'),'control_KRR_maxerr':float(np.max(abs(out['A_current']-expected))),'control021_maxerr':float(np.max(abs(ens['A_current']-ref['blend_L_MLP'])))})
 protect();write(_repo_path(R, 'training-complete.json'),{'completed':True,'main_KRR_fits':nfits,'seconds':time.monotonic()-start,'RSS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'noCNNorMPSfits':True});emit({'completed':True,'fits':nfits,'seconds':time.monotonic()-start})
if __name__=='__main__':globals()[sys.argv[1]]()
