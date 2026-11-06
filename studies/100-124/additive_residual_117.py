"""Fixed physical-block additive RBF; one diagnostic half-size residual comparison."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,importlib.metadata
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nonlinear_residual_115 as n
import numpy as np,joblib
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GroupKFold
e=n.e;r=n.r;OUT=_repo_path(ROOT, 'artifacts/additive-residual-117');ADD=('add1','add01');KINDS=(*n.KINDS,*ADD);SETS={'linear':r.CANDIDATES,'joint':n.KINDS,'additive':(*r.CANDIDATES,*ADD)};LAM={'add1':1.,'add01':.1};SUBSEED=20271017

def fit(X,y,blocks,kind):
 if kind not in ADD:return n.fit(X,y,blocks,kind)
 bound=MinMaxScaler(clip=True).fit(X);sc=StandardScaler().fit(bound.transform(X));weights=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=sc.transform(bound.transform(X))*weights;parts=[];C=np.zeros((len(X),len(X)))
 for name in dict.fromkeys(blocks):
  cols=np.flatnonzero(np.array(blocks)==name);dist=pdist(Z[:,cols],'sqeuclidean');pos=dist[dist>1e-12];gamma=1/float(np.median(pos)) if len(pos) else 1.;K=np.exp(-gamma*squareform(dist));column=K.mean(0);grand=float(K.mean());centered=K-K.mean(1)[:,None]-column[None,:]+grand;scale=float(np.trace(centered)/len(X));degenerate=scale<=1e-12;scale=1. if degenerate else scale;C+=centered/scale;parts.append({'name':name,'columns':cols,'gamma':gamma,'column':column,'grand':grand,'kernel_scale':scale,'degenerate':degenerate})
 C/=len(parts);alpha=LAM[kind]*len(y);rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;coef=np.linalg.solve(C+alpha*np.eye(len(y)),y/rs);eig=np.linalg.eigvalsh(C);df=float(np.sum(eig/(eig+alpha)));assert df<=1/LAM[kind]+1e-8
 return {'kind':kind,'bound':bound,'scale':sc,'weights':weights,'Z':Z,'parts':parts,'n_train':len(y),'alpha':alpha,'residual_scale':rs,'coef':coef,'effective_df':df}

def predict(m,X):
 if m['kind'] not in ADD:return n.predict(m,X)
 Z=m['scale'].transform(m['bound'].transform(X))*m['weights'];K=np.zeros((len(X),m['n_train']))
 for part in m['parts']:
  cols=part['columns'];raw=np.exp(-part['gamma']*cdist(Z[:,cols],m['Z'][:,cols],'sqeuclidean'));K+=(raw-raw.mean(1)[:,None]-part['column'][None,:]+part['grand'])/part['kernel_scale']
 return (K/len(m['parts']))@m['coef']*m['residual_scale']

def prepare():
 n.check();OUT.mkdir(exist_ok=False);blocks=r.load('original')['blocks'];protected=dict(n.check()['protected'])
 for x in ['artifacts/submission-candidate-013.csv','artifacts/submission-candidate-013.manifest.json','artifacts/submission-candidate-013.components.csv','reports/candidate013-summary.json','reports/candidate013-verification.json','reports/nonlinear-residual-115-summary.json']:protected[x]=e.sha(_repo_path(ROOT, x))
 inputs={}
 for split in e.SPLITS:
  assert e.read(n.OUT/f'validation-{split}.json')['passed']
  for name in ['report.json','predictions.npz']:p=n.OUT/split/name;inputs[str(p.relative_to(ROOT))]=e.sha(p)
 plan={'study':'117 physical-block additive RBF and residual-family stability','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_blocks':{b:[i for i,x in enumerate(blocks) if x==b] for b in dict.fromkeys(blocks)},'block_source':'Unchanged physical labels from run022; no regrouping, dropping or weight selection usingvalidation.','hypothesis':'Joint57distance mixesphysicalblocks;uniformadditive blockkernels removebetween-blockinteractions while retainingsmoothwithin-blocknonlinearity. Distinctfrom047DINO+morphologytargetkernel.','primary_candidates':list(SETS['additive']),'matched_control':list(SETS['joint']),'nested':'Exactly113/115 honest residualblocks andinner3fold seed. Samefeature/RMSfit scope. Each blockkernel centeredandunitmeantraindiagonal usinginnertrain;equalmeanoverall10;constantkernelzero,fixeddenominator10. gamma trainingmedianpositive squared distance only. alpha=n*lambda,lambda1/.1;nointercept.','comparison':'Fivechoices foradditive versusfiveforjoint115;replacekernel family ratherthan enlargechoice set. Full106/107residualtraining primary;exact113/115controls preserved.','primary_metric':'All3partition pooleduntrimmedMSE on1200predictionpairs/400uniqueIDs; candidacy onlyiflowerthan115 and011 andvalidationpasses. No allpartition/fold/CI/influence-exclusiongate.','size_diagnostic':'One fixedhalfgroup sample fromeachhonestB106/107, seed20271017+10000*splitindex+100*outer+block, chooseceil(groupcount/2)groups. AnchorA213/214andouter80fixed. Refit trainlocalscalers,3foldselectionandmodels forall3families. Fullsizeusesexisting113/115/new117. Halfsize isdiagnosticonly;nevercandidate orparameterchoice. No anchorrefits or holdout labels. This isolatesresiduallearner-size effect locally,not anchor213->333orresidual107->167 generalization.','family_summary':'Poolederrors,allIDclusterbootstrapcomparisons,foldsigns,conditionalinfluentialIDs; correction/residual correlations andincremental versusbase MSE decomposition 2mean(error011*correction)-mean(correction²). No learnedcrossfamilyblend orglobalOOFfittedcoefficients.','resource':'CPU2existingpackages. Full270newadditiveinnerfits;half810nonnullinnerfits+selected. 45upstreamanchorsreused. Projectfirstfull+halfblock45<180sec,RSS<2GB.','stopping':'Afterthiscomparison, summarizewholefamily;do not launch automaticgamma/alpha/block/mixture sweeps. Furtherresearchneeds evidencefromstability/size diagnostics.','scope':'No newdata/packages/cost/externaltransmission/testorholdouttuning/upload. Development400 only duringcomparison andsizeanalysis.','sources':{x:e.sha(_repo_path(ROOT, x)) for x in ['additive_residual_117.py','validate_additive_residual_117.py','nonlinear_residual_115.py']},'inputs':inputs,'protected':protected,'environment':n.check()['environment']};e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'blocks':list(plan['fixed_blocks']),'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 n.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for x,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, x))==h,x
 assert {x.metadata['Name'].lower().replace('_','-'):x.version for x in importlib.metadata.distributions()}==p['environment'];return p

def subset(d,B,split,k,j):
 groups=np.unique(d['g'][B]);rng=np.random.default_rng(SUBSEED+10000*e.SPLITS.index(split)+100*k+j);chosen=rng.permutation(groups)[:(len(groups)+1)//2];return B[np.isin(d['g'][B],chosen)]

def block(d,split,k,j,size,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();A,B,pa,trace=r.anchor(d,split,k,j);whole=B.copy();oldpa=pa.copy()
 if size=='half':B=subset(d,B,split,k,j);pa=pa[np.isin(whole,B)]
 res=d['y'][B]-pa;va=np.flatnonzero(d['fid']==k);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]));ip=np.full((7,len(B)),np.nan);files={};splits=[]
 for t,(it,iv) in enumerate(folds):
  splits.append({'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()})
  for ci,kind in enumerate(KINDS):
   if size=='full' and kind not in ADD:
    src=(r.OUT/split/f'fold-{k}/block-{j}' if kind in r.CANDIDATES else n.OUT/split/f'fold-{k}-block-{j}')/f'inner-{t}-{kind}.joblib';old=joblib.load(_repo_io(src));np.testing.assert_array_equal(old['train_indices'],B[it]);np.testing.assert_array_equal(old['validation_indices'],B[iv]);m=old['model'];files[f'inner-{t}-{kind}']={'source':str(src.relative_to(ROOT)),'sha256':e.sha(src)}
   else:
    m=fit(d['X'][B[it]],res[it],d['blocks'],kind);src=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(src));files[f'inner-{t}-{kind}']={'source':str(src.relative_to(ROOT)),'sha256':e.sha(src)}
   ip[ci,iv]=predict(m,d['X'][B[iv]])
 scores=np.mean((ip-res[None,:])**2,1);selected={};models={};preds={}
 for family,kinds in SETS.items():
  choices=[KINDS.index(kind) for kind in kinds];kind=KINDS[choices[int(np.argmin(scores[choices]))]];selected[family]=kind
  if size=='full' and family!='additive':
   src=(r.OUT/split/f'fold-{k}/block-{j}' if family=='linear' else n.OUT/split/f'fold-{k}-block-{j}')/'selected.joblib';saved=joblib.load(_repo_io(src));assert saved['chosen']==kind;np.testing.assert_array_equal(saved['train_indices'],B);m=saved['model']
  elif kind in models:m,src=models[kind]
  else:
   m=fit(d['X'][B],res,d['blocks'],kind);src=path/f'selected-{kind}.joblib';joblib.dump({'model':m,'train_indices':B,'chosen':kind},_repo_io(src))
  models[kind]=(m,src);preds[family]=predict(m,d['X'][va]);files[f'selected-{family}']={'source':str(src.relative_to(ROOT)),'sha256':e.sha(src)}
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,whole_block_indices=whole,block_indices=B,anchor_predictions=pa,residual=res,inner_predictions=ip,outer_indices=va,**preds)
 report={'split':split,'fold':k,'block':j,'size':size,'n_train':len(B),'selected':selected,'inner_MSE':dict(zip(KINDS,scores.tolist())),'inner_splits':splits,'anchor_trace':trace,'files':files,'prediction_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);return preds,report

def run(split):
 check();d=r.load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();cor={size:{family:np.full((3,400),np.nan) for family in SETS} for size in ['full','half']};records=[];bench=[]
 for k in range(5):
  for j in range(3):
   for size in ['full','half']:
    pp,br=block(d,split,k,j,size,out/f'{size}-fold-{k}-block-{j}');records.append(br);va=np.flatnonzero(d['fid']==k)
    for family in SETS:cor[size][family][j,va]=pp[family]
    if split=='original' and k==0 and j==0:bench.append(br['seconds'])
   if split=='original' and k==0 and j==0:
    import resource
    b={'first_full_half_block_seconds':sum(bench),'projected45_seconds':45*sum(bench),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected45_seconds']<180 and b['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
  e.emit({'split':split,'outer_complete':k,'elapsed':time.perf_counter()-start})
 ps={'reference011':d['reference011'],**{f'{size}_{family}':d['reference011']+cor[size][family].mean(0) for size in cor for family in SETS}}
 with np.load(_repo_io(n.OUT/split/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(ps['full_linear'],z['reference113']);np.testing.assert_array_equal(ps['full_joint'],z['primary'])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**ps)
 report={'split':split,'scores':{key:{'MSE':float(np.mean((d['y']-p)**2)),'RMSE':r.rmse(d['y'],p)} for key,p in ps.items()},'selected_counts':{size:{family:{kind:sum(x['selected'][family]==kind for x in records if x['size']==size) for kind in kinds} for family,kinds in SETS.items()} for size in ['full','half']},'sample_sizes':{size:sorted(set(x['n_train'] for x in records if x['size']==size)) for size in ['full','half']},'report_sha256':{f'{x["size"]}-fold-{x["fold"]}-block-{x["block"]}/report.json':e.sha(out/f'{x["size"]}-fold-{x["fold"]}-block-{x["block"]}/report.json') for x in records},'prediction_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(out/'report.json',report);check();e.emit({'split':split,'scores':report['scores'],'sample_sizes':report['sample_sizes'],'seconds':report['seconds']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
