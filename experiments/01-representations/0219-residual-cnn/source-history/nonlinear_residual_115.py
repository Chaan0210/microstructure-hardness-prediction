"""One fixed strongly regularized RBF extension of honest-block residual113."""
import sys,time,datetime,argparse,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import residual_physics_113 as r
import numpy as np,joblib
from scipy.spatial.distance import pdist,cdist,squareform
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GroupKFold
e=r.prior;OUT=ROOT/'artifacts/nonlinear-residual-115';KINDS=(*r.CANDIDATES,'rbf1','rbf01');LAM={'rbf1':1.,'rbf01':.1}

def fit(X,y,blocks,kind):
 if kind in r.CANDIDATES:return r.fit_model(X,y,blocks,kind)
 bound=MinMaxScaler(clip=True).fit(X);sc=StandardScaler().fit(bound.transform(X));w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);Z=sc.transform(bound.transform(X))*w;dist=pdist(Z,'sqeuclidean');positive=dist[dist>1e-12];gamma=1/float(np.median(positive)) if len(positive) else 1.;K=np.exp(-gamma*squareform(dist));col=K.mean(0);grand=float(K.mean());C=K-K.mean(1)[:,None]-col[None,:]+grand;scale=float(np.trace(C)/len(C));degenerate=scale<=1e-12;scale=1. if degenerate else scale;C=C/scale;rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;alpha=LAM[kind]*len(y);coef=np.linalg.solve(C+alpha*np.eye(len(y)),y/rs);eigen=np.linalg.eigvalsh(C);df=float(np.sum(eigen/(eigen+alpha)));assert df<=1/LAM[kind]+1e-8
 return {'kind':kind,'n_train':len(y),'bound':bound,'scale':sc,'weights':w,'Z':Z,'gamma':gamma,'column':col,'grand':grand,'kernel_scale':scale,'degenerate':degenerate,'residual_scale':rs,'alpha':alpha,'coef':coef,'effective_df':df}

def predict(m,X):
 if m['kind'] in r.CANDIDATES:return r.predict(m,X)
 Z=m['scale'].transform(m['bound'].transform(X))*m['weights'];K=np.exp(-m['gamma']*cdist(Z,m['Z'],'sqeuclidean'));C=(K-K.mean(1)[:,None]-m['column'][None,:]+m['grand'])/m['kernel_scale'];return C@m['coef']*m['residual_scale']

def prepare():
 r.check();OUT.mkdir(exist_ok=False);protected=dict(r.check()['protected011']);previous=e.read(ROOT/'reports/candidate012-residual-preservation.json');assert previous['passed']
 for n in ['reports/candidate012-residual-summary.json','reports/candidate012-residual-verification.json','artifacts/submission-candidate-012.csv','artifacts/submission-candidate-012.manifest.json','artifacts/submission-candidate-012.components.csv','reports/residual-physics-113-summary.json','reports/residual-physics-113-pooled-addendum.json','artifacts/residual-physics-113/decision.json']:protected[n]=e.sha(ROOT/n)
 inputs={}
 for split in e.SPLITS:
  assert e.read(r.OUT/f'validation-{split}.json')['passed']
  for n in ['predictions.npz','report.json']:p=r.OUT/split/n;inputs[str(p.relative_to(ROOT))]=e.sha(p)
 plan={'study':'115 one nonlinear RBF extension of113','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit':'022/101 KRR trainedhardnesstargets;047jointkernel trainedtargets;074/077predictioncalibration;076diagnostics;113onlylinearresidualRidge. No samephysics57 nonlinear residuallearner found in projectsource/plans/summaries.','hypothesis':'Smooth nonlinear relationships in fixed57 morphology maypredict remaining signed011 error not capturedby linear113. One RBFfamily, notnewfeatures or upstreammodels.','nested_scope':r.check()['honest_block_design'],'candidates':list(KINDS),'control':'Exactly113 none/Ridge1/Ridge.1 selector; reuse verifiedinner predictions andmodels, reassemble outerprediction byteexact.','primary':'Extendselector to five candidates includingrbf1/rbf01. Same3fold per honestblock, minimumpooleduntrimmed residualMSE;tieorder none,ridge1,ridge01,rbf1,rbf01. RefitselectedonB106/107; unchangedouter011 plusuniformmeanof3corrections. No posthocweight/threshold or trimming.','kernel':'Same113 train-onlyMinMaxclip/StandardScaler/blockweights. RBF gamma=1/medianpositive trainingpair squared distance (>1e-12). Training-center kernel and dividebymeantraindiagonal. Crosskernel centeredusingonlytrainingcolumn/grandmean andcrossrowmean. No intercept/residualmean added. ResidualRMS trainonly;alpha=lambda*n,lambda1/.1;effective df<=1/10 respectively. Constantfeaturesfallbackgamma1,degeneratescale1;zero centeredkernel.','primary_metric':'All1200 predictionpairs equalweight pooleduntrimmedMSE, comparedwith both113 and011. Only400uniqueID; candidate ifpooledMSEstrictlybelowboth andindependentchecks pass. No everypartition/fold/CI/minimumgain/influence-exclusiongate.','diagnostics':'SameID3errors clusteredfor10000bootstrapdraws seed20271016. Fixed092fiveIDs and five largestpositivecontributionIDs exclusion diagnosticsonly. No globalOOFmodel fitting oroptimizedOOFweight.','resources':'CPU2,existingpackages,270newinnerKRRfits plusatmost45selectedfits. No newbackbonefit;45verified112anchors reused. Firstblockreusedbenchmark projected45<120sec,RSS<2GB.','if_failure':'Preserve011/113/012. Report next nonduplicate executable hypothesis, notanautomaticgamma/alpha/ratio sweep.','scope':'Development400 only forcomparison. No test/holdouttuning,externaltransmission,install,cost,orupload. Alloldrecords preserved.','sources':{n:e.sha(ROOT/n) for n in ['nonlinear_residual_115.py','validate_nonlinear_residual_115.py','residual_physics_113.py']},'inputs':inputs,'protected':protected,'environment':r.check()['environment']};e.write(OUT/'plan.json',plan);e.emit({'prepared':True,'kinds':KINDS,'plan_sha256':e.sha(OUT/'plan.json')})

def check():
 r.check();p=e.read(OUT/'plan.json')
 for n,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(ROOT/n)==h,n
 assert {x.metadata['Name'].lower().replace('_','-'):x.version for x in importlib.metadata.distributions()}==p['environment'];return p

def block(d,split,k,j,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();A,B,pa,trace=r.anchor(d,split,k,j);res=d['y'][B]-pa;va=np.flatnonzero(d['fid']==k);old=r.OUT/split/f'fold-{k}/block-{j}';rec=e.read(old/'report.json');assert e.sha(old/'fit_data.npz')==rec['fit_data_sha256']
 with np.load(old/'fit_data.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['residual'],res);ip=np.full((5,len(B)),np.nan);ip[:3]=z['inner_predictions'];control=z['correction'].copy()
 folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]));files={}
 for t,(it,iv) in enumerate(folds):
  assert rec['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()}
  for ci,kind in enumerate(KINDS[3:],3):
   m=fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=predict(m,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[it],'validation_indices':B[iv]},file);files[file.name]=e.sha(file)
 scores=[float(np.mean((res-p)**2)) for p in ip];chosen=KINDS[int(np.argmin(scores))];controlchoice=r.CANDIDATES[int(np.argmin(scores[:3]))];assert controlchoice==rec['chosen']
 m=fit(d['X'][B],res,d['blocks'],chosen);pred=predict(m,d['X'][va]);joblib.dump({'model':m,'train_indices':B,'chosen':chosen},path/'selected.joblib')
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(f,anchor_indices=A,block_indices=B,outer_indices=va,anchor_predictions=pa,residual=res,inner_predictions=ip,correction=pred,control_correction=control)
 report={'split':split,'fold':k,'block':j,'chosen':chosen,'control_chosen':controlchoice,'inner_MSE':dict(zip(KINDS,scores)),'inner_splits':rec['inner_splits'],'anchor_trace':trace,'models_sha256':files,'selected_sha256':e.sha(path/'selected.joblib'),'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);return pred,control,report

def run(split):
 check();d=r.load(split);out=OUT/split;out.mkdir(exist_ok=False);start=time.perf_counter();cor=np.full((3,400),np.nan);control=cor.copy();records=[]
 for k in range(5):
  for j in range(3):
   va=np.flatnonzero(d['fid']==k);pred,c,br=block(d,split,k,j,out/f'fold-{k}-block-{j}');cor[j,va]=pred;control[j,va]=c;records.append(br)
   if split=='original' and k==0 and j==0:
    import resource
    b={'first_block_seconds_reused':br['seconds'],'projected45_seconds':45*br['seconds'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected45_seconds']<120 and b['RSS_bytes']<2e9;e.write(OUT/'benchmark.json',b);assert b['passed']
  e.emit({'split':split,'outer_complete':k,'selected':[z['chosen'] for z in records[-3:]],'seconds':time.perf_counter()-start})
 ps={'reference011':d['reference011'],'reference113':d['reference011']+control.mean(0),'primary':d['reference011']+cor.mean(0)}
 with np.load(r.OUT/split/'predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(ps['reference113'],z['primary'])
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(f,ids=np.array(d['ids']),y=d['y'],fold=d['fid'],corrections=cor,**ps)
 report={'split':split,'scores':{n:{'MSE':float(np.mean((d['y']-p)**2)),'RMSE':r.rmse(d['y'],p)} for n,p in ps.items()},'selected_counts':{n:sum(v['chosen']==n for v in records) for n in KINDS},'predictions_sha256':e.sha(out/'predictions.npz'),'block_reports_sha256':{f'fold-{v["fold"]}-block-{v["block"]}/report.json':e.sha(out/f'fold-{v["fold"]}-block-{v["block"]}/report.json') for v in records},'seconds':time.perf_counter()-start};e.write(out/'report.json',report);check();e.emit(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);a=p.parse_args();prepare() if a.mode=='prepare' else run(a.mode)
