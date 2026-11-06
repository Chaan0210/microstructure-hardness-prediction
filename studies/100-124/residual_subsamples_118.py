"""Three prespecified half-size diagnostics for fixed115, no candidate/model search."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import nonlinear_residual_115 as n
import additive_residual_117 as previous
import validate_nonlinear_residual_115 as independent
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=n.e;r=n.r;OUT=_repo_path(ROOT, 'artifacts/residual-subsamples-118');SEEDS=(20271019,20271020,20271021)

def prepare():
 previous.check();OUT.mkdir(exist_ok=False);protected=dict(previous.check()['protected'])
 for name in ['reports/residual-family-117-summary.json','reports/residual-family-117-preservation.json','reports/candidate013-preservation.json']:protected[name]=e.sha(_repo_path(ROOT, name))
 p={'study':'118 three additional fixed115 half-group draws','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':list(SEEDS),'subset':'For eachoriginalhonestB106/107, chooseceil(uniquegroups/2)groupsof fixedrandompermutation seed=drawseed+10000*splitindex+100*outer+block. Same53/54rowsas117 exceptdifferentmembership. Threeadditionaldrawsallretained;117firstdrawretainedasdraw0.','learner':'Exactly115selector none/ridge1/ridge01/rbf1/rbf01;sameinner3foldseed20271013+100*outer+block,fulltrainlocalstatistics,targetRMSandgamma. Anchor213/214andouter011320unchanged.','purpose':'Diagnose compositionvariance andlocalresidualsupervisionsize limits;notselectmodel,kernel,weight,draw,subset orcandidate. Reportall4draws andmeanof squared errorsacrossdraws,notanensembleprediction.','primary_analysis':'SameIDclustered uncertainly;all3partitionpooleduntrimmedMSE,draw dispersion,selection/correctionstability,subsetoverlap. No p-values claimingindependentsamples from repeated400.','scope':'Development400only;noadditionaltest/holdoutdecoding,upstreamtraining,install,cost,externaltransmission orupload. 013and011preserved. No newCSV.','resource':'135halfblockfits,1620nonnullinnerfits+135selectedfits;CPU2existingpackages. Firstblockreusedproject135<120s,RSS<2GB.','sources':{name:e.sha(_repo_path(ROOT, name)) for name in ['residual_subsamples_118.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py']},'protected':protected,'environment':previous.check()['environment']};e.write(_repo_path(OUT, 'plan.json'),p);e.emit({'prepared':True,'seeds':SEEDS,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def check():
 previous.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['sources'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p

def subset(d,B,split,k,j,draw):
 seed=SEEDS[draw]+10000*e.SPLITS.index(split)+100*k+j;groups=np.unique(d['g'][B]);chosen=np.random.default_rng(seed).permutation(groups)[:(len(groups)+1)//2];return B[np.isin(d['g'][B],chosen)]

def block(d,split,k,j,draw,path):
 path.mkdir(exist_ok=False);start=time.perf_counter();A,whole,pa,trace=r.anchor(d,split,k,j);B=subset(d,whole,split,k,j,draw);pa=pa[np.isin(whole,B)];res=d['y'][B]-pa;va=np.flatnonzero(d['fid']==k);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]));ip=np.full((5,len(B)),np.nan);hashes={}
 for t,(it,iv) in enumerate(folds):
  assert not set(d['g'][B[it]])&set(d['g'][B[iv]])
  for ci,kind in enumerate(n.KINDS):
   m=n.fit(d['X'][B[it]],res[it],d['blocks'],kind);ip[ci,iv]=n.predict(m,d['X'][B[iv]]);file=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train_indices':B[it],'validation_indices':B[iv]},_repo_io(file));hashes[file.name]=e.sha(file)
 scores=np.mean((ip-res[None,:])**2,1);chosen=n.KINDS[int(scores.argmin())];m=n.fit(d['X'][B],res,d['blocks'],chosen);pred=n.predict(m,d['X'][va]);joblib.dump({'model':m,'train_indices':B,'chosen':chosen},_repo_io(path/'selected.joblib'))
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),anchor_indices=A,whole_block_indices=whole,block_indices=B,anchor_predictions=pa,residual=res,inner_predictions=ip,outer_indices=va,correction=pred)
 report={'split':split,'fold':k,'block':j,'draw':draw,'seed':SEEDS[draw],'n_train':len(B),'chosen':chosen,'inner_MSE':dict(zip(n.KINDS,scores.tolist())),'inner_splits':[{'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} for it,iv in folds],'anchor_trace':trace,'files_sha256':hashes,'selected_sha256':e.sha(path/'selected.joblib'),'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);return pred,report

def run(split):
 check();d=r.load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);start=time.perf_counter();cor=np.full((3,3,400),np.nan);records=[]
 for draw in range(3):
  for k in range(5):
   for j in range(3):
    pp,rec=block(d,split,k,j,draw,out/f'draw-{draw}-fold-{k}-block-{j}');cor[draw,j,d['fid']==k]=pp;records.append(rec)
    if split=='original' and draw==k==j==0:
     import resource
     b={'first_block_seconds':rec['seconds'],'projected135_seconds':135*rec['seconds'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};b['passed']=b['projected135_seconds']<120 and b['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),b);assert b['passed']
  e.emit({'split':split,'draw_completed':draw,'seconds':time.perf_counter()-start})
 pred=d['reference011'][None,:]+cor.mean(1)
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference011=d['reference011'],corrections=cor,predictions=pred)
 report={'split':split,'draw_RMSE':[r.rmse(d['y'],p) for p in pred],'draw_MSE':[float(np.mean((d['y']-p)**2)) for p in pred],'selected_counts':{str(draw):{kind:sum(v['chosen']==kind for v in records if v['draw']==draw) for kind in n.KINDS} for draw in range(3)},'predictions_sha256':e.sha(out/'predictions.npz'),'block_reports_sha256':{f'draw-{v["draw"]}-fold-{v["fold"]}-block-{v["block"]}/report.json':e.sha(out/f'draw-{v["draw"]}-fold-{v["fold"]}-block-{v["block"]}/report.json') for v in records},'seconds':time.perf_counter()-start};e.write(out/'report.json',report);check();e.emit(report['draw_RMSE'])

def validate(split):
 check();d=r.load(split);out=_repo_path(OUT, split);report=e.read(out/'report.json');assert e.sha(out/'predictions.npz')==report['predictions_sha256'];cor=np.full((3,3,400),np.nan);count=0;maxerr=0.;start=time.perf_counter()
 for draw in range(3):
  for k in range(5):
   for j in range(3):
    A,whole,pa,trace=r.anchor(d,split,k,j);B=subset(d,whole,split,k,j,draw);res=d['y'][B]-pa[np.isin(whole,B)];path=out/f'draw-{draw}-fold-{k}-block-{j}';rec=e.read(path/'report.json');assert e.sha(path/'report.json')==report['block_reports_sha256'][f'draw-{draw}-fold-{k}-block-{j}/report.json'];assert e.sha(path/'predictions.npz')==rec['predictions_sha256'] and trace==rec['anchor_trace']
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:saved={n:z[n].copy() for n in z.files}
    np.testing.assert_array_equal(saved['block_indices'],B);np.testing.assert_array_equal(saved['residual'],res);ip=np.full((5,len(B)),np.nan);folds=list(GroupKFold(3,shuffle=True,random_state=r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
    for t,(it,iv) in enumerate(folds):
     assert rec['inner_splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()} and not set(d['g'][B[it]])&set(d['g'][B[iv]])
     for ci,kind in enumerate(n.KINDS):
      file=path/f'inner-{t}-{kind}.joblib';assert e.sha(file)==rec['files_sha256'][file.name];s=joblib.load(_repo_io(file));np.testing.assert_array_equal(s['train_indices'],B[it]);np.testing.assert_array_equal(s['validation_indices'],B[iv]);m=s['model'];assert m['kind']==kind;pp=independent.manual(m,d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);np.testing.assert_array_equal(n.predict(m,d['X'][B[iv]]),saved['inner_predictions'][ci,iv]);err=float(abs(pp-saved['inner_predictions'][ci,iv]).max());assert err<2e-9;maxerr=max(maxerr,err);ip[ci,iv]=pp;count+=1
    scores=np.mean((ip-res[None,:])**2,1);np.testing.assert_allclose(scores,list(rec['inner_MSE'].values()),atol=2e-9,rtol=0);kind=n.KINDS[int(scores.argmin())];assert kind==rec['chosen'];assert e.sha(path/'selected.joblib')==rec['selected_sha256'];s=joblib.load(_repo_io(path/'selected.joblib'));np.testing.assert_array_equal(s['train_indices'],B);va=np.flatnonzero(d['fid']==k);pp=independent.manual(s['model'],d['X'][B],res,d['X'][va],d['blocks']);np.testing.assert_allclose(pp,saved['correction'],atol=2e-9,rtol=0);np.testing.assert_array_equal(n.predict(s['model'],d['X'][va]),saved['correction']);cor[draw,j,va]=saved['correction'];count+=1
 with np.load(_repo_io(out/'predictions.npz'),allow_pickle=False) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(cor,z['corrections']);np.testing.assert_array_equal(d['reference011'][None,:]+cor.mean(1),z['predictions'])
 v={'passed':True,'split':split,'models_reloaded':count,'inner_choices_recomputed':45,'max_independent_error':maxerr,'subset_and_anchor_exclusions_checked':True,'report_sha256':e.sha(out/'report.json'),'seconds':time.perf_counter()-start};e.write(_repo_path(OUT, f'validation-{split}.json'),v);check();e.emit(v)

def leakage():
 check();d=r.load('original');alter=dict(d);alter['y']=d['y'].copy();ix=np.flatnonzero(d['fid']==0);alter['y'][ix]=9999+np.arange(len(ix));out=_repo_path(OUT, 'leakage-replay');out.mkdir(exist_ok=False)
 for draw in range(3):
  for j in range(3):
   pp,rec=block(alter,'original',0,j,draw,out/f'draw-{draw}-block-{j}');old=_repo_path(OUT, 'original', f'draw-{draw}-fold-0-block-{j}');saved=e.read(old/'report.json');assert rec['chosen']==saved['chosen'] and rec['inner_MSE']==saved['inner_MSE']
   with np.load(_repo_io(old/'predictions.npz'),allow_pickle=False) as z:np.testing.assert_array_equal(pp,z['correction'])
 v={'passed':True,'outer80_labels_perturbed_9blocks':True,'predictions_choices_exact':True,'upstream_certificate':e.sha(e.OUT/'validation-replay.json')};e.write(_repo_path(OUT, 'leakage-validation.json'),v);e.emit(v)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','validate','leakage']);p.add_argument('--split',choices=e.SPLITS);x=p.parse_args();{'prepare':prepare,'leakage':leakage}.get(x.mode,lambda:(run if x.mode=='run' else validate)(x.split))()
