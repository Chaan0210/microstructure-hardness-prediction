"""Reuse existing fixed3seed outer CNNs; fill only missing nested014 CNN means."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,gc,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mse_readout_132 as old
import numpy as np,torch
s=old.s;e=old.e;a=old.a;q=old.q;cal=old.cal;fv=cal.fv;OUT=_repo_path(ROOT, 'artifacts/cnn-seed-mean-133');SEEDS=(20261003,20261023,20261043);ARM=old.ARM

def data(split):return old.data(split)
def ckpath(split,k,seed):
 if split=='original':return _repo_path(ROOT, 'artifacts/run-009', f'fold-{k}.pt') if seed==SEEDS[0] else _repo_path(ROOT, 'artifacts/run-018', f'cnn-seed{seed}-fold{k}.pt')
 return _repo_path(ROOT, 'artifacts/frozen-validation-021', f'seed-{split}', f'fold-{k}', f'cnn-seed{seed}.pt')
def check():
 p=e.read(_repo_path(OUT, 'plan.json'))
 for f,h in p['hashes'].items():assert e.sha(_repo_path(ROOT, f))==h,f
 return p

def prepare():
 old.check();s.previous.check();OUT.mkdir(exist_ok=False);names=['cnn_seed_mean_133.py','frozen_validation.py','cnn_controls.py','cnn_multiseed.py','frozen_multiseed.py','component_ablation_121.py','reports/mse-readout-132-summary.json','reports/mse-readout-132-preservation.json','reports/candidate014-summary.json','reports/candidate014-verification.json','artifacts/submission-candidate-014.csv','artifacts/run-018/plan.json','artifacts/run-018/report.json','artifacts/run-018/development_oof.csv','artifacts/frozen-validation-021/plan.json','artifacts/frozen-validation-021/report.json','artifacts/nested-calibration-074/development_features.npz']
 for split in e.SPLITS:
  if split!='original':names.extend([f'artifacts/frozen-validation-021/seed-{split}/report.json',f'artifacts/frozen-validation-021/seed-{split}/development_oof.csv'])
  for k in range(5):
   for seed in SEEDS:names.append(str(ckpath(split,k,seed).relative_to(ROOT)))
 p={'study':'133 fixedsame014CNN three-seed average','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duplicate_audit':'018 original and021 twoexistingrepartitions already trained exactlythreebase seeds20261003/23/43 withsame014CNN80/last5average. Reuseall45outerCNNs. Priorfusionwas003baseline50/KRR25/CNN25,not014DINO/affine/honestphysics. Currentraw4waycached gains original+.112989139,split1+.037192789,split2-.025199414: descriptive,notcomplete014. No claim seedaveraging isnew orpreviousfull014tested.','fixed_seeds':list(SEEDS),'not_seed_selection':'Samehistoricalbundle,allthreeequalweights;no bestseed,subsetsearch,newseed,learnedweight,D4inferenceorRidge. Individualseed scoresdiagnosticonly.','primary':'ChangeCNNcomponent frombase20261003+offset to equalmean3 predictions. SameTinyCNN128,80epochs,batch32,AdamW.002->.0001wd.01,clip5,MSE,batchD4training,identityinference,meanlast5weights separatelywithinseed.','reuse':'45outer checkpoints across3partitions;alloldsingle-seed middle/deepCNNpredictions from074/112 remainunchanged. Train onlymissingtwo seedmodels perinnernode. Original60nodes=>120newCNNfits. No duplicateoutertraining apartfromoneexactreplaycheck.','nested':'Exactly014 weights .3KRR/.3CNNmean3/.2S/.2B;refitouteraffine from3middleOOF predictions;for eachhonestresidualblock B useA-only anchor-affinefitfrom3deepOOF;retainshortB10%;refitphysics57none/Ridge1/.1/RBF1/.1selectorwithinB. Mean3corrections. Allfeature/targetnormalizationtrainonly;nonewglobalOOFresidualtraining.','offsets':'outerk,middlek+100(j+1),deepk+100(j+1)+100(ell+1),same074/112foldmembership. Seedbase+offset.','gate':'Originalcompletepipelinefirst. IfnotstrictlylowerRMSEthan014afterindependentvalidation,terminate;no rescue seed/weights. Ifpositive,unchangedtwoexistingrepartitions. No newproductionfull500/test/submissionin133 evenifpositive.','prior_sufficient_scope':'Existingexperiments sufficientlycoverouterCNNthree-seed fits andseedvariance; do notrepeat. Missingstrictnested014fusion isdistinctminimumverification requestedhere.','validation':'Reloadall45outerCNNs CPU/MPS,trainIDs/groups/stats,compareoldcurrentseedweights andcachedpredictions. Onefreshadditional-seedouterfitexactoldcheckpointreplay. AllnewinnerCNN reloads;independentOLS andallphysicsmodels; outerlabelpoisononecompletefold,innerBpoisononecompleteanchor.','resource':'CPU2/MPS,firstnewnestedouterreusedbenchmark5fold<1800sec,RSS/MPSdriver<3GB. Threefixedseedsonly,cost0,noinstall. Stopresourcefailure,neverpaidalternative.','bias':'Same400adaptivehistory;historical3seedbundleandcurrent133hypothesisarenotindependentselection. ThreepartitionsoverlapsameIDs. No seedaveragingguarantee forwholeensemble.','submission_status_from_parent':'013/014 unsubmitted;filechooser10sectimeout/Maclocked,site15:22shows0/5. No submissionactionauthorizedforthisstudy.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'cost':0};s.write(_repo_path(OUT, 'plan.json'),p);e.emit({'prepared':True,'reuse_outer_CNNs':45,'new_seeds':0,'fixed_seeds':SEEDS})

def predict(obj,d,V,device):
 m=s.engine.TinyCNN().to(device);m.load_state_dict(obj['state_dict']);p=s.old.prior.predict(m,d['images'],V,obj['normalization'],device);del m;return p

def audit():
 check();out=_repo_path(OUT, 'audit');out.mkdir();results={};checks=[];start=time.perf_counter()
 for split in e.SPLITS:
  d=data(split);p=_repo_path(ROOT, 'artifacts/run-018/development_oof.csv') if split=='original' else _repo_path(ROOT, 'artifacts/frozen-validation-021', f'seed-{split}/development_oof.csv')
  with p.open() as f:rows={r['ID']:r for r in csv.DictReader(f)}
  assert set(rows)==set(d['ids']);np.testing.assert_array_equal([float(rows[i]['hardness']) for i in d['ids']],d['y']);keys=[f'cnn_seed{seed}'+('' if split=='original' else '_identity') for seed in SEEDS];P=np.array([[float(rows[i][key]) for i in d['ids']] for key in keys]);np.testing.assert_allclose(P[0],d['P'][:,2],atol=1e-10,rtol=0)
  if split!='original':np.testing.assert_array_equal([int(rows[i]['fold']) for i in d['ids']],d['fid'])
  for k in range(5):
   T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert not set(d['g'][T])&set(d['g'][V]);current=torch.load(_repo_io(s.old.prior.q.spec(split,k,2,0)['paths']['cnn']),map_location='cpu',weights_only=True)
   for si,seed in enumerate(SEEDS):
    path=ckpath(split,k,seed);obj=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in T];assert obj.get('validation_ids',[d['ids'][i] for i in V])==[d['ids'][i] for i in V];norm=obj['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};assert obj.get('seed',seed+k)==seed+k
    if si==0:assert cal.h.state_hash(obj['state_dict'])==cal.h.state_hash(current['state_dict'])
    if split!='original':fr=e.read(path.parent.parent/f'fold-{k}-report.json');assert e.sha(path)==fr['checkpoint_sha256'][path.name]
    errs={device:float(abs(predict(obj,d,V,device)-P[si,V]).max()) for device in ['mps','cpu']};assert max(errs.values())<1e-3;checks.append({'split':split,'fold':k,'seed':seed,'checkpoint':str(path.relative_to(ROOT)),'sha256':e.sha(path),'reload_errors':errs,'trainlocal_stats_groups_and_seed_verified':True})
  avg=P.mean(0);raw=.3*d['P'][:,1]+.2*d['P'][:,3]+.2*d['P'][:,4];error=P-d['y'];variance=np.var(P,axis=0);lhs=float(np.mean(error**2)-np.mean((avg-d['y'])**2));rhs=float(variance.mean());assert abs(lhs-rhs)<1e-10
  results[split]={'CNN_seed_RMSE':[e.rmse(d['y'],x) for x in P],'CNN_mean3_RMSE':e.rmse(d['y'],avg),'raw_first_RMSE':e.rmse(d['y'],raw+.3*P[0]),'raw_mean3_RMSE':e.rmse(d['y'],raw+.3*avg),'mean_seed_prediction_variance':rhs,'mean_individual_MSE_minus_mean3_MSE':lhs,'seed_residual_correlation':np.corrcoef(error).tolist()}
  with (out/f'{split}.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],seeds=P,mean3=avg)
  e.emit({'audit_split':split,**results[split]})
 d=data('original');T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);prev=fv.CNN_SEED;fv.CNN_SEED=SEEDS[1]
 try:p,r=fv.cnn_fold(d['images'],d['y'],T,V,d['ids'],0,'mps',out/'replay.pt')
 finally:fv.CNN_SEED=prev
 obj=torch.load(_repo_io(out/'replay.pt'),map_location='cpu',weights_only=True);ref=torch.load(_repo_io(ckpath('original',0,SEEDS[1])),map_location='cpu',weights_only=True);assert cal.h.state_hash(obj['state_dict'])==cal.h.state_hash(ref['state_dict'])
 with np.load(_repo_io(out/'original.npz')) as z:np.testing.assert_array_equal(p,z['seeds'][1,V])
 report={'passed':True,'results':results,'checkpoint_checks':checks,'exact_fresh_seed20261023_replay':True,'new_CNN_fits':1,'seconds':time.perf_counter()-start,'source_sha256':e.sha(Path(__file__))};s.write(out/'report.json',report);check();e.emit({'cache_audit_passed':True,'models_reused':45,'exact_replay':True})

def node(d,T,V,offset,oldpred,path,train=True):
 if train:
  path.mkdir();pred=[oldpred];records=[]
  for seed in SEEDS[1:]:
   prev=fv.CNN_SEED;fv.CNN_SEED=seed;start=time.perf_counter()
   try:pp,r=fv.cnn_fold(d['images'],d['y'],T,V,d['ids'],offset,'mps',path/f'seed-{seed}.pt')
   finally:fv.CNN_SEED=prev
   pred.append(pp);records.append({'base_seed':seed,'effective_seed':seed+offset,'seconds':time.perf_counter()-start,'train_RMSE':r['train_rmse'],'curve':r['curve'],'checkpoint_sha256':e.sha(path/f'seed-{seed}.pt')})
  P=np.stack(pred);mean=P.mean(0)
  with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),train=T,validation=V,seeds=P,mean3=mean)
  r={'offset':offset,'train':T.tolist(),'validation':V.tolist(),'models':records,'predictions_sha256':e.sha(path/'predictions.npz')};s.write(path/'report.json',r)
 else:
  r=e.read(path/'report.json');assert r['train']==T.tolist() and r['validation']==V.tolist();z=a.readnp(path/'predictions.npz',r['predictions_sha256']);np.testing.assert_array_equal(z['seeds'][0],oldpred);mean=z['mean3'];np.testing.assert_array_equal(z['seeds'].mean(0),mean)
 return mean

def replace(d,split,k,root,train=True):
 T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);new=dict(d);new['P']=d['P'].copy();new['middleP']=dict(d['middleP']);new['middleP'][k]=d['middleP'][k].copy();new['deepP']=dict(d['deepP']);new['cells']=dict(d['cells']);nodes=[]
 with np.load(_repo_path(OUT, 'audit', f'{split}.npz')) as z:assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['fold'],d['fid']);new['P'][V,2]=z['mean3'][V]
 look={int(i):j for j,i in enumerate(T)}
 for j in range(3):
  A,B,P,short,ref=d['cells'][k,j];path=root/f'middle-{j}';mp=node(d,A,B,k+100*(j+1),P[:,2],path,train);nodes.append(path);Q=P.copy();Q[:,2]=mp;new['cells'][k,j]=(A,B,Q,short,ref);new['middleP'][k][[look[int(i)] for i in B],2]=mp;new['deepP'][k,j]=d['deepP'][k,j].copy();la={int(i):ii for ii,i in enumerate(A)}
  for ell in range(3):
   source=e.OUT/split/f'fold-{k}/inner-{j}/deep-{ell}';C,D,DP=a.bundle(source);assert set(C)|set(D)==set(A) and not(set(C)|set(D))&(set(B)|set(V));path=root/f'deep-{j}-{ell}';pp=node(d,C,D,k+100*(j+1)+100*(ell+1),DP[:,2],path,train);nodes.append(path);new['deepP'][k,j][[la[int(i)] for i in D],2]=pp
  e.emit({'split':split,'outer':k,'nested_block_complete':j,'training':train})
 return new,nodes

def run(split):
 check();assert e.read(_repo_path(OUT, 'audit/report.json'))['passed']
 if split!='original':assert e.read(_repo_path(OUT, 'screen-decision.json'))['advance']
 d=data(split);path=_repo_path(OUT, split);path.mkdir();start=time.perf_counter();primary=np.full(400,np.nan);base=primary.copy();raw=primary.copy();records=[];nodes=[]
 for k in range(5):
  ft=time.perf_counter();root=path/f'cnn-fold-{k}';root.mkdir();dd,nn=replace(d,split,k,root);nodes.extend([str(p.relative_to(ROOT)) for p in nn]);V=np.flatnonzero(d['fid']==k);primary[V],base[V],rec=a.fold(dd,k,ARM,path/f'pipeline-fold-{k}');raw[V]=a.mixture(dd['P'][V],ARM);records.append(rec)
  if split=='original':cp,_,_=a.fold(d,k,ARM,path/f'control014-fold-{k}');np.testing.assert_allclose(cp,d['reference014'][V],atol=1e-10,rtol=0)
  e.emit({'split':split,'outer_complete':k,'elapsed_seconds':time.perf_counter()-start})
  if k==0:
   b={'first_outer_seconds_reused':time.perf_counter()-ft,'projected5outer_seconds':5*(time.perf_counter()-ft),'memory':s.engine.memory('mps')};b['passed']=b['projected5outer_seconds']<1800 and b['memory']['process_peak_rss_bytes']<3e9 and b['memory']['driver_allocated_memory']<3e9;s.write(path/'benchmark.json',b);assert b['passed']
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference014=d['reference014'],primary=primary,base=base,raw=raw)
 r={'split':split,'RMSE':{'reference014':e.rmse(d['y'],d['reference014']),'primary':e.rmse(d['y'],primary),'base':e.rmse(d['y'],base),'raw':e.rmse(d['y'],raw)},'records':records,'nodes':nodes,'new_CNN_fits':2*len(nodes),'reused_outer_CNNs':15,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};s.write(path/'report.json',r);check();e.emit({'split':split,'RMSE':r['RMSE'],'seconds':r['seconds']})
if __name__=='__main__':
 torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit',*e.SPLITS]);x=p.parse_args();prepare() if x.mode=='prepare' else audit() if x.mode=='audit' else run(x.mode)
