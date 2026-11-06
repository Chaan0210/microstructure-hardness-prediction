"""One regularized simplex-weight learner inside the existing014 nested lineage."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,itertools
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import candidate014_pipeline as protected
import numpy as np

a=protected.a;e=protected.e;n=protected.n;r=protected.r;OUT=_repo_path(ROOT, 'artifacts/constrained-weights-125');PRIOR=np.array([.3,.3,.2,.2]);LAMBDA=1.;COLS=('kernel','cnn','Smean','Bmean')

def fit_weights(P,y):
 P=np.asarray(P,float);y=np.asarray(y,float);assert P.shape==(len(y),4);mean=P.mean(0);ym=float(y.mean());ys=float(y.std());assert ys>0;Z=(P-mean)/ys;t=(y-ym)/ys;H=Z.T@Z/len(y)+LAMBDA*np.eye(4);b=Z.T@t/len(y)+LAMBDA*PRIOR;candidates=[]
 for bits in range(1,16):
  ix=np.flatnonzero([(bits>>j)&1 for j in range(4)]);K=np.block([[H[np.ix_(ix,ix)],np.ones((len(ix),1))],[np.ones((1,len(ix))),np.zeros((1,1))]]);sol=np.linalg.solve(K,np.r_[b[ix],1.]);w=np.zeros(4);w[ix]=sol[:-1]
  if w.min()>=-1e-12:
   w=np.maximum(w,0);w/=w.sum();loss=float(np.mean((Z@w-t)**2)+LAMBDA*np.sum((w-PRIOR)**2));candidates.append((loss,bits,w))
 loss,bits,w=min(candidates,key=lambda q:(q[0],q[1]));g=2*(H@w-b);nu=-float(g[w>1e-10].mean());kkt=max(float(abs(g[w>1e-10]+nu).max()),float(max(0.,-(g[w<=1e-10]+nu).min())) if np.any(w<=1e-10) else 0.);assert w.min()>=0 and abs(w.sum()-1)<1e-12 and kkt<1e-9
 return {'weights':w.tolist(),'prior':PRIOR.tolist(),'lambda':LAMBDA,'prediction_mean':mean.tolist(),'target_mean':ym,'target_std':ys,'n_fit':len(y),'objective':loss,'prior_objective':float(np.mean((Z@PRIOR-t)**2)),'KKT_max_error':kkt,'active_set_mask':bits}

def meta(P,y,Q):
 m=fit_weights(P[:,1:],y);w=np.array(m['weights']);pred,cal=a.affine(P[:,1:]@w,y,Q[:,1:]@w);return pred,{'weight_model':m,'calibrator':cal}

def load(split):
 d=a.load(split);path=protected.study.OUT/split;rec=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',rec['predictions_sha256']);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);d['reference014']=z['no_baseline795'];return d

def prepare():
 protected.check();OUT.mkdir(exist_ok=False);diag=e.read(_repo_path(ROOT, 'artifacts/baseline-diagnostics-124/report.json'));assert diag['verification']['trainlocal_scalers_verified_all30Ridges'];p={'study':'125 regularized4way simplex014weights','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duplicate_audit':{'074':'Oneaffineonfixedaggregate;noindividualweightlearning.','077':'UnregularizedOLS legacy003-vs-equalS/Bcontrast;not4remainingcomponentssimplex. Original00913.105158->12.902927butresplitfailureand078size-mismatchwarningretained.','079':'Inner3->5affinecrossfit-only;notweights.','112':'Oneboundedalpha betweencomplete011 andnewattentionbranch;no4way014baseweights orpriorpenalty.','125_distinction':'Same014surviving4branches,nonnegativeunit-sumweightswithfixedridge-to-prior,noattentionornewbaselearner. Refitaffineandhonestphysicsresiduals afterlearnedmixture.'},'hypothesis':'014inheritedrenormalized.3KRR/.3CNN/.2S/.2B isnotoptimizedforremainingbranches. A smallregularizedtrain-onlyweightadjustmentmayimproveconditionalcomplementarity whileboundingcollinearweights.','candidates':list(COLS),'fixed_short_head':'.1Bshort unchanged;.9newaffine(mixtureremaining4). LearningshortBweightwouldrequiredeepcrossfitshortBpredictionsnotavailableinexistingdeep008caches;donotusein-sampleorwrong-scopeproxy.','objective':'min_w mean(((P-columnmean)/std(y) @ w -(y-mean(y))/std(y))^2) + 1*sum((w-[.3,.3,.2,.2])^2); w>=0,sumw=1. Allstatistics trainmeta rows only. Columncenteringonly,nopercomponentvarianceequalization. Lambda1 fixed,no grid/innerselector. Thenfitexistingaffineon sameOOFmixture;wholeproceduretestedouter.','solver':'Enumerate15nonemptyactivesets,solveequalityQPexactly,choosefeasibleminimum;strictconvexlambdaI makesunique. IndependentscipySLSQPandKKTverification.','nested_scope':'Outer320weights+affine use074componentOOF from213/214 fits. EachresidualanchorA213/214 weights+affine use112deepcomponentOOF from142/143 fits;applytoheldoutB106/107 componentpredictions fromA. Fit115residualselectoronlyB,anchorAexcludesallB. NeverfitweightsonglobalouterOOF. CacheIDs/hashesfrom121/122reverified.','control':'Fixed014wholepipeline preserved. Reconstructallfixedweightaffine/anchors exactlywithin1e-10 andrefitonefull014outerfoldascontrol;newlearnerusesidenticalbasepredictions/residualchoices/CVseeds.','comparison':'Runexactlyonelearnedprocedureversusfixed014acrossall3partitions;pooleduntrimmedMSEprimary. No allpartition/fold/CIgate,lambda/weightsearch orposthocblend. Headweights/stability/conditionalCI diagnostic.','limits':['Same400adaptivehistory;124diagnosticand122results motivatedthisstudy.','Training-size mismatch142->213->320 andfinal222->333->500 remains;constraintsdonotproveweighttransport.','Weightobjectivefitscenteredunscaledmixturebeforeaffine;bestcenteredrawMSEneednotgivebestcalibrated/residualpipeline.','Fourweights3freeparametersplusaffine2;regularizationlimitsinstabilitybutcannotcreateindependentlabels.','Thisassessesonefixedlambda1;failuredoesnotjustifyautomaticlambdaorbasecandidateexpansion.'],'resource':'CPU2existingpackages;60smallQP/affinefits,45residualblocks720serializedmodelsincludingnone. No newneuralfit.','sources':{x:e.sha(_repo_path(ROOT, x)) for x in ['constrained_weights_125.py','validate_constrained_weights_125.py','component_ablation_121.py','component_single_ablation_122.py']},'inputs':{str(q.relative_to(ROOT)):e.sha(q) for q in [_repo_path(ROOT, 'artifacts/baseline-diagnostics-124/report.json'),_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json'),*[(protected.study.OUT/s/'predictions.npz') for s in e.SPLITS]]},'protected014':{str(q.relative_to(ROOT)):e.sha(q) for q in [protected.CSV,protected.CSV.with_suffix('.manifest.json'),protected.OUT/'decision.json',protected.OUT/'fit.json',_repo_path(ROOT, 'reports/candidate014-preservation.json')]},'cost':0,'scope':'Development400only;no test/holdoutfit ordecoding,nonewinstall,014preserved,noupload.'};e.write(_repo_path(OUT, 'plan.json'),p);e.emit({'prepared':True,'lambda':LAMBDA,'prior':PRIOR.tolist()})

def check():
 protected.check();p=e.read(_repo_path(OUT, 'plan.json'))
 for name,h in {**p['sources'],**p['inputs'],**p['protected014']}.items():assert e.sha(_repo_path(ROOT, name))==h,name
 return p

def fold(d,k,path):
 path.mkdir(exist_ok=False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);cal,model=meta(d['middleP'][k],d['y'][T],d['P'][V]);model['train_indices']=T.tolist();model['validation_indices']=V.tolist();base=.9*cal+.1*d['short'][V];cor=[];models=[];records=[]
 for j in range(3):
  A,B,P,short,_=d['cells'][k,j];cp,m=meta(d['deepP'][k,j],d['y'][A],P);m['train_indices']=A.tolist();m['validation_indices']=B.tolist();anchor=.9*cp+.1*short;cc,rec=a.fit_block(d,B,anchor,k,j,path/f'block-{j}');cor.append(cc);models.append(m);records.append(rec)
 with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),outer_indices=V,base=base,corrections=np.stack(cor),primary=base+np.mean(cor,0))
 rec={'fold':k,'outer_meta':model,'anchor_meta':models,'selected':[v['chosen'] for v in records],'block_reports_sha256':{f'block-{j}/report.json':e.sha(path/f'block-{j}/report.json') for j in range(3)},'predictions_sha256':e.sha(path/'predictions.npz')};e.write(path/'report.json',rec);return base+np.mean(cor,0),rec

def run(split):
 check();assert e.read(_repo_path(OUT, 'fixture-control-validation.json'))['passed'];d=load(split);out=_repo_path(OUT, split);out.mkdir(exist_ok=False);primary=np.full(400,np.nan);records=[];t=time.perf_counter()
 for k in range(5):
  V=np.flatnonzero(d['fid']==k);primary[V],rec=fold(d,k,out/f'fold-{k}');records.append(rec);e.emit({'split':split,'outer':k,'weights':rec['outer_meta']['weight_model']['weights']})
 with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference014=d['reference014'],primary=primary)
 report={'split':split,'reference014_RMSE':e.rmse(d['y'],d['reference014']),'primary_RMSE':e.rmse(d['y'],primary),'weights':[z['outer_meta']['weight_model']['weights'] for z in records],'fold_reports_sha256':{f'fold-{k}/report.json':e.sha(out/f'fold-{k}/report.json') for k in range(5)},'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-t};e.write(out/'report.json',report);check();e.emit(report)
if __name__=='__main__':
 protected.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare',*e.SPLITS]);x=p.parse_args();prepare() if x.mode=='prepare' else run(x.mode)
