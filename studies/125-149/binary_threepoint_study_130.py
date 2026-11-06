"""Nested additional binary spatial residuals while retaining complete014."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,time,argparse,datetime,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import binary_threepoint_preflight_v2_130 as pre
import binary_threepoint_features_130 as feature
import cnn_orientation_129 as previous
import constrained_weights_125 as loader
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
q=loader;a=q.a;e=q.e;n=q.n;r=q.r;OUT=pre.OUT
ARMS=('pair','pair_triple');LAMS=(1.,.1);CHOICES=((None,0.),(1.,.5),(1.,1.),(.1,.5),(.1,1.));LABELS=('none','ridge1_half','ridge1_full','ridge01_half','ridge01_full')

def check():
    previous.check();pre.check();p=e.read(OUT/'model-plan.json')
    for file,h in {**p['sources'],**p['inputs'],**p['protected']}.items():assert e.sha(_repo_path(ROOT, file))==h,file
    return p

def prepare():
    previous.check();pre.check();assert e.read(OUT/'benchmark.json')['passed'];inputs={}
    for f in [OUT/'feature-plan.json',OUT/'extraction.json',OUT/'development-features.npz',_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/component-single-ablation-122-summary.json')]:inputs[str(f.relative_to(ROOT))]=e.sha(f)
    for split in e.SPLITS:
        for k in range(5):
            path=q.protected.study.OUT/split/f'no_baseline795-fold-{k}'
            for file in path.rglob('*'):
                if file.is_file():inputs[str(file.relative_to(ROOT))]=e.sha(file)
    p={'study':'130 fixed binary2point versus2+3point additional residual control','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':['unchanged014',*ARMS],'added_bank_dimensions':{'pair':114,'pair_triple':186},'base_retained':'Exactly014: raw.3KRR+.3CNN+.2S+.2B,trainlocalaffine90%+shortB10%,existingmean3selectedphysics57residuals. Newbank corrections added after014, no replacing provenphysics57 orreusing014globalOOFasresidualtraininglabels.','honest_anchor':'ForouterT320/V80 useexisting3A213/214,B106/107 partitions. Recompute014prephysicsanchor onB withdeepA-onlycalibrator andB-excludedbaseheads. Define r_B=y_B-anchor_A(B). A excludesallB andV. Existing014physicsmodel m_B selectedinsideB asbefore.','strict_extra_nesting':'Selectnewlambda/gamma by3foldwithinB (sameCVSEED+100*k+j as014). For I=BminusH, refitphysicsselector insideI, predictH. To obtain newRidge targetswithinI, splitI into3; for eachJ/H2, selectphysicsmodel using3foldWITHINJ, refitonJ,predictH2. Newtargets=r_I-crossfit_selected_physics_I. Thus allnewtrainingtargets andphysicschoices excludeouterH. FullBnewRidge usesr_B minuscrossfitphysicsselectedwithinits3I; fullBphysicsm_B remainsoriginal014selectedmodel. No shortcut ofgloballyselectedphysicsOOF orfullBphysics predictionsinnewCV.', 'formula':'014_outer320(V)+mean_j gamma_j*g_j(V). ForinternalCVcompare r_H to m_I(H)+gamma*g_I(H). Gamma0 yieldsnoadditionalcorrection andexact014. Differentbank armsselectonlytheirownlambda/gammainsideB; existingphysicschoicesidenticalacrossarms.','regularization':{'lambdas':list(LAMS),'gammas':[0,.5,1],'candidate_tie_order':list(LABELS),'alpha':'lambda*n_train','regressor':'LinearRidge nointercept,trainingresidualRMSnormalization,dualn-by-nsolve; featuremean/stdRMSfloor perbank trainonly,1/sqrt(bankwidth). NoPCA,clip,hyperparametersweeporouterweightfit.'},'bank_immutability':'Fixedbeforeanynewhardnessscores. No quantile/triangle/radius/direction/feature filtering fromlabels. Pairfullvalidoverlapdiffersfromtrianglecommondomain,explicitfeatureplan.','screen':'Original400 first. AdvancebothknownresplitsONLYifpair_tripleuntrimmedMSEstrictlybelowBOTH014andpair,andindependentvalidationpasses. Ifitdoesnotbeatpairterminatebank; noalternatequantiles/motifs/weights. Pairaloneiscontrolnotfallbackcandidate.','confirmation':'Ifscreenpasses,identicaltworepartitions; all3mustbeat014andpair tojustifyfuturedeploymentwork. No full500/test/inference/CSV/submission/zeroing in130 evenifpositive.','uncertainty':'10000paired400-IDbootstrapseed20271113. RepartitionsclusteredbysameID. Conditionalfixedfituncertainty,not independentconfirmation oradjustmentforhistoricalselection.','training_size_limit':'BaseanchorsA213 versusouter320;physicscrossfitstage71/47/31 versusfull106. Honestbutdifferenttrainingamounts; crossfitresidualdistributionmaydiffer. Zerooptiontestsutilityofwholefixedprocedure.','validation':'Everyserializedphysicscandidate independentlyrecomputedviaexistingrawstat/eigensolvevalidator; everynewRidge independentlySVD andrawscalercheck; allsplits/groupsexclusion; allchoicesandformula. Reconstruct014predictions<1e-10. Poisonouter80targets forfullfold; poisononeinternalH tocheckphysicsselection/newRidgefitexclusion; same upstream112trainingcertificates reused,nonewneuralfits.','resources':'ExistingCPU2,0newneuralfits. Original15honestblocks. Firstblockreusedbenchmark projected15<900sec,RSS<3GB; ifpositive45totalblocks. Feature20pilotpassed. No externaltrainingdata,package,cost,holdout/testdecoding.','sources':{name:e.sha(_repo_path(ROOT, name)) for name in ['binary_threepoint_study_130.py','validate_binary_threepoint_130.py','binary_threepoint_features_130.py','nonlinear_residual_115.py','validate_nonlinear_residual_115.py']},'inputs':inputs,'protected':{name:e.sha(_repo_path(ROOT, name)) for name in ['artifacts/submission-candidate-014.csv','reports/cnn-orientation-129-summary.json','reports/cnn-orientation-129-preservation.json','reports/readme-update-129.json']},'initial_serialization_failure':'Originalpreflightfile,partialfeature-plan,andlogsretained. Separatev2onlycastsNumPyint metadata and serializesJSONbeforeopeningfile. Nofeatureformula change.','cost':0};e.write(OUT/'model-plan.json',p);e.emit({'model_plan_fixed':True,'sha256':e.sha(OUT/'model-plan.json'),'gate':p['screen']})

def load(split):
    d=q.load(split);ex=e.read(OUT/'extraction.json');assert e.sha(OUT/'development-features.npz')==ex['cache_sha256']
    with np.load(_repo_io(OUT/'development-features.npz')) as z:assert z['ids'].tolist()==d['ids'];d['bank']=z['X'].copy()
    return d

def partitions(d,I,seed):
    out=[]
    for ti,vi in GroupKFold(3,shuffle=True,random_state=seed).split(np.zeros((len(I),1)),groups=d['g'][I]):
        A=I[ti];B=I[vi];assert not set(A)&set(B) and not set(d['g'][A])&set(d['g'][B]);out.append((A,B))
    return out

def ridge(X,y,blocks,lam):
    sc=feature.scaling(X,blocks);Z=feature.transform(X,sc);rs=float(np.sqrt(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;alpha=lam*len(y);dual=np.linalg.solve(Z@Z.T+alpha*np.eye(len(y)),y/rs);coef=Z.T@dual
    return {'scaler':sc,'coef':coef,'residual_scale':rs,'alpha':alpha,'lambda':lam,'n_train':len(y),'blocks':blocks}
def ridge_predict(m,X):return feature.transform(X,m['scaler'])@m['coef']*m['residual_scale']
def panel(d,I,arm):return d['bank'][I,:114 if arm=='pair' else 186]
def blocks(arm):return ['pair']*114+([] if arm=='pair' else ['triple']*72)

class Nested:
    def __init__(self,d,res):self.d=d;self.res=res;self.nodes={};self.stages={}
    def select(self,I,seed):
        key=(tuple(I.tolist()),seed)
        if key in self.nodes:return self.nodes[key]
        ip=np.full((5,len(I)),np.nan);lookup={int(i):j for j,i in enumerate(I)};fits=[]
        for t,(T,V) in enumerate(partitions(self.d,I,seed)):
            vi=[lookup[int(i)] for i in V]
            for ci,kind in enumerate(n.KINDS):
                m=n.fit(self.d['X'][T],self.res[T],self.d['blocks'],kind);pp=n.predict(m,self.d['X'][V]);ip[ci,vi]=pp;fits.append({'train':T,'validation':V,'kind':kind,'model':m,'predictions':pp})
        scores=np.mean((ip-self.res[I])**2,axis=1);chosen=n.KINDS[int(scores.argmin())];model=n.fit(self.d['X'][I],self.res[I],self.d['blocks'],chosen);node={'id':len(self.nodes),'train':I,'seed':seed,'fits':fits,'inner_predictions':ip,'MSE':scores,'chosen':chosen,'model':model};self.nodes[key]=node;return node
    def stage(self,I,seed):
        key=(tuple(I.tolist()),seed)
        if key in self.stages:return self.stages[key]
        node=self.select(I,seed);cross=np.full(len(I),np.nan);lookup={int(i):j for j,i in enumerate(I)};trace=[]
        for t,(T,V) in enumerate(partitions(self.d,I,seed)):
            inner=self.select(T,seed+1000+t);cross[[lookup[int(i)] for i in V]]=n.predict(inner['model'],self.d['X'][V]);trace.append({'node':inner['id'],'validation':V})
        target=self.res[I]-cross;models={arm:{str(lam):ridge(panel(self.d,I,arm),target,blocks(arm),lam) for lam in LAMS} for arm in ARMS};stage={'id':len(self.stages),'train':I,'seed':seed,'physics_node':node['id'],'crossfit_trace':trace,'crossfit_physics':cross,'new_residual':target,'models':models};self.stages[key]=stage;return stage

def block(d,split,k,j,path,verify_old=True):
    path.mkdir(exist_ok=False);start=time.perf_counter();V=np.flatnonzero(d['fid']==k);A,B,P,short,_=d['cells'][k,j];cp,coef=a.affine(a.mixture(d['deepP'][k,j],q.protected.ARM),d['y'][A],a.mixture(P,q.protected.ARM));anchor=.9*cp+.1*short;assert not(set(A)|set(B))&set(V) and not set(A)&set(B);res=np.full(400,np.nan);res[B]=d['y'][B]-anchor;engine=Nested(d,res);seed=r.CVSEED+100*k+j;ip={arm:np.full((5,len(B)),np.nan) for arm in ARMS};baseline=np.full(len(B),np.nan);lookup={int(i):z for z,i in enumerate(B)};stageids=[]
    for t,(I,H) in enumerate(partitions(d,B,seed)):
        stage=engine.stage(I,seed+1000+t);stageids.append(stage['id']);phys=engine.select(I,seed+1000+t);pred=n.predict(phys['model'],d['X'][H]);hi=[lookup[int(i)] for i in H];baseline[hi]=pred
        for arm in ARMS:
            ip[arm][0,hi]=pred
            for ci,(lam,gamma) in enumerate(CHOICES[1:],1):ip[arm][ci,hi]=pred+gamma*ridge_predict(stage['models'][arm][str(lam)],panel(d,H,arm))
    full=engine.stage(B,seed);phys=engine.select(B,seed);oldpred=n.predict(phys['model'],d['X'][V]);scores={arm:np.mean((ip[arm]-res[B])**2,axis=1) for arm in ARMS};chosen={arm:int(scores[arm].argmin()) for arm in ARMS};delta={}
    for arm in ARMS:
        lam,gamma=CHOICES[chosen[arm]];delta[arm]=np.zeros(len(V)) if lam is None else gamma*ridge_predict(full['models'][arm][str(lam)],panel(d,V,arm))
    oldpath=q.protected.study.OUT/split/f'no_baseline795-fold-{k}/block-{j}';control_error=None
    if verify_old:
        oldrec=e.read(oldpath/'report.json');assert oldrec['chosen']==phys['chosen']
        with np.load(_repo_io(oldpath/'predictions.npz')) as z:np.testing.assert_array_equal(z['block_indices'],B);np.testing.assert_allclose(z['anchor_predictions'],anchor,atol=1e-10,rtol=0);control_error=float(abs(z['correction']-oldpred).max());assert control_error<1e-10
    payload={'A':A,'B':B,'V':V,'anchor':anchor,'residual':res,'anchor_coefficient':coef,'nodes':list(engine.nodes.values()),'stages':list(engine.stages.values()),'cv_stages':stageids,'full_stage':full['id'],'full_node':phys['id'],'inner_predictions':ip,'inner_baseline':baseline,'scores':scores,'chosen':chosen,'old_correction':oldpred,'delta':delta};joblib.dump(payload,_repo_io(path/'systems.joblib'),compress=3)
    rec={'split':split,'fold':k,'block':j,'n':len(B),'selected':{arm:LABELS[chosen[arm]] for arm in ARMS},'inner_MSE':{arm:dict(zip(LABELS,scores[arm].tolist())) for arm in ARMS},'physics_selected':phys['chosen'],'physics_selectors':len(engine.nodes),'physics_fits':len(engine.nodes)*16,'Ridge_fits':len(engine.stages)*4,'014_old_correction_max_error':control_error,'systems_sha256':e.sha(path/'systems.joblib'),'seconds':time.perf_counter()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};e.write(path/'report.json',rec);return oldpred,delta,rec

def run(split):
    check()
    if split!='original':assert e.read(OUT/'screen-decision.json')['advance']
    d=load(split);path=OUT/split;path.mkdir(exist_ok=False);start=time.perf_counter();old=np.full((3,400),np.nan);delta={arm:old.copy() for arm in ARMS};base=np.full(400,np.nan);records=[]
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);p,_=a.affine(a.mixture(d['middleP'][k],q.protected.ARM),d['y'][T],a.mixture(d['P'][V],q.protected.ARM));base[V]=.9*p+.1*d['short'][V]
        for j in range(3):
            op,de,rec=block(d,split,k,j,path/f'fold-{k}-block-{j}');old[j,V]=op
            for arm in ARMS:delta[arm][j,V]=de[arm]
            records.append(rec)
            if split=='original' and k==0 and j==0:
                bench={'seconds':rec['seconds'],'projected15blocks_seconds':15*rec['seconds'],'RSS_bytes':rec['RSS_bytes'],'first_block_reused':True};bench['passed']=bench['projected15blocks_seconds']<900 and bench['RSS_bytes']<3e9;e.write(OUT/'model-benchmark.json',bench);assert bench['passed']
            e.emit({'split':split,'fold':k,'block':j,'selected':rec['selected'],'seconds':time.perf_counter()-start})
    reference=base+old.mean(0);err=float(abs(reference-d['reference014']).max());assert err<1e-10;pred={arm:reference+delta[arm].mean(0) for arm in ARMS}
    with (path/'predictions.npz').open('xb') as h:np.savez_compressed(_repo_io(h),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference014=reference,base=base,old_corrections=old,**pred,**{arm+'_delta':x for arm,x in delta.items()})
    scores={'reference014':e.rmse(d['y'],reference),**{arm:e.rmse(d['y'],p) for arm,p in pred.items()}};report={'split':split,'RMSE':scores,'014_reconstruction_max_error':err,'records':records,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};e.write(path/'report.json',report);check();e.emit({k:v for k,v in report.items() if k!='records'})

def decide():
    check();r=e.read(OUT/'original/report.json');v=e.read(OUT/'validation-original.json');assert v['passed'] and v['report_sha256']==e.sha(OUT/'original/report.json');scores=r['RMSE'];g={name:scores[name]-scores['pair_triple'] for name in ['reference014','pair']};rec={'advance':all(v>0 for v in g.values()),'gains':g,'rule':'3pointaddition mustbeat both014andmatched2pointafterverification. Pairaloneisnotfallback.','bank_closed_if_false':True};e.write(OUT/'screen-decision.json',rec);e.emit(rec)
if __name__=='__main__':
    q.protected.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','decide',*e.SPLITS]);x=p.parse_args();{'prepare':prepare,'decide':decide}.get(x.mode,lambda:run(x.mode))()
