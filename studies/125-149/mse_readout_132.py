"""Audit131 MSE readout, then replace only CNN throughout014 nested lineage."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,gc,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import rnc_study_131 as s
import constrained_weights_125 as q
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
a=q.a;e=q.e;cal=s.cal;OUT=_repo_path(ROOT, 'artifacts/mse-readout-132');ARM='no_baseline795';BASESEED=s.SEED

def data(split):
    d=q.load(split);x=cal.inputs(split);assert d['ids']==x['ids'];d['images']=x['images'];return d

def check():
    p=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in p['hashes'].items():assert e.sha(_repo_path(ROOT, f))==h,f
    return p

def prepare():
    s.check();s.previous.check();OUT.mkdir(exist_ok=False)
    names=['mse_readout_132.py','rnc_study_131.py','cnn_controls.py','frozen_validation.py','component_ablation_121.py','reports/rnc-131-summary.json','reports/rnc-131-preservation.json','reports/candidate014-summary.json','reports/candidate014-verification.json','artifacts/submission-candidate-014.csv','artifacts/nested-calibration-074/development_features.npz']
    p={'study':'132 separate131 MSE procedure audit and CNN-only nested014 replacement','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'duplicate_check':'No running related process and no previous132study.131MSE is not014CNN; no RNC effect claimed.','audit':{'same':'TinyCNN91905parameters,128BOXinput,trainlocalpixel/targetstats,MSE,AdamWlr.002->.0001 wd.01 clip5,D4train,identity inference,sameoriginalouter320/80 IDs/groups.','014':'80epochs,32distinctIDs/update,singleD4view,800updates/25600views,cosine80epoch,averagefinal5epochweights,trainedDropout.15hidden64 scalarhead,seed20261003+k.','131MSE':'40epochs,16distinctIDs with2batchD4views,800updates/25600views,cosine40epoch,finalweights,noaverage. Frozenhidden64preDropout thenStandardScaler/Ridge10 onidentitytrainingfeatures;seed20271115+k.','learning_rates':'Bothspan800updates but LRconstantfor10stepsvs20steps at320train. Equaltotalbudget doesnotmeanidenticaloptimization.','not_single_schedule_causal_test':'Two-view batching,LR staircase,endpointaveraging all changed together. Fixedseedcheckpointcontrastisolates this training-bundle effect,not individualepoch/schedule/augmentation causality.'},'diagnostics':'Saved014and131MSEcheckpoint each evaluated withoriginalscalar andsameRidge10readout: isolatesreadout conditionaloncheckpoint. Trainone131MSErecipewithlegacyseed peroriginalfold,scorebothreadouts: contrastsoriginal131seed andlegacyseed withinnewrecipe. Diagnostics are not fallback candidates.','primary':'Exactly131MSErecipe,seed20271115+offset,40fixedepochs,2views16IDs,MSEAdamW,finalcheckpoint,frozenlatent64StandardScalerRidge10. Reuse5originalouter131checkpoints/predictions. Fitnew15middle213/214+45deep142/143CNNs. Everylevel replacesonlyCNNcolumn;KRR/S/B/shortBcachedhonestmodels unchanged.','nested':'ForouterT320/V80: middle3A213/B107CNN predictions formT-OOF fornewaffine. For eachA,deep3C142/D71CNN predictions formA-OOF foranchor-affine. AnchorB=.9affine_A(raw_B)+.1shortB_A(B). Fitunchangedphysics57five-wayselectoronlyB; applytoouterV,average3corrections. Wholebase=.9newaffine_T(raw_V)+.1shortB_T(V). Neveruseglobal400OOF for training residuals or reuse old calibrators.','inner_budget':'40epochs withceil(n/16)steps,2nviews/epoch. n142/143=>360updates,213/214=>560updates; legacy80ceil(n/32)=>400/560. Bothsee80nviews; noclaimidenticalinneroptimizersteps. Partiallastbatchretained.','seed_lineage':'outerBASE+k,middleBASE+k+100(j+1),deepBASE+k+100(j+1)+100(ell+1),sameoffsetformulaas074/112 withdifferentbase. Groupfoldindices exactly reused from074/112.','control':'Recompute014downstream fromunchangedcache for5originalfolds and compareexact014. Firstgenericouterfit replaymust reproduce131MSE state/predictions exactly;reusedasvalidation notadditionalcandidate.','gate':'Originalcomplete5foldpipelinefirst. Afterindependentverification advanceunchangedprimaryto2existingresplitsonlyifcompletepipelineRMSE<014. No seed/readout/schedulefallbackorblendsearch. No full500/test/submission regardlessresult in132.','validation':'EverynewCNN CPU/MPSreload,independentSVD Ridge1e-3(float32),trainstats/order/lr/exclusion checks. IndependentOLS+all240physicsfits/partition. Outer80poison allnewCNNfits foroneouterfold anddownstream; innerBpoisononeanchor pipeline(deep3+middle+affine), exact unchanged.','resources':'CPU2/MPS,firstfullnestedouter reused benchmark5fold<900sec,RSS/MPSdriver<3GB. Original60nested+5seedcontrols+1outerreplay; extra16poisonfits. Positive2resplits65fitseach. No paidalternative/install/externaldata.','limits':'Selectedafterseeing131control;same400adaptivehistory,3resplitsnotindependentstudies. Readout/training-bundle comparisonsdescriptive withsinglepairedseed,no isolatedscheduleclaim.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'cost':0}
    for k in range(5):
        for file in [s.OUT/'original'/f'fold-{k}-mse/model.pt',s.OUT/'original'/f'fold-{k}-mse/arrays.npz',s.old.prior.q.spec('original',k,2,0)['paths']['cnn']]:p['hashes'][str(file.relative_to(ROOT))]=e.sha(file)
    s.write(_repo_path(OUT, 'plan.json'),p);e.emit({'prepared':True,'new_training_started':False,'primary':'131MSEprocedureonly;notRNC'})

def readout(model,d,T,norm):
    Z=s.encode(model,d['images'],np.arange(400),norm,'mps');sc=StandardScaler().fit(Z[T].astype(float));y=(d['y'][T]-norm['y_mean'])/norm['y_std'];reg=Ridge(alpha=10).fit(sc.transform(Z[T]),y);pp=reg.predict(sc.transform(Z))*norm['y_std']+norm['y_mean'];return Z,sc,reg,pp

def scalar(model,d,norm):return s.old.prior.predict(model,d['images'],np.arange(400),norm,'mps')

def fit(d,T,V,seed,path):
    path.mkdir(exist_ok=False);start=time.perf_counter();assert not set(T)&set(V) and not set(d['g'][T])&set(d['g'][V]);torch.manual_seed(seed);torch.mps.manual_seed(seed);rng=np.random.default_rng(seed);m=s.engine.TinyCNN().to('mps');init=cal.h.state_hash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.0001);norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];orders=[];augs=[];lrs=[];losses=[];offsets=[0]
    for ep in range(40):
        m.train();order=rng.permutation(len(T))
        for off in range(0,len(T),16):
            ii=order[off:off+16];views=[];aug=[]
            for j in range(2):
                rot=int(rng.integers(4));flip=bool(rng.random()<.5);v=np.rot90(X[ii],rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v;views.append(v);aug.extend([rot,int(flip)])
            xx=np.ascontiguousarray(np.concatenate(views));target=torch.tensor(np.tile(y[ii],2),dtype=torch.float32,device='mps');orders.extend(ii.tolist());offsets.append(len(orders));augs.append(aug);lrs.append(opt.param_groups[0]['lr']);opt.zero_grad(set_to_none=True);z=s.embedding(m,torch.from_numpy(xx).to('mps'));pred=m.head[3](m.head[2](z)).squeeze(1);loss=torch.nn.functional.mse_loss(pred,target);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach().cpu()))
        sch.step();assert time.perf_counter()-start<180
    state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};Z,sc,reg,pp=readout(m,d,T,norm);sp=scalar(m,d,norm)
    obj={'state_dict':state,'normalization':norm,'mean':torch.tensor(sc.mean_),'scale':torch.tensor(sc.scale_),'coef':torch.tensor(reg.coef_),'intercept':float(reg.intercept_),'train':T.tolist(),'validation':V.tolist(),'ids':d['ids'],'seed':seed};torch.save(obj,_repo_io(path/'model.pt'))
    with (path/'arrays.npz').open('xb') as f:np.savez_compressed(_repo_io(f),Z=Z,predictions=pp,scalar=sp,order=orders,offsets=offsets,augmentation=augs,lr=lrs,loss=losses)
    rec={'seed':seed,'n_train':len(T),'train':T.tolist(),'validation':V.tolist(),'updates':len(losses),'image_views':len(orders)*2,'initial_sha256':init,'state_sha256':cal.h.state_hash(state),'model_sha256':e.sha(path/'model.pt'),'arrays_sha256':e.sha(path/'arrays.npz'),'train_RMSE':e.rmse(d['y'][T],pp[T]),'validation_RMSE':e.rmse(d['y'][V],pp[V]),'seconds':time.perf_counter()-start};s.write(path/'report.json',rec);del m,opt,Z;gc.collect();torch.mps.empty_cache();return pp,rec

def diagnostics():
    check();d=data('original');folder=_repo_path(OUT, 'diagnostics');folder.mkdir(exist_ok=False);pred={key:np.full(400,np.nan) for key in ['legacy_scalar','legacy_ridge','new_scalar','new_ridge','new_legacyseed_scalar','new_legacyseed_ridge']};records=[]
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k)
        for key,path in [('legacy',s.old.prior.q.spec('original',k,2,0)['paths']['cnn']),('new',s.OUT/'original'/f'fold-{k}-mse/model.pt')]:
            obj=torch.load(_repo_io(path),map_location='cpu',weights_only=True);assert obj.get('train_ids',[d['ids'][i] for i in obj.get('train',[])])==[d['ids'][i] for i in T];m=s.engine.TinyCNN().to('mps');m.load_state_dict(obj['state_dict']);norm=obj['normalization'];Z,sc,reg,pp=readout(m,d,T,norm);sp=scalar(m,d,norm);pred[key+'_scalar'][V]=sp[V];pred[key+'_ridge'][V]=pp[V]
            with (folder/f'{key}-{k}.npz').open('xb') as f:np.savez_compressed(_repo_io(f),Z=Z,scalar=sp,ridge=pp,mean=sc.mean_,scale=sc.scale_,coef=reg.coef_,intercept=reg.intercept_)
            if key=='legacy':assert abs(sp[V]-d['P'][V,2]).max()<1e-3
            else:
                with np.load(_repo_io(s.OUT/'original'/f'fold-{k}-mse/arrays.npz')) as z:np.testing.assert_array_equal(pp,z['predictions'])
            del m
        pp,rec=fit(d,T,V,20261003+k,folder/f'new-legacyseed-{k}');pred['new_legacyseed_ridge'][V]=pp[V]
        with np.load(_repo_io(folder/f'new-legacyseed-{k}/arrays.npz')) as z:pred['new_legacyseed_scalar'][V]=z['scalar'][V]
        records.append(rec);e.emit({'diagnostic_fold_complete':k,'new_seed_control_seconds':rec['seconds']})
    # Exact functional replay of locked131implementation before using genericinner sizes.
    T=np.flatnonzero(d['fid']!=0);V=np.flatnonzero(d['fid']==0);pp,rec=fit(d,T,V,BASESEED,folder/'exact131-replay');old=e.read(s.OUT/'original/fold-0-mse/report.json');assert rec['state_sha256']==old['state_sha256'];np.testing.assert_array_equal(pp[V],pred['new_ridge'][V])
    raw=.3*d['P'][:,1]+.2*d['P'][:,3]+.2*d['P'][:,4];scores={key:{'CNN':e.rmse(d['y'],p),'raw4way':e.rmse(d['y'],raw+.3*p)} for key,p in pred.items()}
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**pred)
    report={'scores':scores,'exact131_generic_replay':True,'newfits':6,'fit_seconds':sum(x['seconds'] for x in records)+rec['seconds'],'records':records,'source_hashes':{str(p.relative_to(ROOT)):e.sha(p) for p in folder.rglob('*') if p.is_file()},'interpretation':'Readoutchangesisolatedconditionaloncheckpoint. Samelegacyseednewvslegacy contrastsjointtwo-view/LR-staircase/finalvsaverage trainingbundle. Seedcontrastinsidefixednewrecipe. No RNCcausalclaim orfallbackselection.'};s.write(folder/'report.json',report);check();e.emit({'diagnostics_complete':True,'scores':scores,'exact131_replay':True})

def cached_predictions(path):
    rec=e.read(path/'report.json');assert e.sha(path/'model.pt')==rec['model_sha256'] and e.sha(path/'arrays.npz')==rec['arrays_sha256']
    with np.load(_repo_io(path/'arrays.npz')) as z:return z['predictions'].copy()

def replace_fold(d,split,k,root,train=True):
    T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);new=dict(d);new['P']=d['P'].copy();new['middleP']=dict(d['middleP']);new['middleP'][k]=d['middleP'][k].copy();new['deepP']=dict(d['deepP']);new['cells']=dict(d['cells']);models=[]
    def get(Ti,Vi,seed,path):
        if train:pp,rec=fit(d,Ti,Vi,seed,path)
        else:pp=cached_predictions(path);rec=e.read(path/'report.json');assert rec['train']==Ti.tolist() and rec['validation']==Vi.tolist()
        models.append(str(path.relative_to(ROOT)));return pp
    if split=='original':outerpath=s.OUT/'original'/f'fold-{k}-mse';op=cached_predictions(outerpath)
    else:outerpath=root/'outer';op=get(T,V,BASESEED+k,outerpath)
    new['P'][V,2]=op[V];lookup={int(i):j for j,i in enumerate(T)}
    for j in range(3):
        A,B,P,short,old=d['cells'][k,j];mp=get(A,B,BASESEED+k+100*(j+1),root/f'middle-{j}');Q=P.copy();Q[:,2]=mp[B];new['cells'][k,j]=(A,B,Q,short,old);new['middleP'][k][[lookup[int(i)] for i in B],2]=mp[B];new['deepP'][k,j]=d['deepP'][k,j].copy();la={int(i):ii for ii,i in enumerate(A)};seen=[]
        for ell in range(3):
            source=e.OUT/split/f'fold-{k}/inner-{j}/deep-{ell}';C,D,_=a.bundle(source);assert set(C)|set(D)==set(A) and not set(C)&set(D) and not(set(C)|set(D))&(set(B)|set(V));assert not set(d['g'][C])&set(d['g'][D]);pp=get(C,D,BASESEED+k+100*(j+1)+100*(ell+1),root/f'deep-{j}-{ell}');new['deepP'][k,j][[la[int(i)] for i in D],2]=pp[D];seen.extend(D.tolist())
        assert sorted(seen)==sorted(A.tolist())
        e.emit({'split':split,'outer':k,'nested_block_complete':j,'training':train})
    return new,models

def run(split):
    check();assert e.read(_repo_path(OUT, 'diagnostics/report.json'))['exact131_generic_replay']
    if split!='original':assert e.read(_repo_path(OUT, 'screen-decision.json'))['advance']
    d=data(split);path=_repo_path(OUT, split);path.mkdir(exist_ok=False);start=time.perf_counter();primary=np.full(400,np.nan);base=primary.copy();raw=primary.copy();records=[];models=[]
    for k in range(5):
        ft=time.perf_counter();root=path/f'cnn-fold-{k}';root.mkdir();dd,files=replace_fold(d,split,k,root);models.extend(files);V=np.flatnonzero(d['fid']==k);primary[V],base[V],rec=a.fold(dd,k,ARM,path/f'pipeline-fold-{k}');raw[V]=a.mixture(dd['P'][V],ARM);records.append(rec)
        if split=='original':
            cp,_,cr=a.fold(d,k,ARM,path/f'control014-fold-{k}');np.testing.assert_allclose(cp,d['reference014'][V],atol=1e-10,rtol=0)
        e.emit({'split':split,'outer_complete':k,'elapsed_seconds':time.perf_counter()-start})
        if k==0:
            b={'first_outer_seconds_reused':time.perf_counter()-ft,'projected5outer_seconds':5*(time.perf_counter()-ft),'memory':s.engine.memory('mps')};b['passed']=b['projected5outer_seconds']<900 and b['memory']['process_peak_rss_bytes']<3e9 and b['memory']['driver_allocated_memory']<3e9;s.write(path/'benchmark.json',b);assert b['passed']
    with (path/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference014=d['reference014'],primary=primary,base=base,raw=raw)
    r={'split':split,'RMSE':{'reference014':e.rmse(d['y'],d['reference014']),'primary':e.rmse(d['y'],primary),'base':e.rmse(d['y'],base),'raw':e.rmse(d['y'],raw)},'records':records,'new_CNN_paths':models,'new_CNN_fits':len(models),'reused_outer131':split=='original','predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-start};s.write(path/'report.json',r);check();e.emit({'split':split,'RMSE':r['RMSE'],'seconds':r['seconds']})
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','diagnostics',*e.SPLITS]);x=p.parse_args();prepare() if x.mode=='prepare' else diagnostics() if x.mode=='diagnostics' else run(x.mode)
