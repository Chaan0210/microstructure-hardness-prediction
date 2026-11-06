"""Fixed nested saved-model data-size and raw-mixture complementarity diagnostic."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,datetime,argparse,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import domain_risk_audit_126 as previous
import base_learning_curve_127 as frozen
import numpy as np,torch,joblib
from sklearn.preprocessing import StandardScaler
p=previous.p;a=p.a;e=p.e;cal=e.cal;h=cal.h
OUT=_repo_path(ROOT, 'artifacts/base-learning-curve-127/numeric-replay-v2');NAMES=('compact26','CNN80','DINO_S_mean3','DINO_B_mean3');W=np.array([.3,.3,.2,.2]);LEVELS=('142_143','213_214','320');SEEDS=cal.SEEDS

def spec(split,k,level,j):
    outer=_repo_path(ROOT, 'artifacts/dino-frozen-037/regression', split, f'fold-{k}')
    if level==2:
        cr=e.read(outer.parent/f'fold-{k}-report.json');modelpaths={'kernel':outer/'compact-kernel.joblib','cnn':outer/'tiny128.pt'};hashes={n:cr['sha256'][f.name] for n,f in modelpaths.items()};sr=e.read(_repo_path(ROOT, 'artifacts/seed-stability-052', split, f'fold-{k}/report.json'));brroot=_repo_path(ROOT, 'artifacts/dinob-mlp-063' if split=='original' else 'artifacts/dinob-complement-064', split, f'fold-{k}');br=e.read(brroot/'report.json')
        for seed in SEEDS:
            rec=sr['models'][str(seed)];modelpaths[f'S_{seed}']=_repo_path(ROOT, rec['checkpoint']);hashes[f'S_{seed}']=rec['checkpoint_sha256'];modelpaths[f'B_{seed}']=brroot/f'baseB-seed-{seed}.pt';hashes[f'B_{seed}']=br['models'][f'baseB-{seed}']['checkpoint_sha256']
        return {'paths':modelpaths,'hashes':hashes,'source_report':outer.parent/f'fold-{k}-report.json','cache':None,'report':cr}
    folder=cal.OUT/split/f'fold-{k}/inner-{j}' if level==1 else e.OUT/split/f'fold-{k}/inner-{j}/deep-0'
    rec=e.read(folder/'report.json');paths={'kernel':folder/'kernel.joblib','cnn':folder/'cnn.pt'}
    paths.update({f'{arm}_{s}':folder/f'{arm}-{s}.pt' for arm in ['S','B'] for s in SEEDS})
    return {'paths':paths,'hashes':{n:rec['files_sha256'][f.name] for n,f in paths.items()},'source_report':folder/'report.json','cache':folder/'predictions.npz','report':rec}

def check():
    frozen.check();plan=e.read(_repo_path(OUT, 'plan.json'))
    for name,hsh in {**plan['source_sha256'],**plan['inputs_sha256'],**plan['protected_sha256']}.items():assert e.sha(_repo_path(ROOT, name))==hsh,name
    return plan

def prepare():
    frozen.check();OUT.mkdir(exist_ok=False);inputs={}
    for split in e.SPLITS:
        for k in range(5):
            for level in range(3):
                for j in (range(3) if level<2 else [0]):
                    sp=spec(split,k,level,j)
                    for name,f in sp['paths'].items():assert e.sha(f)==sp['hashes'][name];inputs[str(f.relative_to(ROOT))]=sp['hashes'][name]
                    for f in [sp['source_report']]+([sp['cache']] if sp['cache'] else []):inputs[str(f.relative_to(ROOT))]=e.sha(f)
    for name in ['artifacts/base-learning-curve-127/plan.json','reports/base-learning-curve-127-original.log','reports/learning-curve-050-summary.json','reports/post009-research-076-078-summary.json','reports/residual-subsamples-118-summary.json','reports/baseline124-weights125-summary.json','reports/domain-risk-audit-126-summary.json','reports/cnn-controls-summary.json','reports/optimization-057-058-summary.json','reports/frozen-head-087-seed-088-summary.json','reports/dropout-control-089-summary.json','reports/width-control-090-summary.json','reports/width-research-090-091-summary.log']:
        inputs[name]=e.sha(_repo_path(ROOT, name))
    plan={'study':'127 fixed saved-base-model nested data-size curve','numeric_repair':'Original first cache test used1e-8 for float32head predictions despite changed GEMM batch dimensions. Observed Smean6.735e-7, CNN1.526e-5, KRR4.832e-13; matched original S seed batch reproduced exactly0. Preserve original sources/plan/log. V2 allows1e-4 S/B and1e-3CNN for batch-size differences, keeps1e-8KRR, adds exact original-batch head cache checks. No model/training/scoring change.','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'question':'How do the four surviving014 main branches change train/heldout error and raw-mixture complementarity with more labeled training rows? Which evidence supports longerCNN, representation changes, or regression regularization?','duplicate_audit':{'050':'Old frozenS Ridge80/160/320 only; not currentCNN/KRR/S/B MLPs.','078':'213/214-to320 aggregated legacy003 versus S/Bmean reliability; not separate fourbranch curves or142/143point/train gaps. Repeat078aggregate endpoint as numerical control only.','118':'Physics57 residual learner half-sample sizes; not independent main base models.','124':'Baseline795 training gap and213/320; removed model not studied here.','121_122':'Full downstream ablations at320;127 raw-mixture leave-one diagnostics do not replace those full-pipeline results.','127_distinction':'Add nested142/143point, separatecompact26/CNN/S/B, train gaps and fixed raw-mixture diversity on exactlysameouter80 at every size. No new fitting or hyperparameter selection.'},'models':list(NAMES),'paths':'For each of3partitions and5outerfolds, reuse all3existing074inner A213/214 and corresponding112 deep-0 D142/143 subsets. D subset A subset T320; deep-0 fixed without outcomes. Exactly45deep+45middle+15outer bundles,8leafmodels perbundle(1KRR,1CNN,3S,3B).','training_protocol':'Same existing family procedure acrosssizes: CNN12880epochs,last5weightaverage; S/B frozenfeatures same3seed100epochMLPs; compact26 same9candidate nestedtrain-onlyKRRgrid. No new model choice. Seeds/subsetmembership/stepcounts differ; fixedepochs is not fixedupdates. This descriptive data-size curve cannot isolate causal sample-size or optimization effects.','outer_evaluation':'Each nested model evaluated on its root outer80, excluded from every model/scaler/kernel-choice fit. Report mean individual-subset MSE across3paths then RMSE; DO NOT average the3subset models as a newpredictor. 320 prediction repeated overpathaxis for paired arithmetic only, not extra fits/independentobservations.','training_evaluation':'Recompute inference-mode train predictions for each own train subset. Aggregate squared errors weighted by number of training rows. These overlapping training losses are descriptive, never evaluation or tuning.','contribution':'Fixed rawmix [.3,.3,.2,.2] only. Per-size error-covariance matrix, weighted individual MSE minus weighted disagreement identity, leave-one-branch-out normalizedremainingweights MSE delta. No weight fitting. These are raw4way diagnostics excluding014 affine,shortB,residual; not actual removal proposals or new candidate scores.','uncertainty':'10000paired ID-clusterbootstrap seed20271107 for size changes; each ID carries3partitions and3nestedpaths together after averagingpathloss. Only400uniqueimages, not3600independentrows. Condition on existing fits; adaptive research/refitting uncertainty omitted.','resources':'Existinguv CPU2; reload840leafmodels,105bundles; no newtraining/features/images. First deepbundle reusedbenchmark; projected105bundle replay<600seconds andRSS<2GB. No install,cost,holdout/test/full500/upload.','validation':'Checkmodelhashes,trainIDs,target/scalerstats,subsetnesting and rootouterexclusion. Compare everymodel on its oldcachevalidationrows, verify320 predictionsagainst014 inputs. Independenthead functional andkernel distance solve; freshreload paired subset/outer; independent scalar MSE/diversity andbootstrap checks. Priorupstreamtraining leakage certificates reused, no new trainingclaim.','scope_bias':'Same400previouslyused for many model/hypothesis choices. New analysis descriptive/exploratory,not independentconfirmation. Results may set research priority but do not select a new deployablemodel. No extrapolated500/private score or performanceceiling.','source_sha256':{name:e.sha(_repo_path(ROOT, name)) for name in ['base_learning_curve_v2_127.py','validate_base_learning_curve_v2_127.py','frozen_validation.py','dinob_mlp_head_063.py','compact_regression.py']},'inputs_sha256':inputs,'protected_sha256':{str(f.relative_to(ROOT)):e.sha(f) for f in [p.CSV,_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/domain-risk-audit-126-preservation.json')]},'new_fits':0,'cost':0}
    e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'bundles':105,'leaf_models':840,'new_fits':0,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def cnn_predict(saved,X,batch=32):
    m=cal.fv.engine.TinyCNN().eval();m.load_state_dict(saved['state_dict']);norm=saved['normalization'];out=[]
    with torch.inference_mode():
        for st in range(0,len(X),batch):out.extend((m(torch.tensor((X[st:st+batch]-norm['x_mean'])/max(norm['x_std'],1e-6))).numpy()*norm['y_std']+norm['y_mean']).tolist())
    return np.array(out)

def replay(d,split,k,level,j,path):
    t=time.perf_counter();sp=spec(split,k,level,j);rootT=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);ids=d['ids'];s=sp['report'];T=rootT if level==2 else np.array(s['train_indices']);assert set(T)<=set(rootT) and not set(d['g'][T])&set(d['g'][V]);savedhash={};cache=a.readnp(sp['cache'],s['files_sha256']['predictions.npz']) if sp['cache'] else None;matched_error=0.
    for name,f in sp['paths'].items():assert e.sha(f)==sp['hashes'][name];savedhash[str(f.relative_to(ROOT))]=sp['hashes'][name]
    km=joblib.load(_repo_io(sp['paths']['kernel']));assert km['train_ids']==[ids[i] for i in T];ker=km['model'];np.testing.assert_array_equal(ker.bound_.data_min_,d['K'][T].min(0));np.testing.assert_array_equal(ker.bound_.data_max_,d['K'][T].max(0));assert ker.ym_==float(d['y'][T].mean()) and ker.ys_==float(d['y'][T].std());kp=ker.predict(d['K']);cn=torch.load(_repo_io(sp['paths']['cnn']),weights_only=True,map_location='cpu');assert cn['train_ids']==[ids[i] for i in T] and cn['epochs']==80 and cn['average_last_epochs']==5;norm=cn['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};cp=cnn_predict(cn,d['images']);heads=[]
    for arm in ['S','B']:
        X=d[arm];sc=StandardScaler().fit(X[T]);pred=[]
        for seed in SEEDS:
            obj=torch.load(_repo_io(sp['paths'][f'{arm}_{seed}']),weights_only=True,map_location='cpu');assert obj['train_ids']==[ids[i] for i in T];assert obj['target_mean']==float(d['y'][T].mean()) and obj['target_std']==float(d['y'][T].std());np.testing.assert_array_equal(obj['scaler_mean'].numpy(),sc.mean_);np.testing.assert_array_equal(obj['scaler_scale'].numpy(),sc.scale_);assert obj['config']==h.CONFIG
            m=h.Head(X.shape[1]);m.load_state_dict(obj['model']);pred.append(h.predict(m,sc.transform(X),obj['target_mean'],obj['target_std']))
            if cache is not None:
                ci=cache['validation_indices'];cpred=h.predict(m,sc.transform(X[ci]),obj['target_mean'],obj['target_std']);err=float(abs(cpred-cache[f'{arm}_{seed}']).max());assert err<1e-8;matched_error=max(matched_error,err)
        heads.append(np.stack(pred))
    heads=np.stack(heads);P=np.stack([kp,cp,*heads.mean(1)]);assert P.shape==(4,400) and np.isfinite(P).all()
    if sp['cache']:
        z=a.readnp(sp['cache'],s['files_sha256']['predictions.npz']);C=z['validation_indices'];expected=np.stack([z['kernel'],z['cnn'],z['Smean'],z['Bmean']]);cacheerr=np.max(abs(P[:,C]-expected),axis=1);assert cacheerr[1]<1e-3 and cacheerr[0]<1e-8 and np.max(cacheerr[2:])<1e-4
    else:
        C=V;cacheerr=np.max(abs(P[:,V]-d['outerP'][V,1:].T),axis=1);assert cacheerr[1]<1e-3 and cacheerr[0]<1e-8 and np.max(cacheerr[2:])<1e-4
    path.mkdir(exist_ok=False)
    with (path/'predictions.npz').open('xb') as out:np.savez_compressed(_repo_io(out),train_indices=T,outer_indices=V,components=P,seeds=heads)
    curve=s.get('logs',{}).get('cnn',{}).get('curve',[])
    rec={'split':split,'outer_fold':k,'level':LEVELS[level],'path_index':j,'n_train':len(T),'train_indices':T.tolist(),'outer_indices':V.tolist(),'train_SSE':np.sum((P[:,T]-d['y'][T])**2,axis=1).tolist(),'outer_MSE':np.mean((P[:,V]-d['y'][V])**2,axis=1).tolist(),'kernel_bandwidth':ker.bandwidth,'kernel_alpha':ker.alpha,'kernel_alpha_per_trainrow':ker.alpha/len(T),'CNN_epochs':cn['epochs'],'CNN_aug_train_RMSE_last10':float(np.mean([z['augmented_train_rmse'] for z in curve[-10:]])) if curve else None,'CNN_aug_train_RMSE_prev10':float(np.mean([z['augmented_train_rmse'] for z in curve[-20:-10]])) if curve else None,'original_batch_head_cache_max_error':matched_error if cache is not None else None,'old_cache_max_error_by_component':cacheerr.tolist(),'model_sha256':savedhash,'predictions_sha256':e.sha(path/'predictions.npz'),'seconds':time.perf_counter()-t};e.write(path/'report.json',rec);return P,rec

def run(split):
    check();out=_repo_path(OUT, split);out.mkdir(exist_ok=False);d=cal.inputs(split);outer=a.load(split);d['outerP']=outer['P'];arrays=np.full((3,3,4,400),np.nan);records=[];t=time.perf_counter()
    for k in range(5):
        V=np.flatnonzero(d['fid']==k);T=np.flatnonzero(d['fid']!=k)
        for level in range(3):
            for j in (range(3) if level<2 else [0]):
                if level<2:
                    middle=np.array(spec(split,k,1,j)['report']['train_indices']);current=np.array(spec(split,k,level,j)['report']['train_indices']);assert set(current)<=set(middle)<=set(T)
                dest=out/f'fold-{k}-level-{level}-path-{j}';P,rec=replay(d,split,k,level,j,dest);records.append(rec)
                if level==2:arrays[level,:,:,V]=np.broadcast_to(P[:,V].T[:,None,:],(len(V),3,4))
                else:arrays[level,j,:,V]=P[:,V].T
                if split=='original' and k==0 and level==0 and j==0:
                    bench={'seconds':rec['seconds'],'projected105bundle_seconds':105*rec['seconds'],'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'first_bundle_reused':True};bench['passed']=bench['projected105bundle_seconds']<600 and bench['RSS_bytes']<2e9;e.write(_repo_path(OUT, 'benchmark.json'),bench);assert bench['passed']
        e.emit({'split':split,'completed_outer':k,'bundles':len(records),'seconds':time.perf_counter()-t})
    assert np.isfinite(arrays).all()
    with (out/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],components=arrays)
    result={'split':split,'bundles':len(records),'model_reloads':8*len(records),'records':records,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-t};e.write(out/'report.json',result);check();e.emit({'split':split,'complete':True,'seconds':result['seconds'],'RMSE_by_size_component':np.sqrt(np.mean((arrays-d['y'])**2,axis=(1,3))).tolist()})

if __name__=='__main__':
    torch.set_num_threads(2);ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare',*e.SPLITS]);args=ap.parse_args();prepare() if args.mode=='prepare' else run(args.mode)
