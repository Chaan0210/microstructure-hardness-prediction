"""One post-hoc missing factorial cell, D=valid-scale mean without flags."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import sys, json, hashlib, time, resource
from pathlib import Path
R = Path(__file__).resolve().parent
P = R.parents[3]
assert P.name == 'microstructure-2026' and Path(sys.prefix).resolve() == _repo_path(P, '.venv')
sys.dont_write_bytecode = True
sys.path[:0] = [str(R), str(_repo_path(P, 'studies'))]
import numpy as np, joblib
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits, threadpool_info
from policy import PolicyKernel
threadpool_limits(2)
SPLITS = ('original', '20261031', '20261101')
GRID = [(w, a) for w in (.25, 1., 4.) for a in (.01, .1, 1.)]
READS, BLOCKED = set(), []
START = time.monotonic()
FITS = 0

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    with Path(p).open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
def load(p, keys=None):
    with np.load(_repo_io(p), allow_pickle=False) as z: return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def arrays(p, **values):
    with Path(p).open('xb') as f: np.savez_compressed(_repo_io(f), **values)
def emit(obj): print(json.dumps(obj, ensure_ascii=False), flush=True)
def rmse(y, pred): return float(np.sqrt(np.mean((y-pred)**2)))
def resource_check():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2e9
    assert time.monotonic()-START < 300
    assert all(x['num_threads'] <= 2 for x in threadpool_info())
def stat(p):
    s=p.stat(); return {'size':s.st_size, 'mtime_ns':s.st_mtime_ns, 'inode':s.st_ino}

inputs = ['compact_regression.py', 'experiments/grain-validity-218/policy.py',
          'experiments/grain-validity-218/run.py', 'experiments/grain-validity-218/validate_score.py',
          'experiments/grain-validity-218/plan.json', 'experiments/grain-validity-218/validation.json',
          'experiments/grain-validity-218/results.json', 'experiments/grain-validity-218/paired-results.npz',
          'artifacts/run-012/development_features.npz', 'artifacts/run-012/report.json',
          'artifacts/run-006/development_features.npz', 'artifacts/run-006/report.json']
for s in SPLITS:
    inputs += [f'experiments/grain-validity-218/{s}/OOF.npz', f'experiments/grain-validity-218/{s}/completion.json',
               f'experiments/dinol-177/{s}/predictions.npz', f'experiments/dinol-177/{s}/report.json',
               f'artifacts/cnn-seed-mean-133/{s}/predictions.npz', f'artifacts/component-ablation-121/{s}/predictions.npz']
    for k in range(5):
        inputs += [f'experiments/grain-validity-218/{s}/fold-{k}/{arm}/report.json' for arm in ('A_current','B_validity','C_valid_mean')]
        inputs += [f'artifacts/spatial-joint-140/{s}/adjacent-fold-{k}/report.json']
ALLOW = set(inputs)
def audit(event, args):
    if event.startswith('socket.'):
        BLOCKED.append(event); raise RuntimeError('No network')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).absolute()
        mode, flags = args[1:3]
        wr = (isinstance(mode,str) and any(v in mode for v in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if wr and not p.is_relative_to(R) and str(p)!='/dev/null':
            BLOCKED.append(str(p)); raise RuntimeError('Write outside isolated result directory')
        if p.is_relative_to(P):
            rel=str(p.relative_to(P))
            if p.is_relative_to(_repo_path(P, 'data')) or p.name in ('.env','.netrc','.git-credentials','credentials','token'):
                BLOCKED.append(rel); raise RuntimeError('Forbidden data/credential read')
            runtime=p.is_relative_to(_repo_path(P, '.venv')) or (p.parent==_repo_path(P, 'studies') and p.suffix=='.py') or p.is_relative_to(_repo_path(P, 'studies/src'))
            if not p.is_relative_to(R) and rel not in ALLOW and not runtime:
                BLOCKED.append(rel); raise RuntimeError('Unplanned project file')
            READS.add(rel)
        elif str(p).startswith('/Users/'):
            BLOCKED.append('other_user_path'); raise RuntimeError('No unrelated user path')
sys.addaudithook(audit)


def features():
    z=load(_repo_path(P, 'artifacts/run-012/development_features.npz'))
    g=load(_repo_path(P, 'artifacts/run-006/development_features.npz'))
    np.testing.assert_array_equal(z['ids'],g['ids'])
    names=read(_repo_path(P, 'artifacts/run-006/report.json'))['feature_names']
    hp=np.column_stack([g['X'][:,names.index(f'grain_s{s}_interior_hall_petch')] for s in [1,2]])
    count=np.column_stack([g['X'][:,names.index(f'grain_s{s}_interior_count')] for s in [1,2]])
    valid=count>0
    assert np.all(hp[valid]>0) and np.all(hp[~valid]==0)
    np.testing.assert_array_equal(z['X'][:,0],hp.mean(1))
    A=np.column_stack([z['X'],np.zeros((400,2))]); D=A.copy()
    D[:,0]=np.divide((hp*valid).sum(1),valid.sum(1),out=np.full(400,np.nan),where=valid.sum(1)>0)
    missing=~valid.all(1)
    assert missing.sum()==3 and np.sum(D[:,0]!=A[:,0])==3
    np.testing.assert_array_equal(D[:,1:],A[:,1:]); assert np.all(D[:,-2:]==0)
    assert len(set(z['ids']))==len(set(z['groups']))==400 and D.shape==(400,28)
    blocks=read(_repo_path(P, 'artifacts/run-012/report.json'))['blocks']+['grain_validity']*2
    return z,A,D,blocks,missing,hp


def scoped_fit(X,y,g,T,V,blocks,k,expected_splits):
    global FITS
    assert not set(T)&set(V) and not set(g[T])&set(g[V])
    inner=list(GroupKFold(3,shuffle=True,random_state=20261005+k).split(X[T],y[T],g[T]))
    actual=[{'train':T[i].tolist(),'validation':T[h].tolist()} for i,h in inner]
    assert actual==expected_splits
    ip=np.full((9,len(T)),np.nan)
    def fit(I,H,w,a):
        global FITS
        assert not set(I)&set(H) and np.isfinite(y[I]).all()
        m=PolicyKernel(blocks,w,a).fit(X[I],y[I]);FITS+=1
        Z=m.transform(X[I]); expected_median=float(np.median(X[I,0][np.isfinite(X[I,0])]))
        assert m.impute_==expected_median
        np.testing.assert_array_equal(m.model_.bound_.data_min_,Z.min(0))
        np.testing.assert_array_equal(m.model_.bound_.data_max_,Z.max(0))
        assert m.model_.ym_==float(y[I].mean()) and m.model_.ys_==float(y[I].std())
        return m,m.predict(X[H])
    for ci,(w,a) in enumerate(GRID):
        for I,H in inner:
            assert not set(g[T[I]])&set(g[T[H]]) and not (set(T[I])|set(T[H]))&set(V)
            m,ip[ci,H]=fit(T[I],T[H],w,a)
    assert np.isfinite(ip).all()
    scores=np.sqrt(np.mean((ip-y[T])**2,axis=1));choice=int(np.argmin(scores));w,a=GRID[choice]
    m,pred=fit(T,V,w,a)
    return pred,m,ip,{'chosen_index':choice,'chosen':{'width':w,'alpha':a},'inner_scores':scores.tolist(),'inner_splits':actual,'train_indices':T.tolist(),'validation_indices':V.tolist()}


def independent(m,X,y,T,V,blocks,pred):
    U=X[T].copy();Q=X[V].copy();med=float(np.median(U[np.isfinite(U[:,0]),0]))
    U[~np.isfinite(U[:,0]),0]=med;Q[~np.isfinite(Q[:,0]),0]=med
    minimum=U.min(0);span=U.max(0)-minimum;span[span==0]=1
    Us=np.clip((U-minimum)/span,0,1);Qs=np.clip((Q-minimum)/span,0,1)
    mean=Us.mean(0);std=Us.std(0);std[std==0]=1;block=np.array(blocks)
    w=np.array([1/np.sqrt(np.sum(block==b)) for b in block]);Ut=(Us-mean)/std*w;Qt=(Qs-mean)/std*w
    np.testing.assert_allclose(Ut,m.model_.X_,rtol=0,atol=1e-12)
    dist=pdist(Ut,'sqeuclidean');gamma=1/(np.median(dist[dist>1e-12])*m.bandwidth**2)
    assert abs(gamma-m.model_.gamma_)<1e-12
    K=np.exp(-gamma*squareform(dist));K.flat[::len(T)+1]+=m.alpha
    coef=np.linalg.solve(K,(y[T]-y[T].mean())/y[T].std())
    manual=np.exp(-gamma*cdist(Qt,Ut,'sqeuclidean'))@coef*y[T].std()+y[T].mean()
    error=float(abs(manual-pred).max());assert error<1e-8
    return error


def main():
    protected={f:sha(_repo_path(P, f)) for f in inputs}
    best=_repo_path(P, 'artifacts/submission-candidate-021.csv');best_stat=stat(best)
    source={f:sha(_repo_path(R, f)) for f in ('study.py','policy.py')}
    oldplan=read(_repo_path(P, 'experiments/grain-validity-218/plan.json'))
    assert read(_repo_path(P, 'experiments/grain-validity-218/validation.json'))['passed']
    plan={'study':'218 missing factorial cell mean-only','posthoc':True,'selection':'Requested after observing 218 A/B/C and the audit; exploratory, not independent confirmation.',
          'condition':'D: original26 only HP column valid-scale mean; both padded flag columns zero. All other feature/model/grid/ID/fold/seed decisions identical to218.',
          'grid':GRID,'inner_seed':'20261005+outerfold','splits':SPLITS,'main_fits':420,'no_A_B_C_refitting':True,
          'primary':'D-A pooled untrimmed fixed-downstream ensemble RMSE gain; standalone and all pairwise factorial comparisons are secondary.',
          'decision_before_new_scores':{'minimum_pooled_fixed_ensemble_RMSE_gain':.02,'all_three_split_gains_positive':True,'remaining397_pooled_gain_nonnegative':True,'conditional_CI95_lower_positive':True,'if_any_fail':'Reject this condition and stop; no new threshold, feature, flag scaling, weight or model search.','interpretation':'Practical screening rule, not statistical proof; even a pass cannot promote a frozen-downstream proxy.'},
          'statistics':'Same218 ID bootstrap10000/seed202610218 and same397 draws; each ID retains all3 repetitions; no inferred precision for3 cases; CI omits fitting and historical selection uncertainty.',
          'verification':'All420 fit train-only bounds/targets; 15 saved model independent solves/reloads; full nested outer-label and outer-feature poison on fold0 of each split (168 additionalfits); existing A/B/C metrics and C-A bootstrap exactly replayed; frozen downstream independently reconstructed.',
          'resources':'Existing project uv-created .venv; CPU<=2 threads; peakRSS<2GB;300s; no neural/GPU/install/network/rawdata/test/holdout/full500/CSV/submission.',
          'input_SHA256':protected,'source_SHA256':source,'best021_stat_without_content_read':best_stat,'output_directory':str(R.relative_to(P))}
    write(_repo_path(R, 'plan.json'),plan)
    z,A,D,blocks,missing,hp=features();ids,y,g=z['ids'],z['y'],z['groups']
    assert ids.tolist()==oldplan['IDs'] and ids[missing].tolist()==oldplan['known_empty_IDs']
    arrays(_repo_path(R, 'feature-delta.npz'),ids=ids,changed=missing,old_HP=A[:,0],D_HP=D[:,0],flags=D[:,-2:])
    data={};validation=[];changes={};errors=[];main_fits=0;poison_fits=0
    for split in SPLITS:
        dest=_repo_path(R, split);dest.mkdir()
        oldroot=_repo_path(P, f'experiments/grain-validity-218/{split}')
        old=load(oldroot/'OOF.npz');assert sha(oldroot/'OOF.npz')==read(oldroot/'completion.json')['OOF_sha256']
        for name,val in [('ids',ids),('y',y),('groups',g)]:np.testing.assert_array_equal(old[name],val)
        np.testing.assert_array_equal(old['empty'],missing)
        pred=np.full(400,np.nan);split_checks=[];changes[split]=[]
        for k in range(5):
            resource_check();T=np.flatnonzero(old['fold']!=k);V=np.flatnonzero(old['fold']==k);assert len(T)==320 and len(V)==80
            ref=read(oldroot/f'fold-{k}/A_current/report.json')
            before=FITS;pv,m,ip,info=scoped_fit(D,y,g,T,V,blocks,k,ref['inner_splits']);main_fits+=FITS-before
            folder=dest/f'fold-{k}';folder.mkdir();joblib.dump(m,_repo_io(folder/'model.joblib'))
            reload=joblib.load(_repo_io(folder/'model.joblib'));np.testing.assert_array_equal(reload.predict(D[V]),pv)
            err=independent(reload,D,y,T,V,blocks,pv);errors.append(err);pred[V]=pv
            if info['chosen_index']!=ref['chosen_index']:changes[split].append(k)
            arrays(folder/'predictions.npz',train=T,validation=V,inner=ip,prediction=pv)
            info.update(model_SHA256=sha(folder/'model.joblib'),predictions_SHA256=sha(folder/'predictions.npz'),independent_maxerror=err,missing_train=int(missing[T].sum()),missing_validation=int(missing[V].sum()))
            write(folder/'report.json',info)
            if k==0:
                yy=y.copy();yy[V]=np.nan;before=FITS
                p2,m2,i2,r2=scoped_fit(D,yy,g,T,V,blocks,k,ref['inner_splits'])
                np.testing.assert_array_equal(p2,pv);np.testing.assert_array_equal(i2,ip);np.testing.assert_array_equal(m2.model_.coef_,m.model_.coef_);assert r2=={kk:info[kk] for kk in r2}
                altered=D.copy();altered[V,:26]+=11
                _,m3,i3,r3=scoped_fit(altered,y,g,T,V,blocks,k,ref['inner_splits'])
                np.testing.assert_array_equal(i3,ip);np.testing.assert_array_equal(m3.model_.coef_,m.model_.coef_);np.testing.assert_array_equal(m3.model_.X_,m.model_.X_);assert r3=={kk:info[kk] for kk in r3}
                poison_fits+=FITS-before
            split_checks.append({'fold':k,'independent_maxerror':err,'train_only_all28fits':True,'reloaded_exact':True,'poison_paths_checked':k==0})
        old['D_mean_only']=pred
        old['ensemble_D_mean_only']=old['reference021']+old['coefficient']*(pred-old['old_KRR'])
        dino=load(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'));assert sha(_repo_path(P, f'experiments/dinol-177/{split}/predictions.npz'))==read(_repo_path(P, f'experiments/dinol-177/{split}/report.json'))['predictions_sha256']
        raw=load(_repo_path(P, f'artifacts/cnn-seed-mean-133/{split}/predictions.npz'),['ids','fold','raw'])
        comp=load(_repo_path(P, f'artifacts/component-ablation-121/{split}/predictions.npz'),['ids','fold','short'])
        for q in [dino,raw,comp]:np.testing.assert_array_equal(q['ids'],ids);np.testing.assert_array_equal(q['fold'],old['fold'])
        np.testing.assert_array_equal(dino['blend_L_MLP'],old['reference021'])
        for k in range(5):
            V=np.flatnonzero(old['fold']==k);cal=read(_repo_path(P, f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/report.json'))['outer_calibrator']
            np.testing.assert_array_equal(old['coefficient'][V],np.full(len(V),.243*cal['slope']))
            base=.9*(cal['target_center']+cal['slope']*(raw['raw'][V]-cal['prediction_center']))+.1*comp['short'][V]
            cor=dino['reference018'][V]-base;newraw=raw['raw'][V]+.3*(pred[V]-old['old_KRR'][V])
            full=.9*(.9*(cal['target_center']+cal['slope']*(newraw-cal['prediction_center']))+.1*comp['short'][V]+cor)+.1*dino['L_MLP'][V]
            np.testing.assert_allclose(full,old['ensemble_D_mean_only'][V],rtol=0,atol=1e-10)
        arrays(dest/'OOF.npz',**old);data[split]=old;validation.append({'split':split,'folds':split_checks,'OOF_SHA256':sha(dest/'OOF.npz')});emit({'split_complete':split,'main_fits':main_fits,'poison_fits':poison_fits,'seconds':time.monotonic()-START})
    assert main_fits==420 and poison_fits==168 and FITS==588
    write(_repo_path(R, 'validation.json'),{'passed':True,'main_fits':main_fits,'poison_fits':poison_fits,'models_reloaded':15,'independent_maxerror':max(errors),'full_outer_label_poison_paths':3,'full_outer_feature_poison_paths':3,'frozen_ensemble_algebra_verified':True,'records':validation})
    # No thresholds or candidate changes after this point.
    score(data,ids,y,missing,A,D,changes)
    for f,h in protected.items():assert sha(_repo_path(P, f))==h,f
    for f,h in source.items():assert sha(_repo_path(R, f))==h,f
    assert stat(best)==best_stat and not BLOCKED
    resource_check()
    write(_repo_path(R, 'completion.json'),{'passed':True,'main_KRR_fits':main_fits,'verification_KRR_fits':poison_fits,'neural_fits':0,'source_and_input_hashes_unchanged':True,'best021_stat_unchanged':True,'outside_writes':False,'blocked_access':BLOCKED,'read_paths':sorted(READS),'CPU_thread_limits':threadpool_info(),'elapsed_seconds':time.monotonic()-START,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cost':0,'test_holdout_full500_submission_install_network':False})
    emit({'completed':True,'decision':read(_repo_path(R, 'results.json'))['decision'],'primary':read(_repo_path(R, 'results.json'))['comparisons']['pooled_ID_meanSE']['frozen_ensemble']['D-A']['all400']})


def score(data,ids,y,missing,A,D,changes):
    rng=np.random.default_rng(202610218);draws=rng.integers(0,400,(10000,400));counts=np.stack([np.bincount(a,minlength=400) for a in draws])
    rest=np.flatnonzero(~missing);dr=rng.integers(0,len(rest),(10000,len(rest)));cr=np.stack([np.bincount(a,minlength=len(rest)) for a in dr])
    prior=load(_repo_path(P, 'experiments/grain-validity-218/paired-results.npz'),['draws','other397_draws']);np.testing.assert_array_equal(draws,prior['draws']);np.testing.assert_array_equal(dr,prior['other397_draws'])
    results={};boots={};perid={};arms={'A':'A_current','B':'B_validity','C':'C_valid_mean','D':'D_mean_only'}
    comparisons=[('D-A','A','D'),('D-B','B','D'),('D-C','C','D'),('C-B','B','C'),('B-A','A','B'),('C-A','A','C')]
    oldresults=read(_repo_path(P, 'experiments/grain-validity-218/results.json'))['results']
    def compare(a,b,tag,fold=None):
        if a.ndim==1:a=a[None];b=b[None]
        am=a.mean(0);bm=b.mean(0);boot=np.sqrt(counts@am/400)-np.sqrt(counts@bm/400);rb=np.sqrt(cr@am[rest]/397)-np.sqrt(cr@bm[rest]/397)
        out={}
        for name,mask in [('all400',np.ones(400,bool)),('empty3',missing),('other397',~missing)]:
            out[name]={'n':int(mask.sum()),'baseline_RMSE':float(np.sqrt(am[mask].mean())),'candidate_RMSE':float(np.sqrt(bm[mask].mean())),'RMSE_gain':float(np.sqrt(am[mask].mean())-np.sqrt(bm[mask].mean())),'SSE_gain':float((am[mask]-bm[mask]).sum()),'improved_IDs':int((am[mask]>bm[mask]).sum()),'worsened_IDs':int((am[mask]<bm[mask]).sum())}
        out['all400']['conditional_CI95']=np.quantile(boot,[.025,.975]).tolist();out['other397']['conditional_CI95']=np.quantile(rb,[.025,.975]).tolist();out['empty3']['caution']='n=3 descriptive only; no CI or inference.'
        if fold is not None:out['fold_gains']=[float(np.sqrt(am[fold==k].mean())-np.sqrt(bm[fold==k].mean())) for k in range(5)]
        for i,ix in enumerate(draws[:23]):assert abs((np.sqrt(a[:,ix].mean())-np.sqrt(b[:,ix].mean()))-boot[i])<1e-11
        boots[tag]=boot;perid[tag]=am-bm;return out
    for split in [*SPLITS,'pooled_ID_meanSE']:
        results[split]={}
        for mode,prefix in [('KRR',''),('frozen_ensemble','ensemble_')]:
            results[split][mode]={}
            for label,a,b in comparisons:
                parts=SPLITS if split=='pooled_ID_meanSE' else [split]
                ae=np.stack([(y-data[s][prefix+arms[a]])**2 for s in parts]);be=np.stack([(y-data[s][prefix+arms[b]])**2 for s in parts])
                q=compare(ae,be,split+'_'+mode+'_'+label,None if split=='pooled_ID_meanSE' else data[split]['fold']);results[split][mode][label]=q
                if label in ('C-B','B-A','C-A'):
                    old=oldresults[split][mode][label]
                    for group in ('all400','empty3','other397'):
                        for key in ('baseline_RMSE','candidate_RMSE','RMSE_gain'):assert abs(q[group][key]-old[group][key])<1e-11
                    for group in ('all400','other397'):np.testing.assert_allclose(q[group]['conditional_CI95'],old[group]['conditional_CI95'],rtol=0,atol=1e-11)
    primary=results['pooled_ID_meanSE']['frozen_ensemble']['D-A'];gains=[results[s]['frozen_ensemble']['D-A']['all400']['RMSE_gain'] for s in SPLITS]
    gates={'practical_gain_at_least_0.02':primary['all400']['RMSE_gain']>=.02,'all_split_positive':all(v>0 for v in gains),'remaining397_nonnegative':primary['other397']['RMSE_gain']>=0,'conditional_CI_lower_positive':primary['all400']['conditional_CI95'][0]>0}
    decision='screen_signal_only_no_promotion' if all(gates.values()) else 'reject_mean_only_stop_no_tuning_keep_best021'
    # Paired factorial interaction on additive SE scale, not differences of RMSE.
    interaction={}
    for mode in ('KRR','frozen_ensemble'):
        q=perid['pooled_ID_meanSE_'+mode+'_D-A']-perid['pooled_ID_meanSE_'+mode+'_C-B']
        interaction[mode]={'definition':'gain(D-A)-gain(C-B) on MSE scale; positive means correction helps more without flags.','mean_MSE_gain_difference':float(q.mean()),'conditional_CI95':np.quantile(counts@q/400,[.025,.975]).tolist()}
    rows=[]
    for i,ID in enumerate(ids):
        rows.append({'ID':str(ID),'empty':bool(missing[i]),'old_HP':float(A[i,0]),'D_HP':float(D[i,0]),'splits':{s:{'fold':int(data[s]['fold'][i]),'KRR':{a:float(data[s][key][i]) for a,key in arms.items()},'frozen_ensemble':{a:float(data[s]['ensemble_'+key][i]) for a,key in arms.items()},'D_minus_A_squared_error_gain':float((y[i]-data[s]['ensemble_A_current'][i])**2-(y[i]-data[s]['ensemble_D_mean_only'][i])**2)} for s in SPLITS}})
    write(_repo_path(R, 'ID-effects.json'),rows);write(_repo_path(R, 'empty3-effects.json'),[row for row in rows if row['empty']])
    # Read-only target-balance diagnostic; no new split, fitting or seed search.
    edges=np.quantile(y,[.2,.4,.6,.8]);bins=np.searchsorted(edges,y,side='right')
    target_balance={s:[{'fold':k,'n':int((data[s]['fold']==k).sum()),'mean':float(y[data[s]['fold']==k].mean()),'std':float(y[data[s]['fold']==k].std()),'quintile_counts':np.bincount(bins[data[s]['fold']==k],minlength=5).tolist()} for k in range(5)] for s in SPLITS}
    write(_repo_path(R, 'target-balance-readonly.json'),{'bin_edges':edges.tolist(),'global_mean':float(y.mean()),'global_std':float(y.std()),'bins':'Descriptive global development quintiles; not model features or split selection.','partitions':target_balance,'new_split_created':False})
    arrays(_repo_path(R, 'paired-statistics.npz'),ids=ids,empty=missing,**perid,**{k+'_boot':v for k,v in boots.items()})
    result={'posthoc':True,'independent_confirmation':False,'decision':decision,'gates_locked_before_training':gates,'comparisons':results,'factorial_MSE_interaction':interaction,'D_hyperparameter_changed_vs_A_folds':changes,'prior218_A_B_C_scores_and_CIs_reproduced':True,'no_additional_conditions_or_tuning':True,'limitations':['Same400 adaptive reuse;3 partitions not1200 independent rows.','Conditional bootstrap omits fitting/selection uncertainty.','Frozen downstream only; no complete nested pipeline or deployable candidate.','3 missing images too few for subgroup generalization.']}
    write(_repo_path(R, 'results.json'),result)
    lines=['# Mean-only control: '+decision,'','218 A/B/C 확인 뒤 추가한 사후 탐색이다. 독립 확증이 아니다. best021 유지.','', '| 분할 | KRR D-A gain | 고정 결합 D-A gain | 고정 결합 CI95 |', '|---|---:|---:|---|']
    for s in [*SPLITS,'pooled_ID_meanSE']:
        a=results[s]['KRR']['D-A']['all400'];b=results[s]['frozen_ensemble']['D-A']['all400'];lines.append(f"| {s} | {a['RMSE_gain']:+.9f} | {b['RMSE_gain']:+.9f} | {b['conditional_CI95']} |")
    lines+=['','| pooled 고정 결합 D-A | n | baseline RMSE | mean-only RMSE | gain | improved/worsened IDs |','|---|---:|---:|---:|---:|---:|']
    for group in ('all400','empty3','other397'):
        q=primary[group];lines.append(f"| {group} | {q['n']} | {q['baseline_RMSE']:.9f} | {q['candidate_RMSE']:.9f} | {q['RMSE_gain']:+.9f} | {q['improved_IDs']}/{q['worsened_IDs']} |")
    lines+=['','D는 A의 HP 열 3행만 바꾸고 flag 2열은 0으로 유지했다. 동일 모델·9개 grid·inner seed·ID·fold를 사용했다. A/B/C는 재학습 없이 저장 OOF를 재사용했고 기존 점수와 CI를 재현했다.','', '새 본실험 KRR 420회, 검증 KRR 168회. 15개 모델 exact reload 및 독립 kernel solve, 모든 fit의 train-only 통계, 각 분할 fold0 전체 nested label/feature poison 검증 통과. 세부 수치는 validation.json 참조.','', '사전 practical gate: pooled fixed-ensemble gain>=0.02, 세 분할 양수, remaining397 비악화, 조건부CI 하한>0. '+json.dumps(gates,ensure_ascii=False),'', '3장/397장과 A/B/C/D 전체 비교는 results.json 및 empty3-effects.json/ID-effects.json에 있다. 평균 squared-error 기준 factorial interaction도 기록했다. 결과는 후단을 고정한 proxy이며 전체 재학습 효과가 아니다.','', '추가 threshold/feature/weight 탐색 없이 종료한다. rawdata/test/holdout/전체500/새 제출 CSV/설치/외부통신/비용 없음. 기존 소스·입력 해시 및 best021 파일 stat 보존은 completion.json에 기록한다.','', '타깃 층화 문제는 기존 fold 분포만 읽기 진단했다. 새로운 분할이나 점수 기반 seed 선택은 없다.']
    (_repo_path(R, 'REPORT.md')).write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if not (_repo_path(R, 'failure.json')).exists():write(_repo_path(R, 'failure.json'),{'error':repr(exc),'fits_completed':FITS,'blocked':BLOCKED,'seconds':time.monotonic()-START})
        raise
