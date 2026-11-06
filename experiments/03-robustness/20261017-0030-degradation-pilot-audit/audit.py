"""Independent, train-only audit and source-isolated control repair."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import csv,json,hashlib,time,resource,argparse,math
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=_RepoPath(__file__).resolve().parents[3];EXP=_RepoPath(__file__).resolve().parents[2];PILOT=_repo_path(EXP, 'infomatched-pilot-20261016')
NAMES=['original','20261031','20261101'];GR={'ridge':[.1,.3,1,3,10,30,100,1000],'log':[.3,1,2,3,10,30,100]};SEED=2026101701
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (_repo_path(HERE, name)).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def arrays(name,**x):
    assert not (_repo_path(HERE, name)).exists();np.savez_compressed(_repo_path(HERE, name),**x)
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1:];write=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if write and not p.is_relative_to(HERE):raise RuntimeError('Write outside audit: '+str(p))
        if p.is_relative_to(_repo_path(ROOT, 'data/test')) or '-test.' in p.name or p.name=='sample_submission.csv' or p.name.startswith('submission-candidate-'):raise RuntimeError('No test pixels/predictions in audit')
def rmse(p,y):return float(np.sqrt(np.mean((p-y)**2)))
def load():
    labels={r['ID']:float(r['hardness']) for r in csv.DictReader((_repo_path(ROOT, 'data/train.csv')).open())}
    old=dict(np.load(_repo_path(EXP, 'highnoise-relational-20261016/inputs.npz')));ids=old['ids'][old['dev']].astype(str);y=np.array([labels[i] for i in ids]);assert np.array_equal(y,old['y'][old['dev']])
    f=np.load(_repo_path(EXP, 'push-800/feat-u1dmavgH2Z-anghcor-train.npz'),allow_pickle=True);keys=list(map(str,f['keys']));allids=f['ids'].astype(str);X=f['X'].astype(float);ni=keys.index('noise');pos={v:i for i,v in enumerate(allids)}
    row=np.array([pos[i] for i in ids]);high=(old['noise'][old['dev']]>=12)&~old['failed'];assert high.sum()==124
    def copies(new):
        xx=[];ss=[];paths=[]
        for name in (['v1a','v2a'] if new else ['deg','deg2','deg3','deg4']):
            path=_repo_path(PILOT, f'feat/synthrows-{name}-u1dmavgH2Z-anghcor.npz') if new else _repo_path(EXP, f'push-1200/synthrows-{name}-u1dmavgH2Z-anghcor.npz');d=np.load(_repo_io(path),allow_pickle=True);assert list(map(str,d['keys']))==keys
            keep=d['X'][:,ni].astype(float)*255>=12;xx.append(d['X'].astype(float)[keep]);ss.extend(d['ids'].astype(str)[keep]);paths.append(str(path.relative_to(ROOT)))
        return np.vstack(xx),np.array(ss),paths
    PX,PS,prodpaths=copies(False);NX,NS,newpaths=copies(True);R=np.load(_repo_path(PILOT, 'work/oof-ridge.npz'));pred={};fold=[];splitmeta=[]
    for name in NAMES:
        rid=R[f'ids_{name}'].astype(str);assert len(set(rid))==400 and set(rid)==set(ids);ix=np.array([{v:i for i,v in enumerate(rid)}[v] for v in ids]);assert np.array_equal(R[f'y_{name}'][ix],y);fold.append(R[f'fold_{name}'][ix]);splitmeta.append(dict(ids=rid,fold=R[f'fold_{name}'],y=R[f'y_{name}']))
        for kind in GR:
            for arm in ['R0','R1','R2','R3','RD','RS']:pred.setdefault(kind+'_'+arm,[]).append(R[f'{kind}_{arm}_{name}'][ix])
    for arm in ['T0','T1','T2','TD']:
        path=_repo_path(EXP, 'stepchange-20261016/work/e1/oof-F0.npz') if arm=='T0' else _repo_path(PILOT, f'work/oof-tab-{arm}.npz');t=np.load(_repo_io(path));pp=[]
        if arm=='T0':assert list(map(str,t['cols']))==keys and str(t['arm'])=='F0'
        for i,name in enumerate(NAMES):
            tid=t[f'{name}_ids'].astype(str);assert set(tid)==set(ids) and len(set(tid))==400;ix=np.array([{v:j for j,v in enumerate(tid)}[v] for v in ids]);assert np.array_equal(t[f'{name}_y'][ix],y)
            if arm=='T0':assert np.array_equal(t[f'{name}_fold'][ix],fold[i])
            pp.append(t[f'{name}_g'][ix])
        pred['tab_'+arm]=pp
    pred={k:np.array(v) for k,v in pred.items()};assert all(np.isfinite(v).all() for v in pred.values());assert np.array_equal(fold[0],old['fold'])
    return dict(ids=ids,y=y,high=high,fold=np.array(fold),pred=pred,R=R,splitmeta=splitmeta,labels=labels,allids=allids,X=X,keys=keys,ni=ni,pos=pos,PX=PX,PS=PS,NX=NX,NS=NS,old=old,used=prodpaths+newpaths)
def comparison(new,ref,y,fold,mask):
    sn=((new-y)**2)[:,mask];sr=((ref-y)**2)[:,mask];a=sn.mean(0);b=sr.mean(0);delta=a-b;draw=np.random.RandomState(0).randint(0,len(delta),(4000,len(delta)));boot=np.sqrt(b[draw].mean(1))-np.sqrt(a[draw].mean(1));order=np.argsort(delta)
    return dict(n=int(mask.sum()),new_rmse=float(np.sqrt(a.mean())),reference_rmse=float(np.sqrt(b.mean())),gain=float(np.sqrt(b.mean())-np.sqrt(a.mean())),gain_CI95=np.quantile(boot,[.025,.975]).tolist(),dMSE=float(delta.mean()),dMSE_CI95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),se=float(delta.std(ddof=1)/np.sqrt(len(delta))),bootstrap_fraction_better=float(np.mean(delta[draw].mean(1)<0)),per_partition_dMSE=(sn-sr).mean(1).tolist(),per_fold_original_dMSE=[float((sn[0]-sr[0])[fold[0][mask]==k].mean()) for k in range(5)],
        drop={str(k):dict(dMSE=float(delta[order[k:]].mean()),gain=float(np.sqrt(b[order[k:]].mean())-np.sqrt(a[order[k:]].mean()))) for k in [5,10]},
        influential_helping_positions=np.flatnonzero(mask)[order[:10]].tolist())
def score(d,pred):
    pairs=[('ridge_R3','ridge_R1'),('log_R3','log_R1'),('tab_T2','tab_T0'),('ridge_R3','ridge_RD'),('log_R3','log_RD'),('tab_T2','tab_TD'),('ridge_R1','ridge_R0'),('log_R1','log_R0'),('tab_T1','tab_T0')]
    for kind in GR:pairs.extend([(kind+'_R3',kind+'_RS'),(kind+'_R2',kind+'_R1')])
    if 'ridge_RS_fixed' in pred:
        for kind in GR:pairs.append((kind+'_R3',kind+'_RS_fixed'))
    return {a+' vs '+b:dict(high=comparison(pred[a],pred[b],d['y'],d['fold'],d['high']),overall=comparison(pred[a],pred[b],d['y'],d['fold'],np.ones(400,bool))) for a,b in pairs}
def inspect():
    assert (_repo_path(HERE, 'starting-state.json')).exists();d=load();result=score(d,d['pred']);saved=json.loads((_repo_path(PILOT, 'work/pilot-results.json')).read_text());checks=[]
    for key,r in result.items():
        v=saved[key];q=r['high']
        for a,b in [('dMSE','dMSE'),('se','se'),('bootstrap_fraction_better','P_better')]:assert abs(q[a]-v[b])<1e-10,(key,a,q[a],v[b])
        assert np.max(abs(np.array(q['per_partition_dMSE'])-v['per_partition']))<1e-10;assert np.max(abs(np.array(q['per_fold_original_dMSE'])-v['per_fold_original']))<1e-10
        for n in [5,10]:assert abs(q['drop'][str(n)]['dMSE']-v[f'drop_top{n}'])<1e-10
        checks.append(key)
    src=sorted(set(d['NS']));perm=dict(zip(src,np.random.RandomState(20261016).permutation(src)));hold=set(d['allids'])-set(d['ids']);leaks=[];groups=[]
    for p,name in enumerate(NAMES):
        for k in range(5):
            tr=set(d['ids'][d['fold'][p]!=k]);va=set(d['ids'][d['fold'][p]==k]);newsrc=d['NS'][np.isin(d['NS'],list(tr))];don=np.array([perm[i] for i in newsrc]);row=dict(partition=name,fold=k,new_copy_rows=len(newsrc),donor_rows_outside_outer_training=int((~np.isin(don,list(tr))).sum()),donor_rows_from_outer_validation=int(np.isin(don,list(va)).sum()),donor_rows_from_holdout100=int(np.isin(don,list(hold)).sum()),unique_donors_outside=sorted(set(don)-tr));leaks.append(row)
            for tag,ss in [('production',d['PS']),('new',d['NS'])]:
                used=ss[np.isin(ss,list(tr))];assert set(used)<=tr and set(used).isdisjoint(va|hold);groups.append(dict(partition=name,fold=k,kind=tag,rows=len(used),sources=len(set(used)),excluded_sources_present=0))
    feature_noise=np.nan_to_num(d['X'][:,d['ni']]*255,nan=0);rawnoise=np.load(_repo_path(EXP, 'push-300/noise_train.npy'))*255;assert len(rawnoise)==500
    noise_ids=set(d['allids'][feature_noise>=12]);raw_ids=set(d['allids'][rawnoise>=12]);d2=dict(actual_real_noisy_rows=int((feature_noise>=12).sum()),copy_sources_per_draw=min(161,len(set(d['NS']))),raw_noise_pool=int((rawnoise>=12).sum()),feature_and_raw_noise_membership_differences=sorted(noise_ids^raw_ids),stored_key='real_noisy_n161',actual_sample_sizes_match=int((feature_noise>=12).sum())==161)
    s=dict(recomputed_comparisons=result,saved_statistics_exact=True,verified_pairs=checks,primary_copy_source_exclusion=groups,shuffled_control_leakage=leaks,D2_population_audit=d2,
        baseline082_reference=dict(overall_rmse=rmse(d['old']['base082'],d['y']),high_rmse=rmse(d['old']['base082'][d['high']],d['y'][d['high']]),role='Frozen complete-ensemble reference, not directly comparable to the standalone partner scores as a candidate test.'),
        limitations=['All-train renderer calibration is exploratory; its bias need not be optimistic.','All partitions reuse the same400 development labels.','Original RS is invalid and is not used to certify signal.','D2 prediction quality on different populations is not an information-equivalence test.'])
    save('audit-before-refits.json',s);arrays('aligned-original-oof.npz',ids=d['ids'],y=d['y'],high=d['high'],fold=d['fold'],**d['pred']);save('code-lock.json',dict(source_sha256=sha(__file__),plan_sha256=sha(_repo_path(HERE, 'PLAN.md')),seed=SEED,grids=GR,primary_results_already_known=True))
    print('SAVED STATISTICS REPRODUCED',len(checks),'comparisons');print('D2',d2);print('RS outside donors',min(x['donor_rows_outside_outer_training'] for x in leaks),max(x['donor_rows_outside_outer_training'] for x in leaks))
def estimator(kind,alpha):return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True) if kind=='ridge' else SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=alpha))
def nested_fit(kind,Xr,yr,sr,nzr,Xc,yc,sc,isnew,seed,Xq,shuffle=False):
    X=np.vstack([Xr,Xc]);y=np.r_[yr,yc];src=np.r_[sr,sc];real=np.r_[np.ones(len(yr),bool),np.zeros(len(yc),bool)];new=np.r_[np.zeros(len(yr),bool),isnew];nz=np.r_[nzr,np.full(len(yc),99.)];assert set(sc)<=set(sr)
    trace=[]
    def targets(ix,offset):
        yy=y[ix].copy();ss=src[ix];nn=new[ix];rr=real[ix]
        if shuffle:
            pool=sorted(set(ss[nn]));mapping=dict(zip(pool,np.random.RandomState(seed+offset).permutation(pool)));known=dict(zip(ss[rr],yy[rr]));donors=np.array([mapping[v] for v in ss[nn]])
            assert set(donors)<=set(ss[rr]) and set(ss).isdisjoint(set(src)-set(ss));yy[nn]=[known[v] for v in donors]
            trace.append(dict(offset=offset,training_sources=len(set(ss)),new_sources=len(pool),donor_sources=len(set(donors)),outside_donors=0,donor_hash=hashlib.sha256(donors.tobytes()).hexdigest()))
        return np.log(yy) if kind=='log' else yy
    splits=list(GroupKFold(5).split(X,y,src));fit_targets=[targets(tr,j) for j,(tr,va) in enumerate(splits)];best=None;grid=[]
    for a in GR[kind]:
        error=[]
        for j,(tr,va) in enumerate(splits):
            assert set(src[tr]).isdisjoint(src[va]);m=estimator(kind,a).fit(X[tr],fit_targets[j]);sel=va[real[va]&(nz[va]>=12)]
            if len(sel):
                p=m.predict(X[sel]);p=np.exp(p) if kind=='log' else p;error.extend((p-y[sel])**2)
        loss=float(np.mean(error));grid.append(dict(alpha=a,inner_MSE=loss))
        if best is None or loss<best[0]:best=(loss,a)
    m=estimator(kind,best[1]).fit(X,targets(np.arange(len(y)),9));p=m.predict(Xq);return (np.exp(p) if kind=='log' else p),dict(alpha=best[1],grid=grid,donor_checks=trace,n_training_sources=len(sr),n_rows=len(y))
def refit():
    assert sha(__file__)==json.loads((_repo_path(HERE, 'code-lock.json')).read_text())['source_sha256'];d=load();t0=time.monotonic();pred={};records=[];fixture=None;maxerr=0.
    for part,name in enumerate(NAMES):
        meta=d['splitmeta'][part];ids=meta['ids'];y=meta['y'];fold=meta['fold'];X=d['X'][[d['pos'][i] for i in ids]];nz=np.nan_to_num(X[:,d['ni']]*255,nan=0)
        for kind in GR:
            for arm in ['R1','R3','RD','RS_fixed']:
                out=np.zeros(400)
                for k in range(5):
                    if time.monotonic()-t0>1200:raise RuntimeError('20-minute CPU refit budget reached')
                    tr=fold!=k;va=fold==k;trainids=ids[tr];kp=np.isin(d['PS'],trainids);kn=np.isin(d['NS'],trainids);PX=d['PX'][kp];PS=d['PS'][kp];NX=d['NX'][kn];NS=d['NS'][kn]
                    if arm=='R1':XC=PX;SC=PS;new=np.zeros(len(PS),bool)
                    else:
                        extra=d['X'][[d['pos'][s] for s in NS]] if arm=='RD' else NX;XC=np.vstack([PX,extra]);SC=np.r_[PS,NS];new=np.r_[np.zeros(len(PS),bool),np.ones(len(NS),bool)]
                    yc=np.array([d['labels'][s] for s in SC]);seed=SEED+100*part+10*k;assert set(SC)<=set(trainids) and set(SC).isdisjoint(ids[va])
                    p,info=nested_fit(kind,X[tr],y[tr],trainids,nz[tr],XC,yc,SC,new,seed,X[va],arm=='RS_fixed');out[va]=p
                    if arm!='RS_fixed':
                        original=d['R'][f'{kind}_{arm}_{name}'][va];error=float(np.max(abs(p-original)));maxerr=max(maxerr,error);assert error<1e-7,(part,kind,arm,k,error);assert info['alpha']==d['R'][f'{kind}_{arm}_{name}_alphas'][k]
                    else:error=None
                    records.append(dict(partition=name,kind=kind,arm=arm,fold=k,source_ids=trainids.tolist(),validation_ids=ids[va].tolist(),max_replay_difference=error,**info))
                    if part==0 and kind=='ridge' and arm=='RS_fixed' and k==0:fixture=(kind,X[tr],y[tr],trainids,nz[tr],XC,yc,SC,new,seed,X[va],p.copy(),ids[va].copy())
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2**30:raise RuntimeError('Own1GiB RSS cap exceeded')
                ix=np.array([{v:i for i,v in enumerate(ids)}[v] for v in d['ids']]);pred.setdefault(kind+'_'+arm,[]).append(out[ix]);print('REFIT',name,kind,arm,round(time.monotonic()-t0,1),'s',flush=True)
    # Actual complete repeat: outside labels poisoned, fitting can read no dataset.
    kind,xr,yr,sr,nzr,xc,yc,sc,new,seed,xq,original,vids=fixture;poison=dict(d['labels']);outside=set(poison)-set(sr)
    for s in outside:poison[s]+=10000
    def blocked_reads(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0]));mode,flags=args[1:];write=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if not write and p.suffix in ['.npz','.npy','.csv','.png','.joblib','.pt']:raise RuntimeError('Dataset read during isolated replay')
    # Hook is installed only for the isolated replay and remaining writes need no input reads.
    sys.addaudithook(blocked_reads);p,info=nested_fit(kind,xr,np.array([poison[s] for s in sr]),sr,nzr,xc,np.array([poison[s] for s in sc]),sc,new,seed,xq,True);assert np.array_equal(p,original)
    pred={k:np.array(v) for k,v in pred.items()};combined={**d['pred'],**pred};result=score(d,combined)
    arrays('refit-oof.npz',ids=d['ids'],y=d['y'],high=d['high'],fold=d['fold'],**pred);save('refit-verification.json',dict(complete=True,models=len(records),replayed_valid_models=90,repaired_RS_models=30,max_prediction_difference=maxerr,tolerance=1e-7,all_selected_alphas_identical=True,records=records,seconds=time.monotonic()-t0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,poison_replay=dict(outside_labels_changed=len(outside),predictions_exact=True,dataset_reads_blocked=True,training_sources=len(sr)),source_sha256=sha(__file__)));save('audited-results.json',result)
    decisions={}
    for main,duplicate,shuffle in [('ridge_R3 vs ridge_R1','ridge_R3 vs ridge_RD','ridge_R3 vs ridge_RS_fixed'),('log_R3 vs log_R1','log_R3 vs log_RD','log_R3 vs log_RS_fixed'),('tab_T2 vs tab_T0','tab_T2 vs tab_TD',None)]:
        r=result[main]['high'];checks=dict(dMSE_negative=r['dMSE']<0,bootstrap_fraction_at_least95=r['bootstrap_fraction_better']>=.95,all_partitions_benefit=all(x<0 for x in r['per_partition_dMSE']),four_original_folds=sum(x<0 for x in r['per_fold_original_dMSE'])>=4,top5_removed_benefit=r['drop']['5']['gain']>0,beats_duplicate=result[duplicate]['high']['gain']>0,beats_valid_shuffle=result[shuffle]['high']['gain']>0 if shuffle else None)
        decisions[main]=dict(advance=all(v for v in checks.values() if v is not None),checks=checks)
    save('decision.json',dict(stageB_warranted=any(x['advance'] for x in decisions.values()),primary=decisions,incumbent='082',rollback='076',new_candidate=False,notes='No Tab shuffled-label control was planned; applicable controls are used, and every primary already fails its benefit criteria.'))
    with (_repo_path(HERE, 'comparison.csv')).open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['comparison','n_high','reference_high_RMSE','new_high_RMSE','gain','CI95_low','CI95_high','bootstrap_fraction_better','drop_top5_gain','drop_top10_gain','reference_overall_RMSE','new_overall_RMSE'])
        for key in [*decisions,'ridge_R3 vs ridge_RD','log_R3 vs log_RD','ridge_R3 vs ridge_RS_fixed','log_R3 vs log_RS_fixed']:
            r=result[key]['high'];q=result[key]['overall'];w.writerow([key,r['n'],r['reference_rmse'],r['new_rmse'],r['gain'],*r['gain_CI95'],r['bootstrap_fraction_better'],r['drop']['5']['gain'],r['drop']['10']['gain'],q['reference_rmse'],q['new_rmse']])
    with (_repo_path(HERE, 'per-source.csv')).open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','hardness','eligible_high','ridge_R3_minus_R1_MSE','log_R3_minus_R1_MSE','tab_T2_minus_T0_MSE'])
        dd=[((combined[a]-d['y'])**2-(combined[b]-d['y'])**2).mean(0) for a,b in [('ridge_R3','ridge_R1'),('log_R3','log_R1'),('tab_T2','tab_T0')]]
        for i in range(400):w.writerow([d['ids'][i],d['y'][i],int(d['high'][i]),*[v[i] for v in dd]])
    print('REFIT VERIFIED',len(records),'models; max_error',maxerr,'seconds',time.monotonic()-t0,flush=True);print('STAGE_B',any(x['advance'] for x in decisions.values()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inspect','refit']);args=ap.parse_args();sys.addaudithook(guard)
    with threadpool_limits(limits=2):globals()[args.action]()
