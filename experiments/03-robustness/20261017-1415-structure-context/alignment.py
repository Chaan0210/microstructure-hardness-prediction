"""Fixed target-aligned projection with nested compression/zero selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import readout as r
import encoder as e
import evaluate as ev
import numpy as np,json,time,sys,ast,gc,math,csv
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
import torch
OUT=e.HERE/'alignment-followup';ARMS=r.ARMS
def save(name,x):e.save(_repo_io('alignment-followup/'+name),x)
def arrays(name,**x):e.arrays('alignment-followup/'+name,**x)
def teacher(X,y):
    p=np.zeros(len(y));ids=np.arange(len(y))
    for i in ids:
        tr=ids!=i;m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100)).fit(X[tr],y[tr]);p[i]=m.predict(X[i:i+1])[0]
    return y-p,p
def fit():
    assert json.loads((e.HERE/'readout-verification.json').read_text())['all_passed'];d=r.physical();graphs=json.loads((e.OLD/'graphs.json').read_text());old=json.loads((e.HERE/'readouts-complete.json').read_text());replay=json.loads((e.HERE/'encoder-repeatability.json').read_text());start=time.monotonic();remaining=min(720.,7200-old['total_compute_seconds']-replay['seconds']);assert remaining>600
    save('code-lock.json',dict(source_sha256=e.sha(__file__),plan_sha256=e.sha(OUT/'PLAN.md'),parent_readout_sha256=e.sha(e.HERE/'readout.py'),parent_decision_sha256=e.sha(e.HERE/'decision.json'),source_hardness_scores_seen='Parent PCA results known; no PLS outcome inspected',remaining_compute_seconds=remaining));records=[]
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=OUT/f'readouts/fold{k}-{stage}';folder.mkdir(parents=True,exist_ok=False);X=d['X'][tr];Q=d['X'][q];y=d['y'][tr];res,loo=teacher(X,y);assert np.isfinite(res).all();rs=float(res.std());assert rs>0;target=(res-res.mean())/rs
            arrays(str((folder/'teacher.npz').relative_to(OUT)),train=tr,query=q,labels=y,loo_prediction=loo,residual=res);pred={a:[] for a in ARMS}
            for arm in ARMS:
                for repeat in range(2):
                    path=e.HERE/(f'encoders/fold{k}-{stage}-seed{repeat}-'+('spatial' if arm=='random' else arm));f=np.load(_repo_io(path/('random-features.npz' if arm=='random' else 'features.npz')));assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q);sc=StandardScaler().fit(f['train_features']);zt=sc.transform(f['train_features']);zq=sc.transform(f['query_features']);pls=PLSRegression(n_components=16,scale=False,max_iter=500,tol=1e-6).fit(zt,target);pt=pls.transform(zt);pq=pls.transform(zq);xt=np.c_[X,pt];xq=np.c_[Q,pq];sub=folder/f'{arm}-seed{repeat}';sub.mkdir()
                    arrays(str((sub/'context.npz').relative_to(OUT)),X=xt,y=y,Q=xq,train=tr,query=q,scaler_mean=sc.mean_,scaler_scale=sc.scale_,pls_mean=pls._x_mean,pls_scale=pls._x_std,pls_rotations=pls.x_rotations_,teacher_target=target);pp=[]
                    for j in range(3):
                        if time.monotonic()-start>remaining:raise RuntimeError('Fixed remaining compute budget exhausted')
                        out=sub/f'tab{j}';out.mkdir();pp.append(r.predict_tab(xt,y,xq,100*j+k,out,tr,q))
                    pred[arm].append(np.mean(pp,0));print('PLS',k,stage,arm,repeat,round(time.monotonic()-start,1),'s',flush=True)
            parent=np.load(_repo_io(e.HERE/f'readouts/fold{k}-{stage}/predictions.npz'));assert np.array_equal(parent['train'],tr) and np.array_equal(parent['query'],q);arrays(str((folder/'predictions.npz').relative_to(OUT)),train=tr,query=q,base=parent['base'],**{a:np.array(v) for a,v in pred.items()});records.append(str(folder.relative_to(OUT)))
    save('fit-complete.json',dict(complete=True,models=180,teacher_models=sum(len(g['A'])+len(g['T']) for g in graphs),stages=records,seconds=time.monotonic()-start,parent_compute_seconds=old['total_compute_seconds']+replay['seconds'],total_compute_seconds=old['total_compute_seconds']+replay['seconds']+time.monotonic()-start))
def old_selector():
    src=_repo_path(e.ROOT, 'experiments/rmse-research-20261015/round4-full073/noise_calibration.py');tree=ast.parse(src.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['select','fit_weights']];ns=dict(np=np,STRENGTHS=('global',100,30,10,0));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),ns);return ns['select']
def bases(k,an,labels,noise,select):
    d=np.load(_repo_path(e.ROOT, f'experiments/rmse-research-20261015/round4-full073/aux-mean2/fold{k}-selection.npz'));assert np.array_equal(d['B'],an['B']);good=d['eligible'];w,_=select(d['residual'][good],(labels-d['base073'])[good],noise[good]>=12);assert w[1]==float(an['old_weight']);assert np.array_equal(an['p76B']+an['highB']*w[1]*d['residual'],an['p82B']);out={}
    for j in np.flatnonzero(an['highB']):
        tr=good.copy();tr[j]=False;ww,_=select(d['residual'][tr],(labels-d['base073'])[tr],noise[tr]>=12);out[int(j)]=an['p76B']+an['highB']*ww[1]*d['residual']
    return out
def choose(dirs,y,an,loo):
    high=an['highB'];ix=np.flatnonzero(high);gains={}
    for tag,delta in dirs.items():
        gg=[]
        for j in ix:
            keep=high.copy();keep[j]=False;b=loo[int(j)];w=ev.coefficient(delta,y,b,keep);gg.append((y[j]-b[j])**2-(y[j]-b[j]-w*delta[j])**2)
        gg=np.array(gg);gains[tag]=dict(mean=float(gg.mean()),SE=float(gg.std(ddof=1)/np.sqrt(len(gg))),per_source=gg.tolist())
    best=max(gains,key=lambda tag:gains[tag]['mean']);chosen=None
    if gains[best]['mean']>gains[best]['SE']:
        chosen='PCA' if gains['PCA']['mean']>=gains[best]['mean']-gains[best]['SE'] else 'PLS'
    w=ev.coefficient(dirs[chosen],y,an['p82B'],high) if chosen else 0.
    return chosen,w,dict(choice=chosen,beta=w,LOO_gains=gains,omitted_labels_fit_neither_old_nor_new_weight=True)
def metric(y,b,p,fold):
    r=ev.metrics(y,b,p,fold);draw=np.random.default_rng(2026101791).integers(0,len(y),(10000,len(y)));g=np.sqrt(((b-y)[draw]**2).mean(1))-np.sqrt(((p-y)[draw]**2).mean(1));r['conditional_CI975']=np.quantile(g,[.0125,.9875]).tolist();return r
def evaluate():
    assert json.loads((OUT/'fit-complete.json').read_text())['complete'];d=dict(np.load(_repo_io(e.OLD/'inputs.npz')));graphs=json.loads((e.OLD/'graphs.json').read_text());dev=d['dev'];y=d['y'][dev];base=d['base082'];high=(d['noise'][dev]>=12)&~d['failed'];fold=d['fold'];select=old_selector();p={a:base.copy() for a in ARMS};fixed={a:base.copy() for a in ARMS};seeds={f'{a}_seed{s}':base.copy() for a in ARMS for s in range(2)};records=[]
    for k,g in enumerate(graphs):
        an=np.load(_repo_io(e.OLD/f'anchors/fold{k}.npz'));B=an['B'];ix=an['ix'];old=np.load(_repo_io(e.HERE/f'readouts/fold{k}-selector/predictions.npz'));oldv=np.load(_repo_io(e.HERE/f'readouts/fold{k}-refit/predictions.npz'));new=np.load(_repo_io(OUT/f'readouts/fold{k}-selector/predictions.npz'));newv=np.load(_repo_io(OUT/f'readouts/fold{k}-refit/predictions.npz'));loo=bases(k,an,d['y'][B],d['noise'][B],select);row=dict(fold=k,arms={})
        for arm in ARMS:
            ab={'PCA':old[arm]-old['base'],'PLS':new[arm]-new['base']};av={'PCA':oldv[arm]-oldv['base'],'PLS':newv[arm]-newv['base']};dirs={tag:v.mean(0)[:len(B)] for tag,v in ab.items()};tag,w,rec=choose(dirs,d['y'][B],an,loo)
            if tag:p[arm][ix]=base[ix]+an['highV']*w*av[tag].mean(0)
            wf=ev.coefficient(dirs['PLS'],d['y'][B],an['p82B'],an['highB']);fixed[arm][ix]=base[ix]+an['highV']*wf*av['PLS'].mean(0);sr=[]
            for s in range(2):
                t,beta,rr=choose({tag:v[s,:len(B)] for tag,v in ab.items()},d['y'][B],an,loo)
                if t:seeds[f'{arm}_seed{s}'][ix]=base[ix]+an['highV']*beta*av[t][s]
                sr.append(rr)
            rec['individual_seeds']=sr;rec['fixed_PLS_beta']=wf;row['arms'][arm]=rec
        records.append(row)
    result={}
    for arm in ARMS:
        r0=metric(y,base,p[arm],fold);r0['high_noise']=metric(y[high],base[high],p[arm][high],fold[high]);r0['seeds']=[metric(y,base,seeds[f'{arm}_seed{s}'],fold) for s in range(2)];r0['fixed_PLS_diagnostic']=metric(y,base,fixed[arm],fold);result[arm]=r0
    comp={a:metric(y,p[a],p['spatial'],fold) for a in ['permuted','random']};s=result['spatial'];checks=dict(gain05=s['gain']>=.05,high_gain=s['high_noise']['gain']>0,three_folds=s['improved_folds']>=3,CI975_vs082=s['conditional_CI975'][0]>0,CI975_vs_permuted=comp['permuted']['conditional_CI975'][0]>0,CI975_vs_random=comp['random']['conditional_CI975'][0]>0,top10=s['removed_largest_gains']['10']['gain']>0,both_seeds=all(z['gain']>=0 for z in s['seeds']))
    save('results.json',dict(results=result,spatial_vs_controls=comp,selection=records,limitations='Sequential follow-up after parent failure on reused labels; selection/scalers/teachers/PLS/readouts/weights nested within permitted sources, conditional on historical representations. No independent confirmation.'))
    weights={tag:float(np.mean([z['arms']['spatial']['beta'] if z['arms']['spatial']['choice']==tag else 0 for z in records])) for tag in ['PCA','PLS']};save('decision.json',dict(advance=all(checks.values()),checks=checks,production_weights_if_advanced=weights,incumbent='082',rollback='076',official_promotion=False));arrays('oof.npz',ids=d['ids'][dev],y=y,fold=fold,high=high,baseline082=base,**p,**seeds,**{a+'_fixed_PLS':v for a,v in fixed.items()})
    with (OUT/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_RMSE','gain_vs082','CI975_low','CI975_high','improved_folds','top10_removed_gain','seed0_gain','seed1_gain','fixed_PLS_diagnostic_gain'])
        for a,v in result.items():w.writerow([a,v['candidate_rmse'],v['high_noise']['candidate_rmse'],v['gain'],*v['conditional_CI975'],v['improved_folds'],v['removed_largest_gains']['10']['gain'],*[z['gain'] for z in v['seeds']],v['fixed_PLS_diagnostic']['gain']])
    print((OUT/'comparison.csv').read_text());print('ADVANCE',all(checks.values()),checks)
if __name__=='__main__':
    sys.addaudithook(e.guard);torch.set_num_threads(2)
    if sys.argv[1]=='fit':
        assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.27)
    with e.threadpool_limits(limits=2):globals()[sys.argv[1]]()
