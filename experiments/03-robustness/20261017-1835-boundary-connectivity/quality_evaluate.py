"""Nested source-support routing; prospectively registered before hardness scores."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import stats as s
import numpy as np,json,sys,ast,time,csv
ARMS=['topology','local','hardness','random','raw'];OUT=c.HERE/'quality-policy'
def save(name,x):c.save(_repo_io('quality-policy/'+name),x)
def old_selector():
    src=_repo_path(c.ROOT, 'experiments/rmse-research-20261015/round4-full073/noise_calibration.py');tree=ast.parse(src.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['select','fit_weights']];ns=dict(np=np,STRENGTHS=('global',100,30,10,0));exec(compile(ast.Module(body=body,type_ignores=[]),str(src),'exec'),ns);return ns['select']
def loo_bases(k,an,y,noise,select):
    old=np.load(_repo_path(c.ROOT, f'experiments/rmse-research-20261015/round4-full073/aux-mean2/fold{k}-selection.npz'));assert np.array_equal(old['B'],an['B']);eligible=old['eligible'];w,_=select(old['residual'][eligible],(y-old['base073'])[eligible],noise[eligible]>=12);assert w[1]==float(an['old_weight']) and np.array_equal(an['p76B']+an['highB']*w[1]*old['residual'],an['p82B']);out={}
    for j in np.flatnonzero(an['highB']):
        keep=eligible.copy();keep[j]=False;ww,_=select(old['residual'][keep],(y-old['base073'])[keep],noise[keep]>=12);out[int(j)]=an['p76B']+an['highB']*ww[1]*old['residual']
    return out
def choose(delta,y,an,support,loo):
    masks={'all':an['highB'].copy(),'supported':an['highB']&support};gains={}
    for tag,mask in masks.items():
        rows=[]
        for j in np.flatnonzero(an['highB']):
            keep=mask.copy();keep[j]=False;b=loo[int(j)];beta=s.coefficient(delta,y,b,keep) if keep.any() else 0.;correction=beta*delta[j] if mask[j] else 0.;rows.append((y[j]-b[j])**2-(y[j]-b[j]-correction)**2)
        values=np.array(rows);gains[tag]=dict(mean=float(values.mean()),SE=float(values.std(ddof=1)/np.sqrt(len(values))),source_gains=values.tolist())
    best=max(gains,key=lambda x:gains[x]['mean']);route=None
    if gains[best]['mean']>gains[best]['SE']:route='all' if gains['all']['mean']>=gains[best]['mean']-gains[best]['SE'] else 'supported'
    beta=s.coefficient(delta,y,an['p82B'],masks[route]) if route and masks[route].any() else 0.
    return route,beta,dict(route=route,beta=beta,LOO=gains,n_supported_high_B=int(masks['supported'].sum()))
def metric(y,b,p,fold):
    m=s.metrics(y,b,p,fold,seed=2026101797);draw=np.random.default_rng(2026101797).integers(0,len(y),(10000,len(y)));gain=np.sqrt(((b-y)[draw]**2).mean(1))-np.sqrt(((p-y)[draw]**2).mean(1));m['conditional_CI975']=np.quantile(gain,[.0125,.9875]).tolist();return m
def main():
    assert json.loads((c.HERE/'pre-score-verification.json').read_text())['all_passed'];start=time.monotonic();d=c.inputs(labels=True);dev=d['dev'];base=d['base082'];y=d['y'][dev];fold=d['fold'];high=(d['noise'][dev]>=12)&~d['failed'];target=json.loads((c.HERE/'target-audit.json').read_text())['sources'];support=np.zeros(500,bool)
    for row in target:support[int(row['index'])]=row['boundary_evidence_AUC']>.5
    select=old_selector();p={a:base.copy() for a in ARMS};seed={f'{a}_seed{r}':base.copy() for a in ARMS for r in range(2)};records=[]
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));B=an['B'];V=an['V'];ix=an['ix'];a=np.load(_repo_io(c.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(c.HERE/f'readouts/fold{k}-refit/predictions.npz'));loo=loo_bases(k,an,d['y'][B],d['noise'][B],select);row=dict(fold=k,arms={})
        for arm in ARMS:
            db=(a[arm]-a['base'])[:,:len(B)];dv=v[arm]-v['base'];route,beta,rec=choose(db.mean(0),d['y'][B],an,support[B],loo)
            if route:
                mask=an['highV']&(support[V] if route=='supported' else True);p[arm][ix]=base[ix]+mask*beta*dv.mean(0)
            rr=[]
            for r in range(2):
                tag,b,ss=choose(db[r],d['y'][B],an,support[B],loo)
                if tag:
                    mask=an['highV']&(support[V] if tag=='supported' else True);seed[f'{arm}_seed{r}'][ix]=base[ix]+mask*b*dv[r]
                rr.append(ss)
            rec['seeds']=rr;row['arms'][arm]=rec
        records.append(row)
        if time.monotonic()-start>120:raise RuntimeError('Fixed extra CPU budget exceeded')
    results={}
    for arm in ARMS:
        r=metric(y,base,p[arm],fold);r['high_noise']=metric(y[high],base[high],p[arm][high],fold[high]);r['seeds']=[metric(y,base,seed[f'{arm}_seed{i}'],fold) for i in range(2)];results[arm]=r
    comp={a:metric(y,p[a],p['topology'],fold) for a in ARMS[1:]};r=results['topology'];checks=dict(gain05=r['gain']>=.05,high_gain=r['high_noise']['gain']>0,three_folds=r['improved_folds']>=3,CI975_vs082=r['conditional_CI975'][0]>0,CI975_vs_controls=all(x['conditional_CI975'][0]>0 for x in comp.values()),top10=r['removed_largest_gains']['10']['gain']>0,both_seeds=all(x['gain']>=0 for x in r['seeds']));weights={tag:float(np.mean([row['arms']['topology']['beta'] if row['arms']['topology']['route']==tag else 0 for row in records])) for tag in ['all','supported']}
    save('results.json',dict(results=results,topology_vs_controls=comp,selection=records,n_supported_high=int((support[dev]&high).sum()),threshold=.5,seconds=time.monotonic()-start,limitations='Prospective second policy before outer outcomes. Fixed image-local support, B-only LOO routing/calibration, conditional historical features and previously exposed labels.'))
    save('decision.json',dict(advance=all(checks.values()),checks=checks,production_weights_if_advanced=weights,official_promotion=False));c.arrays('quality-policy/oof.npz',ids=d['ids'][dev],y=y,fold=fold,high=high,support=support[dev],baseline082=base,**p,**seed)
    with (OUT/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','overall_RMSE','high_RMSE','gain_vs082','CI975_low','CI975_high','improved_folds','top10_removed_gain'])
        for a,r in results.items():w.writerow([a,r['candidate_rmse'],r['high_noise']['candidate_rmse'],r['gain'],*r['conditional_CI975'],r['improved_folds'],r['removed_largest_gains']['10']['gain']])
    # Fixed candidate priority, not an outer RMSE competition between policies.
    original=json.loads((c.HERE/'decision.json').read_text());old=np.load(_repo_io(c.HERE/'oof.npz'));mainmetric=metric(y,base,old['topology'],fold);maincomp={a:metric(y,old[a],old['topology'],fold) for a in ARMS[1:]};qualified=original['advance'] and mainmetric['conditional_CI975'][0]>0 and all(x['conditional_CI975'][0]>0 for x in maincomp.values());chosen='all_high' if qualified else ('quality_policy' if all(checks.values()) else None)
    c.save(_repo_io('delivery-decision.json'),dict(selected_protocol=chosen,new_candidate_recommended=chosen is not None,main_meets_original_criteria=original['advance'],main_meets_stricter_family_screen=bool(qualified),main_CI975_vs082=mainmetric['conditional_CI975'],main_CI975_vs_controls={a:x['conditional_CI975'] for a,x in maincomp.items()},quality_meets_criteria=all(checks.values()),incumbent='082',rollback='076',official_promotion=False,priority_fixed_before_outer_scores=True));print((OUT/'comparison.csv').read_text());print('DELIVERY PROTOCOL',chosen,flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
