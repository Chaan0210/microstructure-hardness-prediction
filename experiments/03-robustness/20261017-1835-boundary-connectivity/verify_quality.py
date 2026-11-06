"""Independent arithmetic and source-exclusion checks for the support policy."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import quality_evaluate as q
import numpy as np,json,math,sys,time
def beta(d,y,b,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return 0.
    numerator=math.fsum(float(d[j])*float(y[j]-b[j]) for j in ix);denominator=math.fsum(float(d[j])**2 for j in ix)+len(ix);return max(0.,min(.5,numerator/denominator))
def preflight():
    n=44;high=np.ones(n,bool);support=np.arange(n)<22;an={'highB':high,'p82B':np.zeros(n)};loo={j:np.zeros(n) for j in range(n)};delta=np.ones(n);examples=[]
    for label,y,want in [('zero',np.zeros(n),None),('broad',np.ones(n),'all'),('supported',np.where(support,1.,-1.),'supported')]:
        route,b,record=q.choose(delta,y,an,support,loo);assert route==want;examples.append(dict(fixture=label,route=route,beta=b))
    d=c.inputs(labels=True);select=q.old_selector();checks=[]
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));B=an['B'];bases=q.loo_bases(k,an,d['y'][B],d['noise'][B],select);old=np.load(_repo_path(c.ROOT, f'experiments/rmse-research-20261015/round4-full073/aux-mean2/fold{k}-selection.npz'));j=int(np.flatnonzero(an['highB'])[0]);mask=old['eligible'].copy();mask[j]=False;poison=d['y'][B].copy();poison[j]+=10000;w,_=select(old['residual'][mask],(poison-old['base073'])[mask],d['noise'][B][mask]>=12);assert np.array_equal(an['p76B']+an['highB']*w[1]*old['residual'],bases[j]);checks.append(dict(fold=k,omitted_B_source=int(B[j]),old_weight_omitted_label_poison_exact=True))
    assert not (.5>.5) and (.500001>.5);c.save(_repo_io('quality-policy/preflight.json'),dict(all_passed=True,fixtures=examples,incumbent_calibration_checks=checks,no_new_candidate_predictions_or_scores_read=True,fixtures_used_for_training=False,source_sha256=c.sha(__file__)));print('QUALITY PREFLIGHT PASSED',flush=True)
def after():
    start=time.monotonic();d=c.inputs(labels=True);res=json.loads((q.OUT/'results.json').read_text());out=np.load(_repo_io(q.OUT/'oof.npz'));target=json.loads((c.HERE/'target-audit.json').read_text())['sources'];support=np.zeros(500,bool)
    for row in target:support[int(row['index'])]=row['boundary_evidence_AUC']>.5
    assert np.array_equal(support[d['dev']],out['support']);select=q.old_selector();checks=[];arith={}
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));B=an['B'];V=an['V'];a=np.load(_repo_io(c.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(c.HERE/f'readouts/fold{k}-refit/predictions.npz'));labels=d['y'].copy();labels[V]+=10000;loo=q.loo_bases(k,an,labels[B],d['noise'][B],select);high=an['highB'];routes={'all':high,'supported':high&support[B]}
        for arm in q.ARMS:
            all_delta=(a[arm]-a['base'])[:,:len(B)];all_v=v[arm]-v['base'];base_rec=res['selection'][k]['arms'][arm]
            for seed in [None,0,1]:
                delta=all_delta.mean(0) if seed is None else all_delta[seed];vv=all_v.mean(0) if seed is None else all_v[seed];record=base_rec if seed is None else base_rec['seeds'][seed];means={};errors=[]
                for tag,mask in routes.items():
                    values=[]
                    for j in np.flatnonzero(high):
                        keep=mask.copy();keep[j]=False;y=labels[B].copy();w=beta(delta,y,loo[int(j)],keep);y[j]+=10000;assert w==beta(delta,y,loo[int(j)],keep);change=w*delta[j] if mask[j] else 0.;r=labels[B[j]]-loo[int(j)][j];values.append(r*r-(r-change)**2)
                    values=np.array(values);err=float(np.max(abs(values-np.array(record['LOO'][tag]['source_gains']))));assert err<1e-9;errors.append(err);means[tag]=(float(values.mean()),float(values.std(ddof=1)/math.sqrt(len(values))))
                best=max(means,key=lambda x:means[x][0]);choice=None
                if means[best][0]>means[best][1]:choice='all' if means['all'][0]>=means[best][0]-means[best][1] else 'supported'
                assert choice==record['route'];w=beta(delta,labels[B],an['p82B'],routes[choice]) if choice else 0.;assert abs(w-record['beta'])<1e-12;pred=an['p82V'].copy()
                if choice:pred+=an['highV']*(support[V] if choice=='supported' else True)*record['beta']*vv
                key=arm if seed is None else f'{arm}_seed{seed}';assert np.array_equal(pred,out[key][an['ix']]);checks.append(dict(fold=k,arm=arm,seed=seed,route=choice,beta=w,maximum_independent_LOO_gain_error=max(errors),outer_labels_fit_no_decision=True,omitted_B_label_fits_neither_old_nor_new_coefficient=True,assembly_exact=True))
    for arm in q.ARMS:
        score=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(out[arm],out['y']))/len(out['y']));assert abs(score-res['results'][arm]['candidate_rmse'])<1e-12;arith[arm]=score
    r=res['results']['topology'];comp=res['topology_vs_controls'];checks_expected=dict(gain05=r['gain']>=.05,high_gain=r['high_noise']['gain']>0,three_folds=r['improved_folds']>=3,CI975_vs082=r['conditional_CI975'][0]>0,CI975_vs_controls=all(x['conditional_CI975'][0]>0 for x in comp.values()),top10=r['removed_largest_gains']['10']['gain']>0,both_seeds=all(x['gain']>=0 for x in r['seeds']));decision=json.loads((q.OUT/'decision.json').read_text());assert checks_expected==decision['checks'] and all(checks_expected.values())==decision['advance'];delivery=json.loads((c.HERE/'delivery-decision.json').read_text());expected='all_high' if delivery['main_meets_stricter_family_screen'] else ('quality_policy' if all(checks_expected.values()) else None);assert delivery['selected_protocol']==expected and delivery['new_candidate_recommended']==(expected is not None)
    c.save(_repo_io('quality-policy/verification.json'),dict(all_passed=True,checks=checks,independent_RMSE=arith,decision_and_priority_exact=True,seconds=time.monotonic()-start,source_sha256=c.sha(__file__)));print('QUALITY VERIFICATION PASSED',len(checks),flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):globals()[sys.argv[1]]()
