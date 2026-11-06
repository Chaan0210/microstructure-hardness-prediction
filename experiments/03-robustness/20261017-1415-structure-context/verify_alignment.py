"""Audit target-aligned contexts before scoring, then selection and arithmetic."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import alignment as a
import encoder as e
import readout as r
import numpy as np
import json,sys,math,time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

def before():
    assert json.loads((a.OUT/'fit-complete.json').read_text())['complete']
    start=time.monotonic();d=r.physical();graphs=json.loads((e.OLD/'graphs.json').read_text());contexts=0;projections=[];teachers=[]
    lock=json.loads((a.OUT/'code-lock.json').read_text())
    assert lock['source_sha256']==e.sha(a.__file__) and lock['parent_decision_sha256']==e.sha(e.HERE/'decision.json')
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);folder=a.OUT/f'readouts/fold{k}-{stage}';t=np.load(_repo_io(folder/'teacher.npz'));agg=np.load(_repo_io(folder/'predictions.npz'))
            assert np.array_equal(t['train'],tr) and np.array_equal(t['query'],q) and set(tr).isdisjoint(q)
            assert np.array_equal(t['labels'],d['y'][tr]) and np.array_equal(t['residual'],t['labels']-t['loo_prediction'])
            for j in [0,len(tr)//2,len(tr)-1]:
                keep=np.arange(len(tr))!=j;poison=d['y'].copy();poison[np.setdiff1d(np.arange(500),tr[keep])]+=10000
                m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100)).fit(d['X'][tr[keep]],poison[tr[keep]])
                p=float(m.predict(d['X'][tr[j:j+1]])[0]);err=abs(p-float(t['loo_prediction'][j]));assert err<1e-10
                teachers.append(dict(fold=k,stage=stage,row=j,teacher_error=err,own_and_outside_labels_poisoned=True))
            target=(t['residual']-t['residual'].mean())/t['residual'].std()
            parent=np.load(_repo_io(e.HERE/f'readouts/fold{k}-{stage}/predictions.npz'));assert np.array_equal(agg['base'],parent['base'])
            for arm in r.ARMS:
                for seed in range(2):
                    sub=folder/f'{arm}-seed{seed}';ctx=np.load(_repo_io(sub/'context.npz'));src=e.HERE/(f'encoders/fold{k}-{stage}-seed{seed}-'+('spatial' if arm=='random' else arm));f=np.load(_repo_io(src/('random-features.npz' if arm=='random' else 'features.npz')))
                    assert np.array_equal(f['train'],tr) and np.array_equal(f['query'],q)
                    assert np.array_equal(ctx['X'][:,:100],d['X'][tr],equal_nan=True) and np.array_equal(ctx['Q'][:,:100],d['X'][q],equal_nan=True) and np.array_equal(ctx['y'],d['y'][tr])
                    sc=StandardScaler().fit(f['train_features']);assert np.array_equal(sc.mean_,ctx['scaler_mean']) and np.array_equal(sc.scale_,ctx['scaler_scale']) and np.array_equal(target,ctx['teacher_target'])
                    pls=PLSRegression(n_components=16,scale=False,max_iter=500,tol=1e-6).fit(sc.transform(f['train_features']),target)
                    assert np.array_equal(pls.x_rotations_,ctx['pls_rotations']) and np.array_equal(pls._x_mean,ctx['pls_mean'])
                    assert np.array_equal(pls.transform(sc.transform(f['train_features'])),ctx['X'][:,-16:]) and np.array_equal(pls.transform(sc.transform(f['query_features'])),ctx['Q'][:,-16:])
                    manual=(((f['query_features'].astype(float)-ctx['scaler_mean'])/ctx['scaler_scale']-ctx['pls_mean'])/ctx['pls_scale'])@ctx['pls_rotations'];err=float(np.max(abs(manual-ctx['Q'][:,-16:])));assert err<5e-4
                    projections.append(dict(fold=k,stage=stage,arm=arm,seed=seed,refit_exact=True,manual_float64_error=err));pp=[]
                    for j in range(3):
                        path=sub/f'tab{j}';c=json.loads((path/'contract.json').read_text());p=np.load(_repo_io(path/'prediction.npz'))
                        assert c['seed']==100*j+k and c['train']==tr.tolist() and c['query']==q.tolist()
                        assert c['train_X_sha256']==e.ah(ctx['X']) and c['train_y_sha256']==e.ah(ctx['y']) and c['query_X_sha256']==e.ah(ctx['Q'])
                        assert np.array_equal(p['train'],tr) and np.array_equal(p['query'],q) and np.isfinite(p['prediction']).all();pp.append(p['prediction']);contexts+=1
                    assert np.array_equal(np.mean(pp,0),agg[arm][seed])
    a.save(_repo_io('pre-score-verification.json'),dict(all_passed=True,contexts=contexts,projections=projections,teacher_spot_replays=teachers,outcomes_not_yet_scored=True,seconds=time.monotonic()-start,source_sha256=e.sha(__file__)))
    print('ALIGNMENT PRE-SCORE PASSED',contexts,len(projections),len(teachers),flush=True)

def coefficient(d,y,b,high):
    ix=np.flatnonzero(high);num=math.fsum(float(d[j])*float(y[j]-b[j]) for j in ix);den=math.fsum(float(d[j])**2 for j in ix)+len(ix)
    return max(0.,min(.5,num/den)) if len(ix) else 0.

def after():
    d=r.physical();graphs=json.loads((e.OLD/'graphs.json').read_text());result=json.loads((a.OUT/'results.json').read_text());o=np.load(_repo_io(a.OUT/'oof.npz'));checks=[];scalar={};select=a.old_selector()
    for k,g in enumerate(graphs):
        an=np.load(_repo_io(e.OLD/f'anchors/fold{k}.npz'));B=an['B'];high=an['highB'];ix=np.flatnonzero(high);labels=d['y'].copy();labels[an['V']]+=10000
        old=np.load(_repo_path(e.ROOT, f'experiments/rmse-research-20261015/round4-full073/aux-mean2/fold{k}-selection.npz'));loo=a.bases(k,an,labels[B],d['noise'][B],select)
        for j in ix:
            eligible=old['eligible'].copy();eligible[j]=False;poison=labels[B].copy();poison[j]+=10000
            w,_=select(old['residual'][eligible],(poison-old['base073'])[eligible],d['noise'][B][eligible]>=12)
            assert np.array_equal(an['p76B']+high*w[1]*old['residual'],loo[int(j)])
        parent=np.load(_repo_io(e.HERE/f'readouts/fold{k}-selector/predictions.npz'));new=np.load(_repo_io(a.OUT/f'readouts/fold{k}-selector/predictions.npz'));pv=np.load(_repo_io(e.HERE/f'readouts/fold{k}-refit/predictions.npz'));nv=np.load(_repo_io(a.OUT/f'readouts/fold{k}-refit/predictions.npz'))
        for arm in r.ARMS:
            dirs={'PCA':(parent[arm]-parent['base']).mean(0)[:len(B)],'PLS':(new[arm]-new['base']).mean(0)[:len(B)]};rec=result['selection'][k]['arms'][arm];gains={}
            for tag,delta in dirs.items():
                values=[]
                for j in ix:
                    keep=high.copy();keep[j]=False;poison=labels[B].copy();poison[j]+=10000
                    beta=coefficient(delta,labels[B],loo[int(j)],keep);assert abs(beta-coefficient(delta,poison,loo[int(j)],keep))<1e-15
                    y=labels[B[j]];b=loo[int(j)][j];values.append((y-b)**2-(y-b-beta*delta[j])**2)
                values=np.array(values);assert np.allclose(values,rec['LOO_gains'][tag]['per_source'],rtol=0,atol=1e-9);gains[tag]=(float(values.mean()),float(values.std(ddof=1)/math.sqrt(len(values))))
            best=max(gains,key=lambda t:gains[t][0]);choice=None
            if gains[best][0]>gains[best][1]:choice='PCA' if gains['PCA'][0]>=gains[best][0]-gains[best][1] else 'PLS'
            assert choice==rec['choice'];beta=coefficient(dirs[choice],labels[B],an['p82B'],high) if choice else 0.;assert abs(beta-rec['beta'])<1e-12
            pred=an['p82V'].copy()
            if choice:
                v=pv if choice=='PCA' else nv;pred=pred+an['highV']*rec['beta']*(v[arm]-v['base']).mean(0)
            assert np.array_equal(pred,o[arm][an['ix']]);checks.append(dict(fold=k,arm=arm,choice=choice,beta=beta,assembly_exact=True,outer_label_poison_unchanged=True,omitted_B_label_fits_neither_weight=True))
    for arm in r.ARMS:
        score=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(o[arm],o['y']))/len(o['y']));assert abs(score-result['results'][arm]['candidate_rmse'])<1e-12;scalar[arm]=score
    a.save(_repo_io('selection-verification.json'),dict(all_passed=True,checks=checks,independent_RMSE=scalar,source_sha256=e.sha(__file__)))
    print('ALIGNMENT SELECTION VERIFIED',len(checks),flush=True)

if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):globals()[sys.argv[1]]()
