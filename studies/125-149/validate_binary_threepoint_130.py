"""Independent raw-statistic physics solve, Ridge SVD, nesting and poison checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse,math
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import binary_threepoint_study_130 as s
import validate_nonlinear_residual_115 as phys
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
e=s.e

def manual_ridge(m,X,y,V):
    mu=X.mean(0);std=np.sqrt(np.mean((X-mu)**2,axis=0));sc=np.ones(X.shape[1]);weights=np.ones(X.shape[1]);b=np.array(m['blocks'])
    for name in set(b.tolist()):
        ix=b==name;floor=max(math.sqrt(float(np.dot(std[ix],std[ix]))/sum(ix)),1e-12);sc[ix]=np.maximum(std[ix],floor);weights[ix]=1/math.sqrt(sum(ix));assert abs(floor-m['scaler']['floors'][name])<1e-14
    np.testing.assert_allclose(m['scaler']['mean'],mu,atol=1e-15,rtol=0);np.testing.assert_allclose(m['scaler']['scale'],sc,atol=1e-15,rtol=0);Z=(X-mu)/sc*weights;W=(V-mu)/sc*weights;rs=math.sqrt(float(np.mean(y*y)));rs=rs if rs>1e-12 else 1.;assert rs==m['residual_scale'] and m['n_train']==len(y) and m['alpha']==m['lambda']*len(y)
    U,values,Vt=np.linalg.svd(Z,full_matrices=False);coef=Vt.T@((values/(values*values+m['alpha']))*(U.T@(y/rs)));np.testing.assert_allclose(coef,m['coef'],atol=1e-10,rtol=1e-10);return W@coef*rs

def folds(d,I,seed):
    return [(I[t],I[v]) for t,v in GroupKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(I)),groups=d['g'][I])]

def validate(split):
    s.check();start=time.perf_counter();d=s.load(split);path=s.OUT/split;report=e.read(path/'report.json');assert e.sha(path/'predictions.npz')==report['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz')) as z:pred={k:z[k].copy() for k in z.files}
    assert pred['ids'].tolist()==d['ids'];np.testing.assert_array_equal(pred['y'],d['y']);np.testing.assert_array_equal(pred['fold'],d['fid']);np.testing.assert_allclose(pred['reference014'],d['reference014'],atol=1e-10,rtol=0)
    maxphysics=maxridge=0.;physicscount=ridgecount=0;selected=[]
    for rec in report['records']:
        k=rec['fold'];j=rec['block'];folder=path/f'fold-{k}-block-{j}';assert e.read(folder/'report.json')==rec;assert e.sha(folder/'systems.joblib')==rec['systems_sha256'];z=joblib.load(_repo_io(folder/'systems.joblib'));A,B,V=z['A'],z['B'],z['V'];assert not(set(A)|set(B))&set(V) and not set(A)&set(B);assert set(A)|set(B)==set(np.flatnonzero(d['fid']!=k));assert not set(d['g'][B])&set(d['g'][A]);res=z['residual'];np.testing.assert_array_equal(res[B],d['y'][B]-z['anchor']);assert np.isnan(res[np.setdiff1d(np.arange(400),B)]).all();nodes={v['id']:v for v in z['nodes']};stages={v['id']:v for v in z['stages']}
        # Explicitly reconstruct train-local anchor fromexistingdeepA-onlyOOF.
        AA,BB,P,short,_=d['cells'][k,j];np.testing.assert_array_equal(A,AA);np.testing.assert_array_equal(B,BB);raw=s.a.mixture(d['deepP'][k,j],s.q.protected.ARM);coef=np.linalg.lstsq(np.column_stack([np.ones(len(A)),raw]),d['y'][A],rcond=None)[0];anchor=.9*(coef[0]+coef[1]*s.a.mixture(P,s.q.protected.ARM))+.1*short;np.testing.assert_allclose(anchor,z['anchor'],atol=1e-10,rtol=0)
        for node in nodes.values():
            I=node['train'];assert set(I)<=set(B);lookup={int(v):i for i,v in enumerate(I)};ip=np.full((5,len(I)),np.nan);ff=folds(d,I,node['seed']);assert len(node['fits'])==15
            for t,(T,H) in enumerate(ff):
                assert not set(d['g'][T])&set(d['g'][H]);hv=[lookup[int(v)] for v in H]
                for ci,kind in enumerate(s.n.KINDS):
                    fit=node['fits'][5*t+ci];np.testing.assert_array_equal(T,fit['train']);np.testing.assert_array_equal(H,fit['validation']);assert fit['kind']==kind==fit['model']['kind'];pp=phys.manual(fit['model'],d['X'][T],res[T],d['X'][H],d['blocks']);re=s.n.predict(fit['model'],d['X'][H]);np.testing.assert_array_equal(re,fit['predictions']);error=float(abs(pp-re).max());assert error<2e-9;maxphysics=max(maxphysics,error);ip[ci,hv]=pp;physicscount+=1
            scores=np.mean((ip-res[I])**2,axis=1);np.testing.assert_allclose(scores,node['MSE'],atol=2e-9,rtol=0);chosen=s.n.KINDS[int(scores.argmin())];assert chosen==node['chosen']==node['model']['kind'];pp=phys.manual(node['model'],d['X'][I],res[I],d['X'][V],d['blocks']);re=s.n.predict(node['model'],d['X'][V]);error=float(abs(pp-re).max());assert error<2e-9;maxphysics=max(maxphysics,error);physicscount+=1
        for st in stages.values():
            I=st['train'];np.testing.assert_array_equal(I,nodes[st['physics_node']]['train']);lookup={int(v):i for i,v in enumerate(I)};cross=np.full(len(I),np.nan)
            for t,((T,H),tr) in enumerate(zip(folds(d,I,st['seed']),st['crossfit_trace'])):
                node=nodes[tr['node']];np.testing.assert_array_equal(node['train'],T);np.testing.assert_array_equal(tr['validation'],H);assert node['seed']==st['seed']+1000+t;cross[[lookup[int(v)] for v in H]]=s.n.predict(node['model'],d['X'][H]);assert not set(T)&set(H)
            np.testing.assert_array_equal(cross,st['crossfit_physics']);np.testing.assert_array_equal(res[I]-cross,st['new_residual'])
            for arm in s.ARMS:
                for lam in s.LAMS:
                    m=st['models'][arm][str(lam)];assert m['lambda']==lam and m['blocks']==s.blocks(arm);pp=manual_ridge(m,s.panel(d,I,arm),st['new_residual'],s.panel(d,V,arm));re=s.ridge_predict(m,s.panel(d,V,arm));error=float(abs(pp-re).max());assert error<2e-9;maxridge=max(maxridge,error);ridgecount+=1
            for lam in s.LAMS:
                a=st['models']['pair'][str(lam)]['scaler'];b=st['models']['pair_triple'][str(lam)]['scaler'];np.testing.assert_array_equal(a['mean'],b['mean'][:114]);np.testing.assert_allclose(a['scale'],b['scale'][:114],atol=1e-15,rtol=0)
        lookup={int(v):i for i,v in enumerate(B)};ip={arm:np.full((5,len(B)),np.nan) for arm in s.ARMS};seed=s.r.CVSEED+100*k+j
        for t,(I,H) in enumerate(folds(d,B,seed)):
            st=stages[z['cv_stages'][t]];np.testing.assert_array_equal(st['train'],I);assert st['seed']==seed+1000+t;node=nodes[st['physics_node']];base=s.n.predict(node['model'],d['X'][H]);hv=[lookup[int(v)] for v in H]
            # Everymodel/target source underneaththisstage excludesH, includingselectedphysics hyperparameters.
            for tr in st['crossfit_trace']:assert not set(nodes[tr['node']]['train'])&set(H)
            for arm in s.ARMS:
                ip[arm][0,hv]=base
                for ci,(lam,gamma) in enumerate(s.CHOICES[1:],1):ip[arm][ci,hv]=base+gamma*s.ridge_predict(st['models'][arm][str(lam)],s.panel(d,H,arm))
        full=stages[z['full_stage']];np.testing.assert_array_equal(full['train'],B);np.testing.assert_array_equal(full['crossfit_physics'],z['inner_baseline']);old=s.n.predict(nodes[z['full_node']]['model'],d['X'][V]);np.testing.assert_array_equal(old,z['old_correction']);np.testing.assert_array_equal(old,pred['old_corrections'][j,V])
        for arm in s.ARMS:
            np.testing.assert_array_equal(ip[arm],z['inner_predictions'][arm]);scores=np.mean((ip[arm]-res[B])**2,axis=1);np.testing.assert_array_equal(scores,z['scores'][arm]);choice=int(scores.argmin());assert choice==z['chosen'][arm] and s.LABELS[choice]==rec['selected'][arm];lam,gamma=s.CHOICES[choice];delta=np.zeros(len(V)) if lam is None else gamma*s.ridge_predict(full['models'][arm][str(lam)],s.panel(d,V,arm));np.testing.assert_array_equal(delta,z['delta'][arm]);np.testing.assert_array_equal(delta,pred[arm+'_delta'][j,V]);selected.append((arm,s.LABELS[choice]))
    np.testing.assert_array_equal(pred['base']+pred['old_corrections'].mean(0),pred['reference014'])
    for arm in s.ARMS:np.testing.assert_array_equal(pred['reference014']+pred[arm+'_delta'].mean(0),pred[arm])
    for arm in ['reference014',*s.ARMS]:assert e.rmse(d['y'],pred[arm])==report['RMSE'][arm]
    stress=None
    if split=='original':
        k=j=0;folder=path/'fold-0-block-0';old=joblib.load(_repo_io(folder/'systems.joblib'));pert=dict(d);pert['y']=d['y'].copy();pert['y'][old['V']]+=10000;pp,de,rr=s.block(pert,split,0,0,s.OUT/'outer-label-replay');np.testing.assert_array_equal(pp,old['old_correction']);assert rr['selected']==report['records'][0]['selected'] and rr['inner_MSE']==report['records'][0]['inner_MSE']
        for arm in s.ARMS:np.testing.assert_array_equal(de[arm],old['delta'][arm])
        I,H=folds(d,old['B'],s.r.CVSEED)[0];res=old['residual'].copy();res[H]+=10000;eng=s.Nested(d,res);st=eng.stage(I,s.r.CVSEED+1000);orig=old['stages'][old['cv_stages'][0]];np.testing.assert_array_equal(st['new_residual'],orig['new_residual'])
        for arm in s.ARMS:
            for lam in s.LAMS:np.testing.assert_array_equal(st['models'][arm][str(lam)]['coef'],orig['models'][arm][str(lam)]['coef'])
        stress={'outer80_target_poison_exact_choices_and_predictions':True,'internal_H_target_poison_exact_crossfit_residual_and4Ridges':True,'internal_H_rows':len(H),'upstream112_training_certificate_reused':e.sha(s.e.OUT/'validation-replay.json'),'new_neural_training':False}
    result={'passed':True,'split':split,'physics_models_independently_checked':physicscount,'new_Ridge_models_independently_checked':ridgecount,'physics_rawstat_eigensolve_max_error':maxphysics,'new_Ridge_SVD_max_error':maxridge,'all_nested_choices_groups_and_training_only_floors_verified':True,'014_reconstruction_max_error':report['014_reconstruction_max_error'],'leakage':stress,'report_sha256':e.sha(path/'report.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};s.check();e.write(s.OUT/f'validation-{split}.json',result);e.emit(result)
if __name__=='__main__':
    s.q.protected.torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=s.e.SPLITS);validate(p.parse_args().split)
