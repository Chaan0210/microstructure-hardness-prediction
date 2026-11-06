"""Independent saved-model predictions, accounting and subset scope for127."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import base_learning_curve_v2_127 as q
import numpy as np,torch,joblib
from scipy.spatial.distance import cdist,pdist
from sklearn.preprocessing import MinMaxScaler,StandardScaler
e=q.e

def manual_kernel(m,X,y,Q):
    bound=MinMaxScaler(clip=True).fit(X);sc=StandardScaler().fit(bound.transform(X));blocks=np.array(m.blocks);w=np.array([1/np.sqrt(np.sum(blocks==b)) for b in blocks]);Z=sc.transform(bound.transform(X))*w;ZZ=sc.transform(bound.transform(Q))*w
    dist=pdist(Z,'sqeuclidean');gamma=1/max(float(np.median(dist[dist>1e-12]))*m.bandwidth**2,1e-12);np.testing.assert_array_equal(Z,m.X_);assert gamma==m.gamma_
    K=np.exp(-gamma*cdist(Z,Z,'sqeuclidean'));coef=np.linalg.solve(K+m.alpha*np.eye(len(X)),(y-y.mean())/y.std());return np.exp(-gamma*cdist(ZZ,Z,'sqeuclidean'))@coef*y.std()+y.mean()

def manual_head(c,X):
    w=c['model'];xs=torch.tensor((X-c['scaler_mean'].numpy())/c['scaler_scale'].numpy(),dtype=torch.float32);out=[]
    for start in range(0,len(xs),31):
        hidden=torch.nn.functional.gelu(torch.nn.functional.linear(xs[start:start+31],w['shared.0.weight'],w['shared.0.bias']));out.extend(torch.nn.functional.linear(hidden,w['reg.weight'],w['reg.bias']).squeeze(-1).numpy().astype(float))
    return np.array(out)*c['target_std']+c['target_mean']

def main():
    q.check();torch.set_num_threads(2);t=time.perf_counter();maxk=0.;maxh=0.;maxc=0.;bundles=0;heads=0;cnn=0;counts=[];control078=0.
    old=e.read(_repo_path(ROOT, 'artifacts/reliability-size-078/report.json'));assert e.sha(_repo_path(ROOT, 'artifacts/reliability-size-078/predictions.npz'))==old['predictions_sha256']
    with np.load(_repo_path(ROOT, 'artifacts/reliability-size-078/predictions.npz')) as z:oldD={s:z[s+'_DINO_small'].copy() for s in e.SPLITS}
    for split in e.SPLITS:
        d=q.cal.inputs(split);root=q.OUT/split;r=e.read(root/'report.json');assert e.sha(root/'predictions.npz')==r['predictions_sha256']
        with np.load(_repo_io(root/'predictions.npz')) as z:
            assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);np.testing.assert_array_equal(z['fold'],d['fid']);P=z['components'].copy()
        for rec in r['records']:
            k=rec['outer_fold'];level=q.LEVELS.index(rec['level']);j=rec['path_index'];T=np.array(rec['train_indices']);V=np.array(rec['outer_indices']);folder=root/f'fold-{k}-level-{level}-path-{j}';assert e.read(folder/'report.json')==rec;assert e.sha(folder/'predictions.npz')==rec['predictions_sha256']
            with np.load(_repo_io(folder/'predictions.npz')) as z:local=z['components'].copy();seedpred=z['seeds'].copy();np.testing.assert_array_equal(z['train_indices'],T);np.testing.assert_array_equal(z['outer_indices'],V)
            assert len(T)==rec['n_train'];assert set(V)==set(np.flatnonzero(d['fid']==k)) and not set(d['g'][T])&set(d['g'][V])
            np.testing.assert_array_equal(local[:,V],P[level,j][:,V]);np.testing.assert_array_equal(local[2:],seedpred.mean(1));np.testing.assert_allclose(np.sum((local[:,T]-d['y'][T])**2,1),rec['train_SSE'],atol=1e-10,rtol=0)
            for name,hh in rec['model_sha256'].items():assert e.sha(_repo_path(ROOT, name))==hh
            sp=q.spec(split,k,level,j);km=joblib.load(_repo_io(sp['paths']['kernel']));m=km['model'];assert km['train_ids']==[d['ids'][i] for i in T];manual=manual_kernel(m,d['K'][T],d['y'][T],d['K']);err=float(abs(manual-local[0]).max());assert err<1e-8;maxk=max(maxk,err)
            for ai,arm in enumerate(['S','B']):
                for si,seed in enumerate(q.SEEDS):
                    c=torch.load(_repo_io(sp['paths'][f'{arm}_{seed}']),weights_only=True,map_location='cpu');assert c['train_ids']==[d['ids'][i] for i in T]
                    with torch.inference_mode():manual=manual_head(c,d[arm])
                    err=float(abs(manual-seedpred[ai,si]).max());assert err<1e-3;maxh=max(maxh,err);heads+=1
            if k==0 and j==0:
                c=torch.load(_repo_io(sp['paths']['cnn']),weights_only=True,map_location='cpu');fresh=q.cnn_predict(c,d['images'],batch=17);err=float(abs(fresh-local[1]).max());assert err<1e-3;maxc=max(maxc,err);cnn+=1
            if level<2:
                A=np.array(q.spec(split,k,1,j)['report']['train_indices']);assert set(T)<=set(A)<=set(np.flatnonzero(d['fid']!=k))
            else:
                np.testing.assert_array_equal(P[level,0][:,V],P[level,1][:,V]);np.testing.assert_array_equal(P[level,0][:,V],P[level,2][:,V])
            bundles+=1
        err=float(abs(.5*(P[1,:,2]+P[1,:,3])-oldD[split]).max());assert err<1e-4;control078=max(control078,err)
        # Equality of covariance risk and direct mixed squared error, at every size/path/fold.
        errors=P-d['y'];mix=np.einsum('lpcn,c->lpn',P,q.W);left=np.mean((mix-d['y'])**2,axis=(1,2));right=np.einsum('c,lpcn->l',q.W,errors**2)/(3*400)-np.einsum('c,lpcn->l',q.W,(P-mix[:,:,None,:])**2)/(3*400);np.testing.assert_allclose(left,right,atol=1e-10,rtol=0)
        for level in range(3):
            cov=errors[level].transpose(1,0,2).reshape(4,-1);G=cov@cov.T/cov.shape[1];assert abs(q.W@G@q.W-left[level])<1e-10
        counts.append({'split':split,'bundles':len(r['records']),'all_finite':bool(np.isfinite(P).all()),'078_middle_DINOmean_max_error':err});e.emit(counts[-1])
    assert bundles==105 and heads==630 and cnn==9
    result={'passed':True,'bundles_verified':bundles,'kernel_independent_solves':bundles,'kernel_max_error':maxk,'all630_heads_independent_functional_predictions_max_error':maxh,'CNN_fresh_reload_different_batch_bundles':cnn,'CNN_max_error':maxc,'078_middle_DINOmean_reproduction_max_error':control078,'subset_and_outer_group_exclusion_verified':True,'foldtrain_scalers_targets_verified_in_main_replay':True,'MSE_diversity_covariance_identities_verified':True,'partitions':counts,'upstream_training_leakage_certificates_reused':{f:e.sha(_repo_path(ROOT, f)) for f in ['reports/nested-calibration-074-validation.json','artifacts/nested-fusion-112/validation-replay.json','reports/dinob-complement-064-validation.json']},'report_sha256':{s:e.sha(q.OUT/s/'report.json') for s in e.SPLITS},'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-t,'new_training':False,'holdout_test_access':False}
    q.check();e.write(q.OUT/'validation.json',result);e.emit(result)
if __name__=='__main__':main()
