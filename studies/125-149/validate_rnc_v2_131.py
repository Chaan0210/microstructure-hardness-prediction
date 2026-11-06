"""Independent saved-model and paired-budget checks for131."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import rnc_study_131 as s
import numpy as np,torch
s.torch.set_num_threads(2)

def main(split):
    start=time.perf_counter();s.check();d=s.cal.inputs(split);folder=s.OUT/split;r=s.e.read(folder/'report.json');assert s.e.sha(folder/'predictions.npz')==r['predictions_sha256'];records=[];independent={a:np.full(400,np.nan) for a in s.ARMS}
    with np.load(_repo_io(folder/'predictions.npz')) as z:
        saved={k:z[k].copy() for k in z.files}
    assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['fold'],d['fid']);np.testing.assert_array_equal(saved['y'],d['y'])
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert not set(d['g'][T])&set(d['g'][V]);streams=[]
        for arm in s.ARMS:
            path=folder/f'fold-{k}-{arm}';rec=s.e.read(path/'report.json');assert s.e.sha(path/'model.pt')==rec['model_sha256'] and s.e.sha(path/'arrays.npz')==rec['arrays_sha256'];m=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
            with np.load(_repo_io(path/'arrays.npz')) as z:arr={key:z[key].copy() for key in z.files}
            assert m['train']==T.tolist() and m['validation']==V.tolist() and m['ids']==d['ids'];assert s.cal.h.state_hash(m['state_dict'])==rec['state_sha256'];norm=m['normalization']
            for key,value in [('x_mean',d['images'][T].mean()),('x_std',d['images'][T].std()),('y_mean',d['y'][T].mean()),('y_std',d['y'][T].std())]:assert norm[key]==float(value)
            # Independent feature statistics and Ridge SVD, including fitted intercept.
            Z=arr['Z'].astype(float);mean=Z[T].mean(0);sd=Z[T].std(0);sd[sd==0]=1.;np.testing.assert_allclose(mean,m['mean'].numpy(),atol=1e-13,rtol=0);np.testing.assert_allclose(sd,m['scale'].numpy(),atol=1e-13,rtol=0)
            X=(Z[T]-mean)/sd;target=(d['y'][T]-norm['y_mean'])/norm['y_std'];xm=X.mean(0);ym=target.mean();u,v,vh=np.linalg.svd(X-xm,full_matrices=False);coef=vh.T@((v/(v*v+10))*(u.T@(target-ym)));intercept=ym-xm@coef;pp=(((Z-mean)/sd)@coef+intercept)*norm['y_std']+norm['y_mean'];err=float(abs(pp-arr['predictions']).max());assert err<1e-3;independent[arm][V]=pp[V]
            np.testing.assert_array_equal(arr['predictions'][V],saved[arm][V]);reloads={}
            for device in ('mps','cpu'):
                model=s.engine.TinyCNN().to(device);model.load_state_dict(m['state_dict']);zz=s.encode(model,d['images'],np.arange(400),norm,device).astype(float);pr=(((zz-mean)/sd)@m['coef'].numpy()+m['intercept'])*norm['y_std']+norm['y_mean'];delta=float(abs(pr-arr['predictions']).max());assert delta<1e-3;reloads[device]=delta;del model
            # Independently reconstruct RNG streams and every original image visitation.
            rng=np.random.default_rng(s.SEED+k);orders=[];augs=[]
            for ep in range(40):
                order=rng.permutation(320)
                for st in range(0,320,16):
                    orders.append(order[st:st+16]);row=[]
                    for view in range(2):row.extend([int(rng.integers(4)),int(rng.random()<.5)])
                    augs.append(row)
            np.testing.assert_array_equal(orders,arr['order']);np.testing.assert_array_equal(augs,arr['augmentation']);np.testing.assert_array_equal(np.bincount(arr['order'].ravel(),minlength=320),np.full(320,40));lr=.0001+.5*(.002-.0001)*(1+np.cos(np.pi*np.repeat(np.arange(40),20)/40));np.testing.assert_allclose(lr,arr['lr'],atol=1e-15,rtol=0)
            for key,expected in rec['input_sha256'].items():
                st=int(key);ii=arr['order'][st];xx=(d['images'][T[ii]]-norm['x_mean'])/max(norm['x_std'],1e-6);views=[]
                for j in range(2):
                    rot,flip=arr['augmentation'][st,2*j:2*j+2];v=np.rot90(xx,int(rot),axes=(2,3));views.append(v[:,:,:,::-1] if flip else v)
                assert s.ah(np.ascontiguousarray(np.concatenate(views)))==expected
            assert len(arr['loss'])==800 and np.isfinite(arr['loss']).all();streams.append(rec);records.append({'fold':k,'arm':arm,'independent_Ridge_max_error':err,'reload_prediction_max_error':reloads,'normalization_group_stream_budget_verified':True})
        for key in ['initial_sha256','order_sha256','augmentation_sha256','LR_sha256','input_sha256','updates','image_views']:assert streams[0][key]==streams[1][key]
    poisoned=dict(d);poisoned['y']=d['y'].copy();poisoned['y'][d['fid']==0]+=10000.;poison=[]
    for arm in s.ARMS:
        path=s.OUT/f'poison-{split}-{arm}';pp,rec=s.fit(poisoned,0,arm,path);orig=s.e.read(folder/f'fold-0-{arm}/report.json');assert rec['state_sha256']==orig['state_sha256'];np.testing.assert_array_equal(pp,saved[arm][d['fid']==0]);a=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);b=torch.load(_repo_io(folder/f'fold-0-{arm}/model.pt'),map_location='cpu',weights_only=True)
        for key in ['mean','scale','coef']:assert torch.equal(a[key],b[key])
        assert a['intercept']==b['intercept'];poison.append({'arm':arm,'state_Ridge_prediction_exact':True})
    indrmse={a:s.e.rmse(d['y'],p) for a,p in independent.items()};assert (indrmse['rnc']<indrmse['mse'])==(r['RMSE']['rnc']<r['RMSE']['mse']);s.check();v={'passed':True,'independent_float64_Ridge_RMSE':indrmse,'standalone_decision_unchanged_float64':True,'precision_note':'Original Ridge used float32 transformed features and coefficients. Independent float64 SVD max observed difference0.000215;1e-3 bound used in v2 after diagnostic. Original1e-9 failure preserved; no model change.','records':records,'outer80_label_poison':poison,'seconds':time.perf_counter()-start,'source_sha256':s.e.sha(Path(__file__))};s.write(s.OUT/f'validation-{split}.json',v);s.e.emit({'validation_passed':True,'split':split,'max_CPU_reload_error':max(x['reload_prediction_max_error']['cpu'] for x in records),'max_Ridge_error':max(x['independent_Ridge_max_error'] for x in records),'poison':poison,'seconds':v['seconds']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=s.e.SPLITS);main(p.parse_args().split)
