"""Independent CNN readout, nested affine/residual, and label exclusion audit132."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,argparse,gc
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mse_readout_132 as m
import validate_nonlinear_residual_115 as vn
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
s=m.s;e=m.e;a=m.a

def neural(d,path):
    r=e.read(path/'report.json');obj=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');assert e.sha(path/'model.pt')==r['model_sha256'] and e.sha(path/'arrays.npz')==r['arrays_sha256'];T=np.array(obj['train']);V=np.array(obj['validation']);assert r['train']==T.tolist() and r['validation']==V.tolist() and not set(d['g'][T])&set(d['g'][V]);assert obj['ids']==d['ids'];norm=obj['normalization'];assert norm=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())}
    with np.load(_repo_io(path/'arrays.npz')) as z:z={k:z[k].copy() for k in z.files}
    Z=z['Z'].astype(float);mean=Z[T].mean(0);sd=Z[T].std(0);sd[sd==0]=1.;np.testing.assert_allclose(mean,obj['mean'].numpy(),atol=1e-13,rtol=0);np.testing.assert_allclose(sd,obj['scale'].numpy(),atol=1e-13,rtol=0);X=(Z-mean)/sd;xm=X[T].mean(0);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];ym=y.mean();u,v,vh=np.linalg.svd(X[T]-xm,full_matrices=False);coef=vh.T@((v/(v*v+10))*(u.T@(y-ym)));p=((X-xm)@coef+ym)*norm['y_std']+norm['y_mean'];err=float(abs(p-z['predictions']).max());assert err<1e-3
    reload={}
    for device in ['mps','cpu']:
        model=s.engine.TinyCNN().to(device);model.load_state_dict(obj['state_dict']);features=s.encode(model,d['images'],np.arange(400),norm,device).astype(float);pp=(((features-mean)/sd)@obj['coef'].numpy()+obj['intercept'])*norm['y_std']+norm['y_mean'];rr=float(abs(pp-z['predictions']).max());assert rr<1e-3;reload[device]=rr;del model
    rng=np.random.default_rng(obj['seed']);order=[];offset=[0];aug=[]
    for ep in range(40):
        ix=rng.permutation(len(T))
        for start in range(0,len(T),16):
            group=ix[start:start+16];order.extend(group.tolist());offset.append(len(order));row=[]
            for view in range(2):row.extend([int(rng.integers(4)),int(rng.random()<.5)])
            aug.append(row)
    for key,x in [('order',order),('offsets',offset),('augmentation',aug)]:np.testing.assert_array_equal(x,z[key])
    np.testing.assert_array_equal(np.bincount(z['order'],minlength=len(T)),np.full(len(T),40));assert r['image_views']==80*len(T) and r['updates']==40*mathceil(len(T)/16);lr=.0001+.5*(.002-.0001)*(1+np.cos(np.pi*np.repeat(np.arange(40),mathceil(len(T)/16))/40));np.testing.assert_allclose(lr,z['lr'],atol=1e-15,rtol=0);assert np.isfinite(z['loss']).all();return {'path':str(path.relative_to(ROOT)),'independentSVD_max_error':err,'reload_max_error':reload,'group_stats_budget_verified':True}

def mathceil(x):return int(np.ceil(x))
def ols(P,y,Q):
    mu=P.mean();D=np.column_stack([np.ones(len(P)),P-mu]);coef=np.linalg.lstsq(D,y,rcond=None)[0];return coef[0]+coef[1]*(Q-mu)

def pipeline(d,dd,split,k):
    T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);path=m.OUT/split/f'pipeline-fold-{k}';r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);base=.9*ols(a.mixture(dd['middleP'][k],m.ARM),d['y'][T],a.mixture(dd['P'][V],m.ARM))+.1*d['short'][V];np.testing.assert_allclose(base,z['base'],atol=1e-10,rtol=0);cor=[];errors=[];count=0
    for old,new in [(d['P'],dd['P']),(d['middleP'][k],dd['middleP'][k])]:np.testing.assert_array_equal(old[:,[0,1,3,4]],new[:,[0,1,3,4]])
    for j in range(3):
        A,B,P,short,_=dd['cells'][k,j];np.testing.assert_array_equal(d['deepP'][k,j][:,[0,1,3,4]],dd['deepP'][k,j][:,[0,1,3,4]]);np.testing.assert_array_equal(d['cells'][k,j][2][:,[0,1,3,4]],P[:,[0,1,3,4]]);anchor=.9*ols(a.mixture(dd['deepP'][k,j],m.ARM),d['y'][A],a.mixture(P,m.ARM))+.1*short;bp=path/f'block-{j}';br=e.read(bp/'report.json');assert e.sha(bp/'report.json')==r['block_reports_sha256'][f'block-{j}/report.json'];zz=a.readnp(bp/'predictions.npz',br['predictions_sha256']);np.testing.assert_allclose(anchor,zz['anchor_predictions'],atol=1e-10,rtol=0);res=d['y'][B]-zz['anchor_predictions'];np.testing.assert_array_equal(res,zz['residual']);ip=np.empty_like(zz['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=m.q.r.CVSEED+100*k+j).split(d['X'][B],res,d['g'][B]))
        for t,(it,iv) in enumerate(folds):
            assert br['splits'][t]=={'train_indices':B[it].tolist(),'validation_indices':B[iv].tolist()}
            for ci,kind in enumerate(m.q.n.KINDS):
                f=bp/f'inner-{t}-{kind}.joblib';assert e.sha(f)==br['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B[it]);pp=vn.manual(obj['model'],d['X'][B[it]],res[it],d['X'][B[iv]],d['blocks']);err=float(abs(pp-zz['inner_predictions'][ci,iv]).max());assert err<2e-9;ip[ci,iv]=pp;errors.append(err);count+=1
        scores=((ip-res)**2).mean(1);np.testing.assert_allclose(scores,list(br['inner_MSE'].values()),atol=2e-9,rtol=0);chosen=m.q.n.KINDS[int(scores.argmin())];assert chosen==br['chosen'];f=bp/'selected.joblib';assert e.sha(f)==br['models_sha256'][f.name];obj=joblib.load(_repo_io(f));np.testing.assert_array_equal(obj['train_indices'],B);pp=vn.manual(obj['model'],d['X'][B],res,d['X'][V],d['blocks']);err=float(abs(pp-zz['correction']).max());assert err<2e-9;cor.append(pp);errors.append(err);count+=1
    pe=float(abs(base+np.mean(cor,0)-z['primary']).max());assert pe<1e-9;return {'fold':k,'physics_models':count,'independent_physics_error':max(errors),'pipeline_error':pe,'all_other_components_unchanged':True}

def compare_models(left,right):
    x=torch.load(_repo_io(left/'model.pt'),weights_only=True,map_location='cpu');y=torch.load(_repo_io(right/'model.pt'),weights_only=True,map_location='cpu');assert s.cal.h.state_hash(x['state_dict'])==s.cal.h.state_hash(y['state_dict']);assert x['normalization']==y['normalization'] and x['intercept']==y['intercept']
    for key in ['coef','mean','scale']:assert torch.equal(x[key],y[key])
    with np.load(_repo_io(left/'arrays.npz')) as p,np.load(_repo_io(right/'arrays.npz')) as q:np.testing.assert_array_equal(p['predictions'],q['predictions'])

def main(split):
    m.check();start=time.perf_counter();d=m.data(split);r=e.read(m.OUT/split/'report.json');z=a.readnp(m.OUT/split/'predictions.npz',r['predictions_sha256']);assert z['ids'].tolist()==d['ids'];neuralchecks=[];pipes=[]
    if split=='original':
        with np.load(_repo_io(s.OUT/'original/predictions.npz')) as old:
            np.testing.assert_array_equal(old['fold'],d['fid']);np.testing.assert_array_equal(old['y'],d['y']);assert old['ids'].tolist()==d['ids']
        for k in range(5):
            obj=torch.load(_repo_io(s.OUT/'original'/f'fold-{k}-mse/model.pt'),weights_only=True,map_location='cpu');assert obj['train']==np.flatnonzero(d['fid']!=k).tolist() and obj['validation']==np.flatnonzero(d['fid']==k).tolist()
    for path in r['new_CNN_paths']:neuralchecks.append(neural(d,_repo_path(ROOT, path)))
    if split=='original':
        for k in range(5):neuralchecks.append(neural(d,m.OUT/'diagnostics'/f'new-legacyseed-{k}'))
        neuralchecks.append(neural(d,m.OUT/'diagnostics/exact131-replay'))
    diagnosticchecks=[]
    if split=='original':
        for k in range(5):
            T=np.flatnonzero(d['fid']!=k);ym=d['y'][T].mean();ys=d['y'][T].std()
            for key in ['legacy','new']:
                path=m.OUT/'diagnostics'/f'{key}-{k}.npz'
                with np.load(_repo_io(path)) as zz:zz={n:zz[n].copy() for n in zz.files}
                Z=zz['Z'].astype(float);mu=Z[T].mean(0);sd=Z[T].std(0);sd[sd==0]=1.;np.testing.assert_allclose(mu,zz['mean'],atol=1e-13,rtol=0);np.testing.assert_allclose(sd,zz['scale'],atol=1e-13,rtol=0);X=(Z-mu)/sd;xm=X[T].mean(0);y=(d['y'][T]-ym)/ys;u,v,vh=np.linalg.svd(X[T]-xm,full_matrices=False);coef=vh.T@((v/(v*v+10))*(u.T@(y-y.mean())));pp=((X-xm)@coef+y.mean())*ys+ym;err=float(abs(pp-zz['ridge']).max());assert err<1e-3;diagnosticchecks.append({'fold':k,'checkpoint':key,'independent_readout_error':err})
    for k in range(5):dd,_=m.replace_fold(d,split,k,m.OUT/split/f'cnn-fold-{k}',False);pipes.append(pipeline(d,dd,split,k))
    # Full outer-fold newCNN/downstream exclusion replay; originalouter131alreadycertified.
    pert=dict(d);pert['y']=d['y'].copy();V=np.flatnonzero(d['fid']==0);pert['y'][V]+=10000;root=m.OUT/f'poison-{split}';root.mkdir();dd,paths=m.replace_fold(pert,split,0,root);p,_,rec=a.fold(dd,0,m.ARM,root/'pipeline');np.testing.assert_array_equal(p,z['primary'][V]);orig=e.read(m.OUT/split/'pipeline-fold-0/report.json');assert rec['outer_calibrator']==orig['outer_calibrator'] and rec['anchor_calibrators']==orig['anchor_calibrators'] and rec['selected']==orig['selected']
    for pp in paths:
        new=_repo_path(ROOT, pp);compare_models(new,m.OUT/split/'cnn-fold-0'/new.name)
    # Whole A-only anchor upstream must exclude B labels, including all deep source fits.
    A,B,_,_,_=d['cells'][0,0];pert=dict(d);pert['y']=d['y'].copy();pert['y'][B]+=50000;ipath=m.OUT/f'inner-poison-{split}';ipath.mkdir();innerpaths=[]
    for name in ['middle-0','deep-0-0','deep-0-1','deep-0-2']:
        old=m.OUT/split/'cnn-fold-0'/name;obj=torch.load(_repo_io(old/'model.pt'),weights_only=True,map_location='cpu');pp,rr=m.fit(pert,np.array(obj['train']),np.array(obj['validation']),obj['seed'],ipath/name);compare_models(ipath/name,old);innerpaths.append(name)
    # All downstream training labels in anchor are A and unchanged.
    dd,_=m.replace_fold(d,split,0,m.OUT/split/'cnn-fold-0',False);ap=a.affine(a.mixture(dd['deepP'][0,0],m.ARM),d['y'][A],a.mixture(dd['cells'][0,0][2],m.ARM))[0];aq=a.affine(a.mixture(dd['deepP'][0,0],m.ARM),pert['y'][A],a.mixture(dd['cells'][0,0][2],m.ARM))[0];np.testing.assert_array_equal(ap,aq)
    m.check();out={'passed':True,'diagnostic_readout_checks':diagnosticchecks,'CNN_checks':neuralchecks,'pipeline_checks':pipes,'outer80_poison_new_CNN_fits':len(paths),'outer_poison_models_predictions_calibrators_choices_exact':True,'innerB_poison_A_anchor_fits':len(innerpaths),'inner_anchor_exact':True,'original_outer131_training_exclusion_certificate_reused':split=='original','old_unchanged_KRR_DINO_training_exclusion_certificates_reused':True,'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start};s.write(m.OUT/f'validation-{split}.json',out);e.emit({'validation_passed':True,'split':split,'seconds':out['seconds'],'max_CNN_CPU_error':max(v['reload_max_error']['cpu'] for v in neuralchecks),'max_SVD_error':max(v['independentSVD_max_error'] for v in neuralchecks),'new_poison_fits':len(paths)+len(innerpaths)})
if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('split',choices=e.SPLITS);main(p.parse_args().split)
