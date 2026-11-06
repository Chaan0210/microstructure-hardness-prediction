"""Fixed 32D cubical H1 versus same-capacity old H0 residual extension of018."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
import sys,time,json,resource
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
import directional_energy_148 as core
import cubical_h1_168 as f
import persistence_features_107 as old
R=_RepoPath(__file__).resolve().parents[1];ROOT=_RepoPath(__file__).resolve().parents[1];OUT=_repo_path(R, 'artifacts/cubical-h1-168');s=core.s;a=core.a;e=core.e;ARMS=('h0','h1');torch.set_num_threads(2);threadpool_limits(2)
def write(p,z):
    with p.open('x') as fp:json.dump(z,fp,indent=2,allow_nan=False)
def prepare():
    assert e.read(_repo_path(OUT, 'implementation-validation.json'))['passed'] and e.read(_repo_path(OUT, 'pilot.json'))['passed']
    p=e.read(_repo_path(OUT, 'plan.json'));files=['cubical_study_168.py','cubical_h1_168.py','persistence_features_107.py','directional_energy_148.py','bootstrap_selector_144.py','nonlinear_residual_115.py','reports/nested-selector-146-audit.json','artifacts/run-001/split_manifest.json','artifacts/nested-calibration-074/extraction.json']
    for split in e.SPLITS:
        files += [f'artifacts/bootstrap-selector-144/{split}/report.json',f'artifacts/bootstrap-selector-144/{split}/predictions.npz']
    for name in ['plan.json','implementation-validation.json','pilot.json']:files.append(str((_repo_path(OUT, name)).relative_to(R)))
    rec={'source_sha256':{p:e.sha(_repo_path(R, p)) for p in files},'protected':p['protected'],'feature_width_per_arm':32,'base_width':585,'fit_width':617,'kind_list':list(s.KINDS),'bootstrap_draws':512,'model':'Exact018 cachedbase/anchors; appendone32wideblock. TrainonlyMinMaxclip/StandardScaler/1sqrtblocksize, fixedexisting residuallearner settings. H0control diagnostics only, not fallback.','gate':'OriginalH1 below BOTH018 andH0 plusatleast3/5 improvingfolds vs018, all independentaudits. Onlythenexisting2resplits; finalpooledMSEbelowboth.','resources':'pairedfeature400<300seconds, firstmodelblock projected90blocks<300seconds,RSS<2GB; CPU2.','scope':'400developmentonly. No newinstallation/download/holdout/test/full500/CSV/upload.','same400_exploratory':True}
    write(_repo_path(OUT, 'model-lock.json'),rec);print('MODEL_LOCKED',e.sha(_repo_path(OUT, 'model-lock.json')),flush=True)
def check():
    p=e.read(_repo_path(OUT, 'model-lock.json'))
    for file,h in {**p['source_sha256'],**p['protected']}.items():assert e.sha(_repo_path(R, file))==h,file
    return p

def extract():
    check();ids=sorted(e.read(_repo_path(R, 'artifacts/run-001/split_manifest.json'))['development_folds']);assert len(ids)==400
    hashes=e.read(_repo_path(R, 'artifacts/nested-calibration-074/extraction.json'))['image_sha256'];H=[];C=[];counts=[];start=time.perf_counter()
    for i,ident in enumerate(ids):
        path=_repo_path(R, 'data/train', f'{ident}.png');assert e.sha(path)==hashes[ident]
        with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);x=np.array(im)
        hs=[];cs=[];ct=[]
        for v in (x,255-x):
            pd=f.diagram(v);pp=old.diagram(v);hs.append(f.encode(pd,x.size));cs.append(f.encode(pp,x.size));ct.append(len(pd))
        H.append(np.concatenate(hs));C.append(np.concatenate(cs));counts.append(ct)
        if (i+1)%40==0:print('EXTRACTED',i+1,'seconds',round(time.perf_counter()-start,2),flush=True)
    H=np.array(H);C=np.array(C);assert H.shape==C.shape==(400,32) and np.isfinite(H).all() and np.isfinite(C).all()
    with np.load(_repo_path(OUT, 'pilot-features.npz')) as p:
        ix=[ids.index(v) for v in p['ids']];np.testing.assert_array_equal(H[ix],p['h1']);np.testing.assert_array_equal(C[ix],p['h0'])
    with (_repo_path(OUT, 'features.npz')).open('xb') as fp:np.savez_compressed(_repo_io(fp),ids=np.array(ids),h1=H,h0=C,counts=np.array(counts))
    r={'passed':time.perf_counter()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9,'rows':400,'width':32,'cache_sha256':e.sha(_repo_path(OUT, 'features.npz')),'seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all400_image_hashes_verified':True,'pilot16_exactly_reproduced':True,'labels_used':False,'stats':{k:{'min':float(v.min()),'max':float(v.max()),'min_column_std':float(v.std(0).min()),'max_column_std':float(v.std(0).max()),'zero_variance_columns':int((v.std(0)==0).sum())} for k,v in [('h0',C),('h1',H)]}}
    write(_repo_path(OUT, 'extraction.json'),r);assert r['passed'];check();print('EXTRACTION_PASS',json.dumps(r),flush=True)
def data(split,arm):
    d=s.data(split);rr=e.read(s.OUT/split/'report.json');z=a.readnp(s.OUT/split/'predictions.npz',rr['predictions_sha256']);assert z['ids'].tolist()==d['ids'];np.testing.assert_array_equal(z['y'],d['y']);d['reference018']=z['primary'];rec=e.read(_repo_path(OUT, 'extraction.json'));assert rec['passed'];v=a.readnp(_repo_path(OUT, 'features.npz'),rec['cache_sha256']);assert v['ids'].tolist()==d['ids'];assert d['X'].shape[1]==585;d['X']=np.column_stack([d['X'],v[arm]]);d['blocks']=list(d['blocks'])+['topology168']*32;assert d['X'].shape==(400,617);return d

def run(split):
    check()
    if split!='original':assert e.read(_repo_path(OUT, 'original/report.json'))['advance'] and e.read(_repo_path(OUT, 'validation-original.json'))['passed'] and e.read(_repo_path(OUT, 'inner-poison-original.json'))['passed']
    D={arm:data(split,arm) for arm in ARMS};d=D['h1'];out=_repo_path(OUT, split);out.mkdir();P={arm:np.full(400,np.nan) for arm in ARMS};hashes={};start=time.perf_counter()
    for k in range(5):
        src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor={arm:[] for arm in ARMS};control=[]
        for j in range(3):
            br=e.read(src/f'block-{j}/report.json');bz=a.readnp(src/f'block-{j}/predictions.npz',br['predictions_sha256']);B=bz['block_indices'];assert not set(B)&set(V)
            for arm in ARMS:
                t=time.perf_counter();path=out/f'{arm}-fold-{k}-block-{j}';pp,rr=core.system(D[arm],B,bz['anchor_predictions'],V,a.r.CVSEED+100*k+j,s.SEED+10000*e.SPLITS.index(split)+100*k+j,path);cor[arm].append(pp);hashes[path.name+'/report.json']=e.sha(path/'report.json')
                if split=='original' and k==j==0 and arm=='h0':
                    bm={'first_block_seconds':time.perf_counter()-t,'projected90blocks_seconds':90*(time.perf_counter()-t),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};bm['passed']=bm['projected90blocks_seconds']<300 and bm['RSS_bytes']<2e9;write(_repo_path(OUT, 'model-benchmark.json'),bm);assert bm['passed'];print('BENCHMARK',json.dumps(bm),flush=True)
            rr=e.read(s.OUT/split/f'fold-{k}-block-{j}/report.json');zz=a.readnp(s.OUT/split/f'fold-{k}-block-{j}/predictions.npz',rr['predictions_sha256']);control.append(zz['primary'])
        np.testing.assert_array_equal(fz['base']+np.mean(control,0),d['reference018'][V])
        for arm in ARMS:P[arm][V]=fz['base']+np.mean(cor[arm],0)
        print('FOLD_DONE',split,k,flush=True)
    scores={'reference018':e.rmse(d['y'],d['reference018']),**{arm:e.rmse(d['y'],v) for arm,v in P.items()}};gain=[e.rmse(d['y'][d['fid']==k],d['reference018'][d['fid']==k])-e.rmse(d['y'][d['fid']==k],P['h1'][d['fid']==k]) for k in range(5)]
    with (out/'predictions.npz').open('xb') as fp:np.savez_compressed(_repo_io(fp),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],reference018=d['reference018'],**P)
    r={'split':split,'RMSE':scores,'fold_gains_vs018':gain,'advance':scores['h1']<min(scores['reference018'],scores['h0']) and sum(g>0 for g in gain)>=3,'models_including_none':600,'block_reports_sha256':hashes,'predictions_sha256':e.sha(out/'predictions.npz'),'seconds':time.perf_counter()-start};write(out/'report.json',r);check();print('RESULT',json.dumps({k:v for k,v in r.items() if k!='block_reports_sha256'}),flush=True)

def validate_main(split):
 check();D={arm:data(split,arm) for arm in ARMS};d=D['h1'];out=_repo_path(OUT, split);rr=e.read(out/'report.json');saved=a.readnp(out/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);count=0;maxerr=0.
 for k in range(5):
  src=s.q.OUT/split/f'adjacent-fold-{k}';fr=e.read(src/'report.json');fz=a.readnp(src/'predictions.npz',fr['predictions_sha256']);V=fz['outer_indices'];cor={arm:[] for arm in ARMS}
  for j in range(3):
   old=e.read(src/f'block-{j}/report.json');oz=a.readnp(src/f'block-{j}/predictions.npz',old['predictions_sha256']);B=oz['block_indices']
   for arm in ARMS:
    path=out/f'{arm}-fold-{k}-block-{j}';assert e.sha(path/'report.json')==rr['block_reports_sha256'][path.name+'/report.json'];pp,c,err=core.verify_system(D[arm],B,oz['anchor_predictions'],V,path);cor[arm].append(pp);count+=c;maxerr=max(maxerr,err)
    if k==j==0:
     pert=dict(D[arm]);pert['y']=d['y'].copy();pert['y'][V]+=100000;rp,rec=core.system(pert,B,oz['anchor_predictions'],V,a.r.CVSEED,s.SEED+10000*e.SPLITS.index(split),_repo_path(OUT, f'poison-{split}-{arm}'));np.testing.assert_array_equal(rp,pp)
  for arm in ARMS:np.testing.assert_array_equal(fz['base']+np.mean(cor[arm],0),saved[arm][V])
 r={'passed':True,'split':split,'models_verified':count,'independent_max_error':maxerr,'bootstrap_independent_exact':True,'outer_target_poison_exact':True,'upstream146_certificate':e.sha(_repo_path(ROOT, 'reports/nested-selector-146-audit.json')),'report_sha256':e.sha(out/'report.json')};e.write(_repo_path(OUT, f'validation-{split}.json'),r);check();e.emit(r)

def validate(split):
 validate_main(split);count=0
 for arm in ARMS:
  d=data(split,arm);path=_repo_path(OUT, split, f'{arm}-fold-0-block-0');r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);it=np.array(r['splits'][0]['train_indices']);iv=np.array(r['splits'][0]['validation_indices']);B=z['block_indices'];lookup={int(v):i for i,v in enumerate(B)};res=d['y'].copy();res[B]=z['residual'];res[iv]+=23456
  for ci,kind in enumerate(s.KINDS):
   m=core.n.fit(d['X'][it],res[it],d['blocks'],kind);p=core.n.predict(m,d['X'][iv]);expected=z['inner_predictions'][ci,[lookup[int(v)] for v in iv]];np.testing.assert_array_equal(p,expected);count+=1
 e.write(_repo_path(OUT, f'inner-poison-{split}.json'),{'passed':True,'fresh_fits':count,'H_labels_excluded_from_innerfit':True,'scope':'CachedA anchors excludeallB; perturbHresiduals andrefitI. Selectorweights intentionallydependonH OOF loss.'});check();e.emit({'inner_poison_passed':True,'split':split,'fresh_fits':count})

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='extract':extract()
    elif mode.startswith('validate-'):validate(mode[9:])
    else:run(mode)
