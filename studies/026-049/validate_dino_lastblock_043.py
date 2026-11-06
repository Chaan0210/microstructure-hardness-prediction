"""Read-only artifact reload checks; writes validation report, never fits."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dino_lastblock_043 as e
import numpy as np,torch,joblib
fv=e.fv;out=e.OUT;plan=json.loads((out/'plan.json').read_text());ids,y,groups,X=e.load();meta=json.loads((out/'prefix.json').read_text());assert meta['ids']==ids and fv.sha(out/'prefix.npy')==meta['sha256'];cache=np.load(_repo_io(out/'prefix.npy'),mmap_mode='r');torch.set_num_threads(2);m=e.ds.base.model().to('mps')
for n,h in {**plan['sources'],**plan['protected']}.items():assert fv.sha(_repo_path(ROOT, n))==h
# Official frozen source/weights and two deterministic prefix rows, labels unused.
official=json.loads((_repo_path(ROOT, 'artifacts/dino-frozen-037/smoke.json')).read_text())
for n,h in official['official_source_sha256'].items():assert fv.sha(e.ds.base.SOURCE/n)==h
prefixerr=0.
for j in range(2):
 z=e.prefix(m,torch.from_numpy(e.ds.inputs(ids[j],'full504')).to('mps')).cpu().numpy()[0];np.testing.assert_allclose(z,cache[j],atol=2e-3,rtol=0);prefixerr=max(prefixerr,float(abs(z-cache[j]).max()))
selection=json.loads((out/'selection.json').read_text());required=('original','20261031','20261101') if selection['advance'] else ('original',)
assert all((out/s/'report.json').exists() for s in required)
results={};count=0;maxerr=0.;cpuerr=0.
for split in required:
 d=out/split
 if not (d/'report.json').exists():continue
 report=json.loads((d/'report.json').read_text());assert len(report['folds'])==5 and [f['fold'] for f in report['folds']]==list(range(5));fid,ref=e.rows(split,ids);oldreport=json.loads((_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, 'report.json')).read_text())
 with (d/'development_oof.csv').open() as f:rows=list(csv.DictReader(f))
 assert [r['ID'] for r in rows]==ids;np.testing.assert_array_equal(y,[float(r['hardness']) for r in rows]);np.testing.assert_array_equal(fid,[int(r['fold']) for r in rows]);p=np.array([float(r['adapted']) for r in rows]);blend=np.array([float(r['adapted_blend']) for r in rows]);assert np.isfinite(p).all();np.testing.assert_allclose(blend,.75*ref['reference003']+.25*p,atol=1e-12,rtol=0)
 for k,rec in enumerate(report['folds']):
  tr=np.flatnonzero(fid!=k);va=np.flatnonzero(fid==k);assert not set(groups[tr])&set(groups[va]);assert rec['train_ids']==[ids[j] for j in tr] and rec['validation_ids']==[ids[j] for j in va]
  cp=d/f'fold-{k}'/'model.pt';hp=_repo_path(ROOT, 'artifacts/dino-spatial-039/regression', split, f'fold-{k}', 'full504.joblib');assert fv.sha(cp)==rec['checkpoint_sha256'] and fv.sha(hp)==rec['head_source_sha256']==oldreport['folds'][k]['models']['full504']['sha256']
  obj=joblib.load(_repo_io(hp));r=obj['model'];assert obj['train_ids']==rec['train_ids'] and obj['validation_ids']==rec['validation_ids'];np.testing.assert_allclose(r.scaler.mean_,X[tr].mean(0),atol=1e-10,rtol=0);assert abs(r.ymean-y[tr].mean())<1e-10 and abs(r.ystd-y[tr].std())<1e-10
  ck=torch.load(_repo_io(cp),map_location='cpu',weights_only=True);assert ck['epoch']==8 and ck['train_ids']==rec['train_ids'] and ck['validation_ids']==rec['validation_ids'];assert ck['ymean']==r.ymean and ck['ystd']==r.ystd
  tail=e.Tail(m,r);initial={n:v.detach().cpu() for n,v in tail.state_dict().items()}
  for n,v in ck['state'].items():
   if not n.startswith('block.'):assert torch.equal(v,initial[n])
  assert any(not torch.equal(v,initial[n]) for n,v in ck['state'].items() if n.startswith('block.'))
  tail.load_state_dict(ck['state']);tail=tail.to('mps');pred=e.predict(tail,cache,va)*r.ystd+r.ymean;np.testing.assert_allclose(pred,p[va],atol=1e-5,rtol=0);maxerr=max(maxerr,float(abs(pred-p[va]).max()));assert all(not z.training for z in tail.modules());assert all(p.requires_grad==n.startswith('block.') for n,p in tail.named_parameters())
  if k==0:
   tail=tail.cpu()
   with torch.no_grad():pc=tail(torch.from_numpy(np.array(cache[va[:2]],copy=True))).numpy()*r.ystd+r.ymean
   np.testing.assert_allclose(pc,p[va[:2]],atol=2e-3,rtol=0);cpuerr=max(cpuerr,float(abs(pc-p[va[:2]]).max()))
  assert [q['epoch'] for q in rec['curve']]==list(range(1,9)) and all(np.isfinite(q['train_objective']) for q in rec['curve']);count+=1;del tail;torch.mps.empty_cache()
 assert abs(fv.rmse(y,p)-report['standalone_rmse'])<1e-10 and abs(fv.rmse(y,blend)-report['blend_rmse'])<1e-10
 gain=fv.rmse(y,ref['full504_blend'])-fv.rmse(y,blend);assert abs(gain-report['gain'])<1e-10
 fg=[fv.rmse(y[fid==k],ref['full504_blend'][fid==k])-fv.rmse(y[fid==k],blend[fid==k]) for k in range(5)];np.testing.assert_allclose(fg,report['fold_gains'],atol=1e-12,rtol=0)
 ix=np.random.default_rng(20261128).integers(0,400,(3000,400));diff=np.sqrt(np.mean((y-ref['full504_blend'])[ix]**2,axis=1))-np.sqrt(np.mean((y-blend)[ix]**2,axis=1));assert abs(float(diff.std())-report['bootstrap_sd'])<1e-12;np.testing.assert_allclose(np.percentile(diff,[2.5,97.5]),report['conditional_ci95'],atol=1e-12,rtol=0)
 results[split]={k:v for k,v in report.items() if k!='folds'}
 if split=='original':
  sel=json.loads((out/'selection.json').read_text());assert sel['advance']==bool(gain>report['bootstrap_sd'] and sum(g>0 for g in report['fold_gains'])>=3)
assert count==5*len(required) and set(results)==set(required)
for n,h in plan['protected'].items():assert fv.sha(_repo_path(ROOT, n))==h
summary={'results':results,'models_reloaded':count,'maximum_prediction_reload_error':maxerr,'cpu_prediction_max_error':cpuerr,'prefix_recompute_max_error':prefixerr,'all_frozen_norm_head_tensors_unchanged':True,'lastblock_only_trainable':True,'fold_scaler_target_group_ids_checkpoint_and_head_hashes_verified':True,'best005_protected':True,'source_review':'Independent read-only audit: no blocking gradient/leakage error. Resume shortcut requires separate hash/ID checks; validation here checks all artifacts.','scope':'Development400 only; no fit, holdout/test access, package install, cost or upload.','limitations':'Repeated model selection on reused development400; bootstrap screening is not significance or independent validation.'}
path=_repo_path(ROOT, 'reports/dino-043-validation.json')
if path.exists():assert json.loads(path.read_text())==summary
else:fv.write(path,summary)
print(json.dumps(summary,indent=2))
