from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent
d=np.load(_repo_path(HERE, 'data.npz'));z=np.load(_repo_path(HERE, 'domain-semantics/readout/oof.npz'));y=z['y'];b=z['base083'];p=z['H300_flow_system'];fold=z['fold'];dev=d['dev'];noise=d['noise'][dev];nobj=d['nobj'][dev];old=(b-y)**2;new=(p-y)**2;gain=old-new;rng=np.random.default_rng(2026101991);ix=rng.integers(len(y),size=(10000,len(y)));boot=np.sqrt(old[ix].mean(1))-np.sqrt(new[ix].mean(1));keep=np.ones(len(y),bool);top=np.argsort(gain)[-10:];keep[top]=False
masks={'all':np.ones(len(y),bool),'noise_low':noise<9.5,'noise_mid':(noise>=9.5)&(noise<12),'noise_high':noise>=12,'target_low':y<186.1664820705299,'target_mid':(y>=186.1664820705299)&(y<200.4092597154498),'target_high':y>=200.4092597154498,'grain_count_low':nobj<223.0833333333,'grain_count_mid':(nobj>=223.0833333333)&(nobj<368.3333333333),'grain_count_high':nobj>=368.3333333333}
rm=lambda a,m:float(np.sqrt(np.mean((a[m]-y[m])**2)))
groups={k:dict(n=int(m.sum()),base=rm(b,m),candidate=rm(p,m),gain=rm(b,m)-rm(p,m)) for k,m in masks.items()}
seed_results=[]
for seed in [0,100,200]:
 v=np.zeros(400)
 for f in range(5):
  q=np.flatnonzero(fold==f);c=np.load(_repo_path(HERE, f'conductivity/readout/physical-f{f}-s{seed}.npz'));a=np.load(_repo_path(HERE, f'domain-semantics/readout/H300_flow_only-f{f}-s{seed}.npz'));assert np.array_equal(c['query'],q) and np.array_equal(a['query'],q);v[q]=b[q]+.384*(~d['failed'][q])*(a['prediction']-c['prediction'])
 seed_results.append(dict(seed=seed,rmse=float(np.mean((v-y)**2)**.5),gain=float(np.mean(old)**.5-np.mean((v-y)**2)**.5)))
r=dict(rmse=rm(p,masks['all']),base_rmse=rm(b,masks['all']),gain=rm(b,masks['all'])-rm(p,masks['all']),fold_gains=[float(np.sqrt(old[fold==f].mean())-np.sqrt(new[fold==f].mean())) for f in range(5)],paired_id_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),top10_positive_contributors_removed_gain=float(np.sqrt(old[keep].mean())-np.sqrt(new[keep].mean())),top10_ids=z['ids'][top].tolist(),top10_sse_gain_fraction=float(gain[top].sum()/gain.sum()),positive_sse_images=int((gain>0).sum()),prediction_change_rms=float(np.mean((p-b)**2)**.5),groups=groups,seed_results=seed_results,caveat='Conditional reused400screen; IDbootstrapisnotproofagainstadaptiveoverfitandnotaprivateleaderboardprobability. H300auxiliarytrainedgloballyoncleanest250providedimages. No newPublicresult.')
(_repo_path(HERE, 'flow-finalist-diagnostics.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
