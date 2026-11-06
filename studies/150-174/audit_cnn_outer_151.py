"""Reload all 45 existing outer CNNs and independently verify cached D4 means."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,csv
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_view_mean_151 as q
np=q.np;e=q.e;q.torch.set_num_threads(2);q.check();records=[];maximum=0.
for split in e.SPLITS:
 d=q.data(split)
 with q.outer_file(split).open() as f:rows={v['ID']:v for v in csv.DictReader(f)}
 assert set(rows)==set(d['ids']);np.testing.assert_array_equal([float(rows[i]['hardness']) for i in d['ids']],d['y']);
 if split=='original':
  manifest=e.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));folds=[manifest['development_folds'][i] for i in d['ids']]
 else:folds=[int(rows[i]['fold']) for i in d['ids']]
 np.testing.assert_array_equal(folds,d['fid'])
 for k in range(5):
  T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert not set(d['g'][T])&set(d['g'][V]);pp=[]
  for seed in q.x.SEEDS:
   file=q.x.ckpath(split,k,seed);obj=q.torch.load(_repo_io(file),map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in T];assert obj['normalization']=={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};vv=q.views.predict_views(d['images'][V],file,'mps');pp.append(vv);records.append({'split':split,'fold':k,'seed':seed,'checkpoint':str(file.relative_to(ROOT)),'sha256':e.sha(file),'train_scope_verified':True})
  mean=np.stack(pp).mean((0,1));expected=np.array([float(rows[d['ids'][i]]['cnn_mean3_d4']) for i in V]);err=float(abs(mean-expected).max());assert err<1e-4;maximum=max(maximum,err)
 e.emit({'split':split,'outer_CNNs_verified':15,'maximum_D4_mean_error':maximum})
r={'passed':True,'outer_models_verified':len(records),'all_outer_ID_fold_target_and_train_stats_exact':True,'D4_mean_max_error':maximum,'models':records,'plan_sha256':e.sha(q.OUT/'plan.json'),'source_sha256':e.sha(Path(__file__)),'test_read':False,'new_neural_fits':0};e.write(q.OUT/'outer-reload-audit.json',r);e.emit({'passed':True,'models':len(records),'D4_mean_max_error':maximum})
