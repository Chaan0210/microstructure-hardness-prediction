"""Verify provenance, matching, pixels and descriptive arithmetic; no model fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import diagnose_images_094 as d
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment
start=time.perf_counter();o=d.OUT;plan=json.loads((o/'plan.json').read_text());lock=json.loads((o/'matching-locked.json').read_text());report=json.loads((_repo_path(ROOT, 'reports/image-diagnostics-094.json')).read_text());assert d.sha(_repo_path(ROOT, 'diagnose_images_094.py'))==plan['source_sha256'];assert d.sha(_repo_path(ROOT, 'diagnose_errors_068.py'))==plan['prior_metrics_source_sha256'];assert d.sha(o/'plan.json')==lock['plan_sha256'];assert d.sha(o/'image_metrics.npz')==lock['metrics_sha256'];assert d.sha(o/'matching-locked.json')==report['matching_locked_sha256']
with np.load(_repo_io(o/'image_metrics.npz'),allow_pickle=False) as z:ids=z['ids'].tolist();names=z['names'].tolist();X=z['X'];R=z['rank'];D=z['distance']
assert ids==d.ids400() and X.shape==(400,30) and np.isfinite(X).all()
with np.load(_repo_path(ROOT, 'artifacts/error-diagnostics-068/development_diagnostics.npz'),allow_pickle=False) as z:
 assert z['ids'].tolist()==ids;oldnames=z['names'].tolist();np.testing.assert_array_equal(z['X'],X[:,[names.index(n) for n in oldnames]])
prior=json.loads((_repo_path(ROOT, 'artifacts/error-diagnostics-068/report.json')).read_text());raw_replay=0
for i,ident in enumerate(ids):
 p=_repo_path(ROOT, 'data/train', f'{ident}.png');assert d.sha(p)==lock['image_sha256'][ident]==prior['image_sha256'][ident]
 with Image.open(_repo_io(p)) as im:assert im.mode=='L' and im.size==(256,256);a=np.array(im)
 values=d.additional(a);np.testing.assert_array_equal([values[n] for n in values],X[i,[names.index(n) for n in values]]);raw_replay+=1
rr=np.column_stack([(rankdata(v,method='average')-.5)/400 for v in X.T]);np.testing.assert_array_equal(rr,R);D2=sum(cdist(R[:,[names.index(n) for n in ns]],R[:,[names.index(n) for n in ns]],metric='sqeuclidean')/len(ns)/5 for ns in d.GROUPS.values());np.testing.assert_allclose(np.sqrt(D2),D,rtol=0,atol=2e-16)
ti=np.array([ids.index(i) for i in d.TOP]);rest=np.array([i for i in range(400) if i not in ti]);slots=np.repeat(ti,3);ar,ac=linear_sum_assignment(D2[slots][:,rest]);ci=rest[ac]
for k,pair in enumerate(lock['pairs']):
 assert pair['anchor']==ids[ti[k]];assert set(c['ID'] for c in pair['controls'])==set(ids[j] for j in ci[3*k:3*k+3]);assert [c['rank_among395'] for c in pair['controls']]==[1,2,3]
# Pixel verification: displayed panels are exact native source, with no contrast rescaling.
with Image.open(_repo_io(o/'matched-images.png')) as im:canvas=np.array(im)
for row,pair in enumerate(lock['pairs']):
 for col,ident in enumerate([pair['anchor']]+[v['ID'] for v in pair['controls']]):
  with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:raw=np.array(im.convert('RGB'))
  np.testing.assert_array_equal(canvas[row*292+28:row*292+284,col*276+10:col*276+266],raw)
checks={}
for split in ['original','20261031','20261101']:
 folder='artifacts/patch-attention-092/original' if split=='original' else f'artifacts/patch-replication-093/{split}';p=_repo_path(ROOT, folder, 'development_predictions.npz');vfile='reports/patch-attention-092-validation.json' if split=='original' else 'reports/patch-replication-093-validation.json';v=json.loads((_repo_path(ROOT, vfile)).read_text());assert d.sha(p)==v['results'][split]['predictions_sha256']==report['results'][split]['outcome_source_sha256']
 with np.load(_repo_io(p),allow_pickle=False) as z:assert z['ids'].tolist()==ids;y=z['y'];a=z['reference011'];b=z['attention_blend']
 for key,ix in [('top5',ti),('matched15',ci),('other395',rest)]:
  r=report['results'][split][key];gain=np.linalg.norm(y[ix]-a[ix])/np.sqrt(len(ix))-np.linalg.norm(y[ix]-b[ix])/np.sqrt(len(ix));assert abs(r['RMSE_gain']-gain)<1e-13
 checks[split]=True
# Analytical frequency fixtures check interpretation, not competition ground truth.
xx=np.tile(np.arange(256),(256,1));coarse=np.uint8(np.rint(128+60*np.sin(2*np.pi*xx/64)));middle=np.uint8(np.rint(128+60*np.sin(2*np.pi*xx/16)));c=d.additional(coarse);m=d.additional(middle);flat=d.additional(np.full((256,256),128,np.uint8));assert c['spectral_low_fraction']>.99 and m['spectral_mid_fraction']>.99;assert np.isfinite(list(flat.values())).all() and flat['dark_component_density']==0
protected=json.loads((_repo_path(ROOT, 'reports/candidate011-public-submission.json')).read_text())['protected_sha256']
for p,h in protected.items():assert d.sha(_repo_path(ROOT, p))==h,p
result={'passed':True,'development_rows':400,'prior20_metrics_exact':True,'new10_metrics_raw_replay_rows':raw_replay,'matching_independent_cdist_max_error':float(abs(np.sqrt(D2)-D).max()),'all15controls_are_unique_and_actual_nearest3':True,'display20_panels_native_pixels_exact':True,'all400_image_hashes_match_prior068':True,'outcome_sources_match_verified092_093':checks,'analytical_fixture_spectral_fractions':{'64px_low':c['spectral_low_fraction'],'16px_mid':m['spectral_mid_fraction'],'flat_finite':True},'protected011_artifacts_unchanged':len(protected),'no_training':True,'limits':'New10 replay uses same implementation; distance/arithmetic checked by separate formulations. Fixtures validate frequency meaning, not grain/pore or hardness truth. No holdout/test image or label reading. Protected test prediction artifact hashed only, never parsed.','seconds':time.perf_counter()-start,'source_sha256':d.sha(Path(__file__))};d.write(_repo_path(ROOT, 'reports/image-diagnostics-094-validation.json'),result);print(json.dumps(result),flush=True)
