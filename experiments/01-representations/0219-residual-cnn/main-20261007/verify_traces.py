"""No new fits: exact stored training-trace checks for planned34 replays."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json
from pathlib import Path
O=Path(__file__).resolve().parent;S=O.parent;sys.path.insert(0,str(S))
import joblib,numpy as np
from contracts import sha,require,lock_manifest

def main():
 completion=json.loads((_repo_path(O, 'fit-completion.json')).read_text());require(completion['passed'],'Main incomplete');records=json.loads((_repo_path(O, 'CNN-records.json')).read_text());stages={p:[r for r in records if r['phase']==p] for p in ['main','verification']};require(len(stages['main'])==145 and len(stages['verification'])==34,'CNN record count')
 def obj(stage,n):
  r=stages[stage][n-1];require(sha(_repo_path(O, r['path']))==r['sha256'],'CNN artifact changed');return joblib.load(_repo_path(O, r['path']))
 pairs=[(n,n) for n in range(1,30)]+[(1,30),(2,31),(3,32),(28,33),(29,34)];checks=[]
 for mn,vn in pairs:
  m=obj('main',mn)['meta'];v=obj('verification',vn)['meta']
  for key in ['seed','train','train_ids','target_sha256','normalization','schedule_sha256','state_sha256','residual']:require(m[key]==v[key],'Replay trace mismatch:'+key)
  require(len(m['curve'])==len(v['curve'])==80,'CNN epochs');mc=[(c['epoch'],c['train_rmse_augmented']) for c in m['curve']];vc=[(c['epoch'],c['train_rmse_augmented']) for c in v['curve']];require(mc==vc,'Training loss curve replay mismatch');checks.append({'main':mn,'verification':vn,'weights_normalization_orders_D4_curves_exact':True})
 controls=[]
 for k in range(5):
  true=obj('main',29*k+28)['meta'];permuted=obj('main',29*k+29)['meta']
  for key in ['seed','train','train_ids','normalization','schedule_sha256','residual']:require(true[key]==permuted[key],'True/permutation execution conditions mismatch:'+key)
  require(true['seed']==20264000+k and len(true['curve'])==len(permuted['curve'])==80,'Control seed/epoch');controls.append({'fold':k,'same_seed_trainIDs_normalization_batchorder_D4_schedule':True,'target_hashes_differ':true['target_sha256']!=permuted['target_sha256']})
 result={'passed':True,'CNN_replays_checked':34,'all80epoch_loss_curves_exact':True,'all_weights_normalization_target_orders_exact':True,'true_permuted_controls':controls,'replay_pairs':checks,'new_fits':0,'limitations':'Order/D4 hash is reconstructed from the locked RNG loop and seed, not sampled from internal tensor execution; fixed source has been independently reviewed. Memory/timing telemetry intentionally excluded from equality.'};lock_manifest(_repo_path(O, 'trace-validation.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ['replay_pairs','true_permuted_controls']}))
if __name__=='__main__':main()
