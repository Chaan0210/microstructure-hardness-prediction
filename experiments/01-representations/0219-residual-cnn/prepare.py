"""One-time lock of source/cache contracts; no model fitting or feature extraction."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=_RepoPath(__file__).resolve().parents[3];sys.path.insert(0,str(O))
import numpy as np
from contracts import sha,require,lock_manifest
old=json.loads((_repo_path(R, 'experiments/cnn-nested-204/plan.json')).read_text());feas=json.loads((_repo_path(R, 'docs/records/plans/d2-residual-cnn-20261007/feasibility.json')).read_text());feature_files={'K':'artifacts/run-012/development_features.npz','S':'artifacts/dino-multilayer-044/development_features.npz','B':'artifacts/dinob-062/development_features.npz','L':'experiments/dinol-177/features.npz'}
with np.load(_repo_path(R, 'experiments/cnn-nested-204/cache/data.npz'),allow_pickle=False) as z:ids=z['ids'].copy();y=z['y'].copy()
inputs={p:sha(_repo_path(R, p)) for p in ['experiments/cnn-nested-204/cache/data.npz',*feature_files.values(),'artifacts/run-012/report.json','experiments/cnn-nested-204/plan.json','docs/records/plans/d2-residual-cnn-20261007/feasibility.json']};require(inputs['experiments/cnn-nested-204/cache/data.npz']==old['data_sha256'],'204 data changed')
registry=[];graphs=[]
for cell in feas['minimal_plan_cells']:
 k=cell['outer'];nodes=[n for n in old['nodes'] if n['si']==0 and n['outer']==k and n['middle']==0];require(len(nodes)==4,'Selected cached nodes count');cells=[]
 for n in nodes:
  src=_repo_path(R, n['source']);record=json.loads((src/'report.json').read_text());require(record['train_indices']==n['train'],'Old train membership mismatch');files={}
  for key,name in [('kernel','kernel.joblib'),('CNN-20261003','cnn.pt'),*[(f'{arm}-{s}',f'{arm}-{s}.pt') for arm in ['S','B'] for s in [20261128,20261205,20261219]]]:
   p=src/name;digest=record['files_sha256'][name];require(sha(p)==digest,'Old base model hash');files[key]={'path':str(p.relative_to(R)),'sha256':digest}
  cr=json.loads((_repo_path(R, n['old3_source'], 'report.json')).read_text())
  for s in [20261023,20261043]:
   p=_repo_path(R, n['old3_source'], f'seed-{s}.pt');rec=next(a for a in cr['models'] if a['base_seed']==s);require(sha(p)==rec['checkpoint_sha256'],'Old CNN seed hash');files[f'CNN-{s}']={'path':str(p.relative_to(R)),'sha256':sha(p)}
  if n['type']=='middle':
   p=_repo_path(R, f'artifacts/nested-fusion-112/original/fold-{k}/inner-0/shortB/model.pt');rec=json.loads(p.with_name('report.json').read_text());require(sha(p)==rec['model_sha256'],'Old short hash');files['short']={'path':str(p.relative_to(R)),'sha256':sha(p)}
  else:
   deep=[e for e in feas['new_deep_nodes'] if e['outer']==k and e['middle_inside_anchor']==n['deep']];require(len(deep)==3,'New deep count');cells.append({'ell':n['deep'],'C':n['train'],'D':n['validation'],'offset':n['offset'],'deep':[{'E':e['train'],'F':e['query'],'offset':e['offset']} for e in deep]})
  registry.append({'train':n['train'],'offset':n['offset'],'target_sha256':hashlib.sha256(np.asarray(y[n['train']],dtype=float).tobytes()).hexdigest(),'files':files,'cache_name':n['name']})
  inputs[str((_repo_path(R, 'experiments/cnn-nested-204/cache', f"{n['name']}.npz")).relative_to(R))]=n['cache_sha256']
 graphs.append({'outer':k,'T':cell['T'],'V':cell['V'],'A':cell['A'],'B':cell['B'],'offset':cell['root_offset'],'cells':sorted(cells,key=lambda c:c['ell'])})
require(len(registry)==20 and len(graphs)==5,'Graph counts');contract={'scope':'development400 only','ids':ids.tolist(),'feature_files':feature_files,'inputs':inputs,'blocks':old['blocks'],'kblocks':json.loads((_repo_path(R, 'artifacts/run-012/report.json')).read_text())['blocks'],'graphs':graphs,'registry':registry,'feature_provenance':'S/B/L cached from fixed official pretrained backbones, not fine-tuned on400 labels; all supervised readout/cache training IDs verified separately','protected':feas['protected']};lock_manifest(_repo_path(O, 'data-contract.json'),contract);print(json.dumps({'locked':True,'cached_base_nodes':20,'new_deep_nodes':45,'data_contract_sha256':sha(_repo_path(O, 'data-contract.json'))}))
