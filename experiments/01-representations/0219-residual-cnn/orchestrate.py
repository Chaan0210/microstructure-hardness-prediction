"""Train one complete paired package, never inspect evaluation labels.
No command-line main entrypoint is provided at this implementation-only stage.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import numpy as np
from contracts import Labels,require,lock_manifest,sha
from anchor import fit_complete_anchor
from backend import training_guard,fit_cnn
from bundle import freeze_anchor,residual_targets,save_bundle

def fit_pair(features,train_labels,graph,B,V,backend,folder,outer,protocol_sha,split_sha):
 training_guard();A=np.array(graph.A);B=np.asarray(B);V=np.asarray(V);T=np.sort(np.r_[A,B]);features.partition(np.arange(len(features.ids)),T,V);features.partition(T,A,B);require(set(train_labels.ids)==set(T),'Pair labels must be exactly outer train T');folder=Path(folder);require(not folder.exists(),'Refuse pair overwrite');folder.mkdir()
 anchor=fit_complete_anchor(features,train_labels.restrict(A),graph,backend);path=folder/'frozen-anchor.joblib';binding=freeze_anchor(anchor,path,protocol_sha);r=residual_targets(features,B,train_labels.restrict(B),path,binding);permutation=np.random.default_rng(20264050+outer).permutation(len(B));permuted=Labels(B,r.take(B)[permutation]);records={};shared_stats=(float(r.values.mean()),max(float(r.values.std()),1e-6))
 for arm,target in [('true',r),('permuted',permuted)]:
  model,curve=fit_cnn(features,target,20264000+outer,backend.device,residual=True,anchor_sha256=binding['anchor_sha256'],target_stats=shared_stats);manifest=save_bundle(folder/arm,path,binding,model,features,B,protocol_sha,split_sha);records[arm]={'curve':curve,'manifest':manifest,'prediction_V':anchor.predict(features,V)+model.predict(features,V)}
 result={'R0':anchor.predict(features,V),'arms':records,'V':V,'no_evaluation_labels_received':True}
 with (folder/'predictions.npz').open('xb') as handle:np.savez_compressed(_repo_io(handle),V=V,R0=result['R0'],R1=records['true']['prediction_V'],Rperm=records['permuted']['prediction_V'])
 lock_manifest(folder/'training-report.json',{'outer':outer,'protocol_sha256':protocol_sha,'split_sha256':split_sha,'anchor_binding':binding,'residual_shared_normalization':shared_stats,'permutation':permutation.tolist(),'curves':{k:v['curve'] for k,v in records.items()},'backend_events':backend.events,'predictions_sha256':sha(folder/'predictions.npz'),'no_evaluation_labels_received':True})
 return result
