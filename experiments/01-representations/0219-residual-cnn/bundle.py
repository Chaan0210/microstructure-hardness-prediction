"""Bind residual targets to one frozen complete anchor, never swap in full-data021."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,joblib
import numpy as np
from contracts import sha,require,Labels,lock_manifest

def freeze_anchor(anchor,path,protocol_sha):
 p=Path(path);require(not p.exists(),'Refuse anchor overwrite');joblib.dump(anchor,_repo_io(p))
 rec={'anchor_sha256':sha(p),'train':list(map(int,anchor.train)),'protocol_sha256':protocol_sha};lock_manifest(p.with_suffix('.json'),rec);return rec

def residual_targets(features,B,labels,anchor_path,binding):
 require(sha(anchor_path)==binding['anchor_sha256'],'Anchor changed before residual targets');anchor=joblib.load(_repo_io(anchor_path));require(list(map(int,anchor.train))==binding['train'],'Anchor train mismatch');require(set(B).isdisjoint(anchor.train),'Anchor/residual overlap');require(not set(features.groups[B])&set(features.groups[list(anchor.train)]),'Anchor/residual group overlap');require(set(labels.ids)==set(B),'Residual label capability must be exactly B')
 r=labels.take(B)-anchor.predict(features,B);return Labels(B,r)

def save_bundle(folder,anchor_path,binding,residual,features,B,protocol_sha,split_sha):
 import shutil
 folder=Path(folder);require(not folder.exists(),'Refuse bundle overwrite');require(sha(anchor_path)==binding['anchor_sha256'],'Anchor changed');require(protocol_sha==binding['protocol_sha256'],'Protocol changed');require(tuple(residual.train)==tuple(B),'Residual train mismatch')
 require(getattr(residual,'anchor_sha256',None)==binding['anchor_sha256'],'Residual trained against different anchor');require(set(B).isdisjoint(binding['train']),'Bundle overlap');require(not set(features.groups[B])&set(features.groups[binding['train']]),'Bundle group overlap')
 folder.mkdir();shutil.copyfile(_repo_io(anchor_path),_repo_io(folder/'anchor.joblib'));joblib.dump(residual,_repo_io(folder/'residual.joblib'));manifest={'anchor_sha256':sha(folder/'anchor.joblib'),'residual_sha256':sha(folder/'residual.joblib'),'anchor_train':binding['train'],'residual_train':list(map(int,B)),'anchor_ids':features.ids[binding['train']].tolist(),'residual_ids':features.ids[B].tolist(),'protocol_sha256':protocol_sha,'split_sha256':split_sha,'policy':'paired A-only complete anchor + B-only CNN; no anchor substitution','combination':1.0,'feature_schema':{k:list(v.shape[1:]) for k,v in features.arrays.items()},'blocks':features.blocks,'kblocks':features.kblocks};lock_manifest(folder/'manifest.json',manifest);return manifest

def load_bundle(folder,protocol_sha,split_sha):
 folder=Path(folder);m=json.loads((folder/'manifest.json').read_text());require(m['protocol_sha256']==protocol_sha and m['split_sha256']==split_sha,'Recipe/split mismatch')
 require(sha(folder/'anchor.joblib')==m['anchor_sha256'] and sha(folder/'residual.joblib')==m['residual_sha256'],'Bundle checksum mismatch')
 a=joblib.load(_repo_io(folder/'anchor.joblib'));r=joblib.load(_repo_io(folder/'residual.joblib'));require(r.anchor_sha256==m['anchor_sha256'],'Residual/anchor binding mismatch');require(list(map(int,a.train))==m['anchor_train'] and list(map(int,r.train))==m['residual_train'],'Bundle train metadata mismatch');require(set(a.train).isdisjoint(r.train),'Bundle training overlap');return PairedBundle(a,r,m)
class PairedBundle:
 def __init__(self,anchor,residual,manifest):self.anchor=anchor;self.residual=residual;self.manifest=manifest
 def predict(self,features,Q):
  require({k:list(v.shape[1:]) for k,v in features.arrays.items()}==self.manifest['feature_schema'] and features.blocks==self.manifest['blocks'] and features.kblocks==self.manifest['kblocks'],'Bundle feature schema mismatch')
  return self.anchor.predict(features,Q)+self.residual.predict(features,Q)
