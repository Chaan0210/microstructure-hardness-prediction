"""Capabilities for development-only, explicitly disjoint label flow."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import numpy as np

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def indices(value):
 a=np.asarray(value,dtype=np.int64);require(a.ndim==1 and len(a)>0 and len(np.unique(a))==len(a),'Invalid/duplicate indices');return a
class Labels:
 """Own only a copied subset: no pointer to source/global target vector."""
 def __init__(self,ids,values):
  self.ids=indices(ids).copy();self.values=np.array(values,dtype=float,copy=True)
  require(self.values.shape==self.ids.shape and np.isfinite(self.values).all(),'Invalid labels')
  self.ids.flags.writeable=False;self.values.flags.writeable=False;self._lookup={int(k):i for i,k in enumerate(self.ids)};self.reads=[]
 def take(self,ids):
  ids=indices(ids);require(set(ids)<=set(self._lookup),'Forbidden label read');self.reads.append(ids.tolist());return self.values[[self._lookup[int(k)] for k in ids]].copy()
 def restrict(self,ids):return Labels(ids,self.take(ids))
class Features:
 def __init__(self,ids,groups,arrays,blocks,kblocks,scope='development400'):
  self.ids=np.array(ids,copy=True);self.groups=np.array(groups,copy=True);require(len(set(self.ids))==len(self.ids),'Duplicate feature IDs')
  require(scope=='synthetic_unit_fixture' or (scope=='development400' and len(self.ids)==400),'Only development400 or explicit unit fixture')
  require(len(groups)==len(ids),'Group length');self.scope=scope;self.arrays={}
  for k,v in arrays.items():
   require(k not in ('y','target','labels'),'Labels in feature store');a=np.array(v,copy=True);require(len(a)==len(ids) and np.isfinite(a).all(),'Invalid features');a.flags.writeable=False;self.arrays[k]=a
  self.blocks=list(blocks);self.kblocks=list(kblocks)
 def take(self,key,ii):return self.arrays[key][indices(ii)]
 def partition(self,parent,train,valid):
  p,t,v=map(indices,(parent,train,valid));require(set(t).isdisjoint(v) and set(t)|set(v)==set(p),'Partition membership overlap/coverage');require(not set(self.groups[t])&set(self.groups[v]),'Partition group overlap')
@dataclass(frozen=True)
class AnchorGraph:
 A:tuple
 offset:int
 cells:tuple
 def validate(self,features):
  covered=[]
  for c in self.cells:
   C,D=indices(c['C']),indices(c['D']);features.partition(self.A,C,D);covered.extend(D);inner=[]
   for e in c['deep']:
    features.partition(C,e['E'],e['F']);inner.extend(e['F'])
   require(sorted(inner)==sorted(C),'Deep OOF coverage')
  require(sorted(covered)==sorted(self.A),'Middle OOF coverage')
  return self

def lock_manifest(path,objects):
 p=Path(path);require(not p.exists(),'Refuse overwrite');p.write_text(json.dumps(objects,indent=2,allow_nan=False)+'\n')
