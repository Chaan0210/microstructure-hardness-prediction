"""Complete021-family anchor, recursively honest calibration and residuals.
This module has no global target vector, data reader or import-time execution.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GroupKFold
from contracts import Labels,require,indices
import residual_models as residual
KINDS=tuple(residual.KINDS)
def coefficient(p,y):
 p=np.asarray(p);y=np.asarray(y);z=p-p.mean();require(np.isfinite(p).all() and z@z>1e-12,'Degenerate affine')
 slope=float(z@(y-y.mean())/(z@z));return {'prediction_center':float(p.mean()),'target_center':float(y.mean()),'slope':slope,'intercept':float(y.mean()-slope*p.mean())}
def apply(m,p):return m['target_center']+m['slope']*(np.asarray(p)-m['prediction_center'])
def bootstrap_weights(loss,groups,seed):
 u,inv=np.unique(groups,return_inverse=True);draw=np.random.default_rng(seed).integers(0,len(u),(512,len(u)));counts=np.stack([np.bincount(r,minlength=len(u)) for r in draw]);wins=np.argmin(counts[:,inv]@loss.T,axis=1);return np.bincount(wins,minlength=5)/512,draw,wins
@dataclass
class ResidualBlock:
 models:list
 weights:np.ndarray
 train:tuple
 record:dict
 def predict(self,features,Q):return self.weights@np.stack([residual.predict(m,features.take('X',Q)) for m in self.models])
def fit_block(features,D,r,split_seed,bootstrap_seed):
 D=indices(D);r=np.asarray(r);require(r.shape==D.shape and np.isfinite(r).all(),'Residual shape');ip=np.full((5,len(D)),np.nan);splits=[]
 for ti,vi in GroupKFold(3,shuffle=True,random_state=split_seed).split(D,groups=features.groups[D]):
  U,V=D[ti],D[vi];features.partition(D,U,V);splits.append({'train':U.tolist(),'validation':V.tolist()})
  for j,kind in enumerate(KINDS):
   m=residual.fit(features.take('X',U),r[ti],features.blocks,kind);ip[j,vi]=residual.predict(m,features.take('X',V))
 require(np.isfinite(ip).all(),'Incomplete residual OOF');loss=(ip-r)**2;w,draw,wins=bootstrap_weights(loss,features.groups[D],bootstrap_seed)
 models=[residual.fit(features.take('X',D),r,features.blocks,kind) for kind in KINDS]
 return ResidualBlock(models,w,tuple(D),{'splits':splits,'inner_predictions':ip,'loss':loss,'draws':draw,'winners':wins,'split_seed':split_seed,'bootstrap_seed':bootstrap_seed})
@dataclass
class CompleteAnchor:
 root:object
 short:object
 L:object
 affine:dict
 residuals:list
 train:tuple
 record:dict
 def predict(self,features,Q):
  raw=self.root.predict(features,Q);base=.9*apply(self.affine,raw)+.1*self.short.predict(features,Q)
  corr=np.mean([m.predict(features,Q) for m in self.residuals],axis=0)
  p=.9*(base+corr)+.1*self.L.predict(features,Q);require(np.isfinite(p).all(),'Nonfinite complete anchor');return p

def fit_complete_anchor(features,labels,graph,backend):
 """backend never receives labels outside the particular node's training set."""
 graph.validate(features);A=indices(graph.A);require(set(labels.ids)==set(A),'Anchor label capability must be exactly A')
 root=backend.base(features,labels.restrict(A),graph.offset);short=backend.short(features,labels.restrict(A),graph.offset);L=backend.long(features,labels.restrict(A),graph.offset,'L')
 lookup={int(i):j for j,i in enumerate(A)};oof=np.full(len(A),np.nan);corrections=[];records=[]
 for ell,c in enumerate(graph.cells):
  C,D=indices(c['C']),indices(c['D']);lc=labels.restrict(C);middle=backend.base(features,lc,c['offset']);small=backend.short(features,lc,c['offset']);deep=np.full(len(C),np.nan);lookupC={int(i):j for j,i in enumerate(C)}
  for e in c['deep']:
   E,F=indices(e['E']),indices(e['F']);m=backend.base(features,lc.restrict(E),e['offset']);deep[[lookupC[int(i)] for i in F]]=m.predict(features,F)
  require(np.isfinite(deep).all(),'Incomplete deep OOF');cal=coefficient(deep,lc.take(C));raw=middle.predict(features,D);honest=.9*apply(cal,raw)+.1*small.predict(features,D);r=labels.take(D)-honest
  # All selection is internal to this residual block; no B or outerV labels exist here.
  block=fit_block(features,D,r,20271013+100*graph.offset+ell,20271204+100*graph.offset+ell);corrections.append(block);oof[[lookup[int(i)] for i in D]]=raw
  records.append({'C':C,'D':D,'calibrator':cal,'deep_oof':deep,'anchor_D':honest,'residual_D':r,'offset':c['offset']})
 cal=coefficient(oof,labels.take(A));return CompleteAnchor(root,short,L,cal,corrections,tuple(A),{'cells':records,'raw_oof':oof,'offset':graph.offset,'formula':'.9*(.9*affine(raw)+.1*short+mean(residual3))+.1*L','labels_only_A':True})
