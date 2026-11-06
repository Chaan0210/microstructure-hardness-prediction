"""Assemble source-honest A→B and T→V predictors, never full-OOF meta training."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import sys,json
import numpy as np
def main():
 raw=dict(np.load(_repo_io(c.R4.parent/'inputs.npz')));graphs=json.loads((c.R4/'label-graph.json').read_text());base=np.load(_repo_io(c.OLD/'baseline076.npz'));ref=np.load(_repo_io(c.EXP/'highnoise-relational-20261016/inputs.npz'));fixed=json.loads((c.R4/'noise-calibration/summary.json').read_text())['aux']['selection'];sources={};checks=[]
 def use(p):sources[str(p.relative_to(c.ROOT))]=c.sha(p);return np.load(_repo_io(p))
 for k,g in enumerate(graphs):
  A,B,T,V=[np.array(g[z]) for z in ['A','B','T','V']];assert set(A)|set(B)==set(T) and set(A).isdisjoint(B) and set(T).isdisjoint(V);ix=np.flatnonzero(base['fold']==k);assert np.array_equal(base['dev'][ix],V);tpath=c.PREV/f'teachers076/fold{k}-outer0/predictions.npz'
  if not tpath.exists():tpath=c.OLD/f'teachers076/fold{k}-outer0/predictions.npz'
  t=use(tpath);t73=use(c.R4/f'teachers/fold{k}-outer0/predictions.npz');a=use(c.R4/f'anchors/fold{k}.npz');assert np.array_equal(t['train'],A) and np.array_equal(t['query'],B) and np.array_equal(a['A'],A) and np.array_equal(t73['train'],A) and np.array_equal(t73['query'],B)
  p76=.16*a['prediction_B']+.84*t['base'];p73=.16*a['prediction_B']+.84*t73['base'];nobj=raw['X'][:,list(raw['keys']).index('n_obj')];good=np.nan_to_num(nobj,nan=0)>=20;p76[~good[B]]=a['prediction_B'][~good[B]];p73[~good[B]]=a['prediction_B'][~good[B]];mats={}
  for stage,tr,q in [('selector',A,B),('final',T,V)]:
   columns=[]
   for kind in ['old','retarget','pick','pool']:
    preds=[]
    for seed in [0,1]:
     if kind=='old':p=c.R4/f'aux-seed{seed}/fold{k}-{stage}'
     elif kind=='retarget':
      p=c.PREV/f'cnn-retarget/seed{seed}/fold{k}-{stage}'
      if not p.exists():p=c.OLD/f'retarget-cnn/seed{seed}/fold{k}-{stage}'
     else:p=c.PREV/f'cnn-dense-{kind}/seed{seed}/fold{k}-{stage}'
     z=use(p/'predictions.npz');assert np.array_equal(z['train_index'],tr) and np.array_equal(z['validation_index'],q);meta=json.loads((p/'training.json').read_text())['meta'];assert meta['train_ids']==raw['ids'][tr].tolist() and meta['validation_ids']==raw['ids'][q].tolist();sources[str((p/'training.json').relative_to(c.ROOT))]=c.sha(p/'training.json');preds.append(z['validation_prediction'])
    columns.append(np.mean(preds,0))
   p=c.TOPO/f"readouts/fold{k}-{'selector' if stage=='selector' else 'refit'}/predictions.npz";z=use(p);expected=np.r_[B,V] if stage=='selector' else V;assert np.array_equal(z['train'],tr) and np.array_equal(z['query'],expected);delta=z['topology'].mean(0)-z['base'];columns.append(delta[:len(B)] if stage=='selector' else delta);mats[stage]=np.array(columns).T
  beta=fixed[k]['addition']['weights'][1];expected=base['p76'][ix]+(raw['noise'][V]>=12)*good[V]*beta*mats['final'][:,0];assert np.max(abs(expected-ref['base082'][ix]))<1e-10
  c.arrays(f'inputs/fold{k}.npz',A=A,B=B,T=T,V=V,ix=ix,idsB=raw['ids'][B],idsV=raw['ids'][V],yB=raw['y'][B],yV=raw['y'][V],p76B=p76,p73B=p73,p76V=base['p76'][ix],p82V=ref['base082'][ix],RB=mats['selector'],RV=mats['final'],noiseB=raw['noise'][B],noiseV=raw['noise'][V],qualityB=np.c_[raw['noise'][B],np.log1p(np.maximum(np.nan_to_num(nobj[B],nan=0),0))],qualityV=np.c_[raw['noise'][V],np.log1p(np.maximum(np.nan_to_num(nobj[V],nan=0),0))],eligibleB=good[B],eligibleV=good[V],old_beta=np.array(beta));checks.append(dict(fold=k,A=len(A),B=len(B),T=len(T),V=len(V),highB=int(((raw['noise'][B]>=12)&good[B]).sum()),baseline_maxdiff=float(abs(expected-ref['base082'][ix]).max())))
 c.arrays('reference.npz',ids=base['ids'],y=raw['y'][base['dev']],fold=base['fold'],high=(base['noise']>=12)&~base['failed'],base082=ref['base082'],base076=base['p76'])
 c.save(_repo_io('input-provenance.json'),dict(checks=checks,sources=sources,plan_sha256=c.sha(c.HERE/'PLAN.md'),code_sha256=c.sha(__file__),meta_training='A-only predictions onB, never other-foldOOF predictions',families=['oldCNN','076targetCNN','dense-pickCNN','dense-poolCNN','topology-minus-physicalTab'],limitations='Inherited representations and previously inspected labels/family choices; topology uses original evaluated saved models.'));print('PREPARED5FOLDS',checks,flush=True)
if __name__=='__main__':sys.addaudithook(c.guard);main()
