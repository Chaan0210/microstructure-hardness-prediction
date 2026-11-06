from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import numpy as np,joblib
import readout as r
f=r.f;R=f.R;P=f.P

def main(split):
 d=r.data(split);out=R/split;rec=f.read(out/'report.json');z=r.load(out/'predictions.npz');assert f.sha(out/'predictions.npz')==rec['predictions_sha256'];count=0;errors=[];reconstructed={a:np.full((3,400),np.nan) for a in ['B','AMP','WPH']}
 for name,hh in rec['block_report_hashes'].items():
  file=out/name;assert f.sha(file)==hh;br=f.read(file);arm=br['arm'];k=br['fold'];j=br['block'];root=file.parent;v=r.load(root/'predictions.npz');assert f.sha(root/'predictions.npz')==br['predictions_sha256'];B=v['B'];V=v['V'];inner=np.full_like(v['inner'],np.nan);full=[];index={int(i):t for t,i in enumerate(B)}
  for mf,mh in br['models_sha256'].items():
   assert f.sha(root/mf)==mh;o=joblib.load(_repo_io(root/mf));m=o['model'];I=o['train_indices'];H=o['validation_indices'];target=np.array([v['residual'][index[int(i)]] for i in I]);p=r.n.predict(m,d['X'][arm][H]);q=r.manual(m,d['X'][arm][I],target,d['X'][arm][H],d['blocks'][arm]);errors.append(float(abs(p-q).max()));count+=1
   assert not set(I)&set(H) and not set(d['g'][I])&set(d['g'][H]);ci=r.KINDS.index(m['kind'])
   if mf.startswith('inner'):inner[ci,[index[int(i)] for i in H]]=p
   else:full.append((ci,p))
  full=np.stack([p for _,p in sorted(full,key=lambda v:v[0])]);np.testing.assert_array_equal(inner,v['inner']);np.testing.assert_array_equal(full,v['full']);ww,dd,win=r.weights((inner-v['residual'])**2,d['g'][B],br['seed']);np.testing.assert_array_equal(ww,v['weights']);np.testing.assert_array_equal(dd,v['draws']);np.testing.assert_array_equal(win,v['winners']);pred=ww@full;np.testing.assert_array_equal(pred,v['primary']);reconstructed[arm][j,V]=pred
 for arm in reconstructed:np.testing.assert_array_equal(reconstructed[arm],z[arm+'_corrections'])
 for k in range(5):
  base=r.load(P/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/predictions.npz');V=base['outer_indices']
  for arm in ['B','AMP','WPH']:
   pred=base['base']+reconstructed[arm][:,V].mean(0)
   if arm!='B':pred=.9*pred+.1*d['globalL'][V]
   np.testing.assert_array_equal(pred,z['B_replay' if arm=='B' else arm][V])
 for key in ['ids','y','g']:np.testing.assert_array_equal(z[key],d[key])
 np.testing.assert_array_equal(z['fold'],d['fid']);np.testing.assert_array_equal(z['B_replay'],d['018']);np.testing.assert_array_equal(.9*z['B_replay']+.1*d['globalL'],d['021']);assert max(errors)<2e-9
 report={'passed':True,'second_process_main_model_reloads':count,'all_inner_full_bootstrap_predictions_exact':True,'independent_solver_max_error':max(errors),'all_train_scalers_weights_and_group_scopes_checked':True,'all_final_predictions_exact':True,'ID_target_group_fold_order_exact':True,'protected_unchanged':len(f.check()['protected']),'plan_sha256':f.sha(R/'plan.json'),'prediction_sha256':f.sha(out/'predictions.npz')};f.write(out/'validation.json',report);f.emit(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('split',choices=r.SPLITS);main(p.parse_args().split)
