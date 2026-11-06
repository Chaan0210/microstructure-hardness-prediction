from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();assert torch.backends.mps.is_available();records=[]
with c.Monitor('baseline',900):
 for split in c.SPLITS:
  d=c.data(split);root=R/split;root.mkdir();f=c.lineage(d);c.arrays(root/'frozen-components.npz',**f);pred=np.full(400,np.nan)
  for k in range(5):
   T,V=c.indices(d,k);old=torch.load(_repo_io(c.old_checkpoint(split,k)),map_location='cpu',weights_only=True);norm,*_=c.preprocessing(d,T,k);assert old['seed']==20261003+k and old['train_ids']==d['ids'][T].tolist() and old['validation_ids']==d['ids'][V].tolist() and old['normalization']==norm and old['epochs']==80 and old['average_last_epochs']==5
   before=c.legacy_predict(old,d['images'],V,'mps');np.testing.assert_array_equal(before,d['old_CNN'][V]);folder=root/f'fold-{k}-detach';e.fit(d,k,'detach',folder);new=torch.load(_repo_io(folder/'model.pt'),map_location='cpu',weights_only=True);base={n:v for n,v in new['state_dict'].items() if not n.startswith('aux.')};assert base.keys()==old['state_dict'].keys() and all(torch.equal(v,old['state_dict'][n]) for n,v in base.items());pp=c.canonical_predictions(d,k,folder);np.testing.assert_array_equal(pp,d['old_CNN'][V]);pred[V]=pp;rec={'split':split,'fold':k,'historical_checkpoint_seed_trainIDs_norm_verified':True,'historical_MPS_cached_OOF_exact_before_fit':True,'detach_fresh_weights_exact_historical':True,'detach_canonical_OOF_exact_historical':True,'historical_model_SHA256':c.sha(c.old_checkpoint(split,k)),'new_model_SHA256':c.sha(folder/'model.pt')};c.write(folder/'legacy-equivalence.json',rec);records.append(rec);c.emit({'detach_verified':split,'fold':k,'completed_detach_fits':len(records),'weights_and_OOF_exact':True});torch.mps.empty_cache()
  np.testing.assert_array_equal(pred,d['old_CNN']);c.arrays(root/'detach-OOF.npz',ids=d['ids'],fold=d['fold'],prediction=pred)
assert len(records)==10;c.check();c.write(R/'baseline-validation.json',{'passed':True,'models':10,'records':records,'all10_detach_weight_and_OOF_exact':True,'no_true_permuted_comparison_before_null_certification':True});c.emit({'all10_detach_complete_and_exact':True})
