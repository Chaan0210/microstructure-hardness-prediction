"""Independent adapter scope, frozen-control, checkpoint, formula and replay checks."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import dinob_lastblock_083 as e
import numpy as np,torch
fv=e.fv

def main():
 plan=e.check();torch.set_num_threads(2);start=time.perf_counter();results={};count=0;maxreload=0.;maxwarm=0.;maxcpu=0.;replay=None;m=e.backbone.model()
 for split in ('original','20261031','20261101'):
  out=e.OUT/split
  if not (out/'report.json').exists():continue
  d=e.inputs(split);tokens=e.get_cache(d);r=json.loads((out/'report.json').read_text());assert fv.sha(out/'development_predictions.npz')==r['predictions_sha256']
  with np.load(_repo_io(out/'development_predictions.npz'),allow_pickle=False) as z:allp={n:z[n].copy() for n in z.files}
  assert allp['ids'].tolist()==d['ids'];np.testing.assert_array_equal(allp['y'],d['y']);np.testing.assert_array_equal(allp['fold'],d['fid']);np.testing.assert_array_equal(allp['reference010'],d['reference010']);np.testing.assert_array_equal(allp['reference009'],d['reference009']);np.testing.assert_array_equal(allp['frozenB'],d['Bmean']);assembled=np.full(400,np.nan);new=np.full(400,np.nan)
  for k in range(5):
   tr=np.flatnonzero(d['fid']!=k);va=np.flatnonzero(d['fid']==k);path=out/f'fold-{k}';rec=json.loads((path/'report.json').read_text());assert fv.sha(path/'report.json')==r['fold_report_sha256'][f'fold-{k}/report.json'];assert fv.sha(path/'model.pt')==rec['checkpoint_sha256'] and fv.sha(path/'predictions.npz')==rec['predictions_sha256'];assert rec['train_ids']==[d['ids'][i] for i in tr] and rec['validation_ids']==[d['ids'][i] for i in va] and not set(d['g'][tr])&set(d['g'][va]);assert rec['config']==e.CFG and len(rec['curve'])==8
   with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:pp={n:z[n].copy() for n in z.files}
   np.testing.assert_array_equal(pp['train_indices'],tr);np.testing.assert_array_equal(pp['validation_indices'],va);states,old=e.head_states(d,k);np.testing.assert_array_equal(pp['frozen'],old[va]);t=e.Tail(m,states,e.CFG['seed']+k);assert e.h.state_hash(t.state_dict())==rec['initial_state_sha256'];assert e.frozen_hash(t)==rec['frozen_state_sha256'];trainable=[n for n,p in t.named_parameters() if p.requires_grad];assert set(trainable)=={n for n,p in t.named_parameters() if n.startswith('block.')};assert sum(p.numel() for p in t.parameters() if p.requires_grad)==sum(p.numel() for p in m.blocks[-1].parameters())
   a=torch.load(_repo_io(path/'model.pt'),weights_only=True,map_location='cpu');assert a['config']==e.CFG and a['train_ids']==rec['train_ids'] and a['validation_ids']==rec['validation_ids'];assert a['target_mean']==states[0]['target_mean'] and a['target_std']==states[0]['target_std'];assert e.h.state_hash(a['state'])==rec['final_state_sha256'];initial=t.state_dict()
   for n,v in initial.items():
    if n not in trainable:assert torch.equal(v,a['state'][n]),n
   assert rec['block_parameter_delta_L2']>0;orders=np.random.default_rng(e.CFG['seed']+k).permuted(np.tile(np.arange(len(tr)),(8,1)),axis=1);assert hashlib.sha256(orders.tobytes()).hexdigest()==rec['batch_order_sha256'];t=t.to('mps');warm=e.predict(t,d,tokens,va)*a['target_std']+a['target_mean'];we=float(abs(warm-old[va]).max());assert we<.003;maxwarm=max(maxwarm,we);np.testing.assert_array_equal(warm,pp['warm']);t.load_state_dict(a['state']);p=e.predict(t,d,tokens,va)*a['target_std']+a['target_mean'];err=float(abs(p-pp['adapted']).max());assert err<1e-5;maxreload=max(maxreload,err);np.testing.assert_array_equal(pp['adapted'],allp['adaptedB'][va]);assembled[va]=p
   # Same saved model evaluated on CPU for two heldout rows per fold.
   t=t.cpu();ii=va[:2]
   with torch.no_grad():pc=t(torch.tensor(np.array(tokens[ii],copy=True)),torch.tensor(d['B'][ii,:4608],dtype=torch.float32)).numpy().astype(float)*a['target_std']+a['target_mean']
   ce=float(abs(pc-p[:2]).max());assert ce<.003;maxcpu=max(maxcpu,ce);del t;torch.mps.empty_cache();fr=json.loads((e.previous.base.OUT/split/f'fold-{k}/report.json').read_text());assert fr['slope']==r['unchanged009_slopes'][k];new[va]=d['reference009'][va]+.125*fr['slope']*(pp['adapted']-d['Bmean'][va]);np.testing.assert_array_equal(new[va],allp['primary'][va]);count+=1
   if split=='original' and k==0:
    dd=dict(d);dd['y']=d['y'].copy();dd['y'][va]=10000+np.arange(len(va));rp,rr=e.fit_fold(dd,tokens,m,k,e.OUT/'validation-replay-original');re=float(abs(rp-pp['adapted']).max());assert re<1e-4;assert rr['final_state_sha256']==rec['final_state_sha256'];replay={'outer_validation_labels_perturbed_in_memory':True,'full8epoch_lastblock_replay_max_error':re,'lastblock_and_all_frozen_states_exact':True}
  np.testing.assert_allclose(assembled,allp['adaptedB'],atol=1e-5,rtol=0);assert e.prior.old.comparison(d['y'],d['reference009'],new,d['fid'])==r['comparison']
  for n,s in r['scores'].items():assert s=={'rmse':fv.rmse(d['y'],allp[n]),'fold_rmse':[fv.rmse(d['y'][d['fid']==k],allp[n][d['fid']==k]) for k in range(5)]}
  assert e.prior.old.comparison(d['y'],d['reference010'],new,d['fid'])==r['comparison_vs010'];results[split]={'scores':r['scores'],'comparison':r['comparison'],'comparison_vs010':r['comparison_vs010'],'report_sha256':fv.sha(out/'report.json'),'predictions_sha256':r['predictions_sha256']}
 # Independently rebuild three cached prefix rows using unchanged official model.
 d=e.inputs('original');tokens=e.get_cache(d);m=m.to('mps');prefixerror=0.
 for i in (0,199,399):
  z=e.prefix(m,torch.tensor(e.backbone.ds.inputs(d['ids'][i],'full504'),device='mps')).cpu().numpy()[0];prefixerror=max(prefixerror,float(abs(z-tokens[i]).max()));np.testing.assert_array_equal(z,tokens[i])
 hashes=json.loads((_repo_path(ROOT, 'artifacts/dinob-062/extraction.json')).read_text())['image_sha256'];assert set(hashes)==set(d['ids'])
 for i,digest in hashes.items():assert fv.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))==digest
 selection=json.loads((e.OUT/'selection.json').read_text());assert selection['advance']==(results['original']['comparison']['gain']>0 and results['original']['comparison_vs010']['gain']>0);e.check();report={'passed':True,'results':results,'folds_verified':count,'frozen_heads_verified':3*count,'trainable_lastblock_parameters':sum(p.numel() for p in m.blocks[-1].parameters()),'only_lastblock_changed':True,'max_MPS_reload_error':maxreload,'max_frozen_control_error':maxwarm,'max_CPU_MPS_error':maxcpu,'prefix_rows_independently_rebuilt':3,'prefix_max_error':prefixerror,'development400_image_hashes_verified':True,'leakage_replay':replay,'unchanged009_formula_and_coefficients':True,'eligible_for_candidate':len(results)==3 and all(r['comparison']['gain']>0 and r['comparison_vs010']['gain']>0 for r in results.values()),'best009_sha256':fv.sha(_repo_path(ROOT, 'artifacts/submission-candidate-009.csv')),'protected010_files_verified':len(plan['protected010']),'holdout_test_selection_access':False,'cost':0,'seconds':time.perf_counter()-start,'source_sha256':fv.sha(Path(__file__)),'limits':plan['bias']};fv.write(_repo_path(ROOT, 'reports/dinob-lastblock-083-validation.json'),report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
