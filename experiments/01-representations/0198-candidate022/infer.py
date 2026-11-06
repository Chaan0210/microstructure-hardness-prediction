from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import csv
import common as c
from common import R,P,np,torch
assert c.read(R/'training-validation.json')['passed'];assert c.read(R/'validate-exit.json')['exit_code']==0;c.check();cert=c.read(R/'training-validation.json');assert c.sha(R/'true/model.pt')==cert['primary_model_SHA256']
# No backward, optimizer step, preprocessing fit or sample hardness is used here.
torch.Tensor.backward=c.blocked;torch.optim.AdamW.step=c.blocked
from sklearn.preprocessing import StandardScaler,MinMaxScaler
StandardScaler.fit=c.blocked;MinMaxScaler.fit=c.blocked

def table(path):
 with path.open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['ID','hardness'];rows=list(reader)
 return np.array([q['ID'] for q in rows]),np.array([float(q['hardness']) for q in rows])
def write_csv(path,ids,p):
 with path.open('x',newline='') as f:w=csv.writer(f);w.writerow(['ID','hardness']);w.writerows(zip(ids,p))
def stats(p):return {'min':float(p.min()),'max':float(p.max()),'mean':float(p.mean()),'std':float(p.std())}
with c.Monitor('inference',300):
 meta=c.read(c.OLD/'test-run/test-feature-validation.json');file=c.OLD/'test-run/test-features.npz';assert meta['passed'] and meta['rows']==1000 and c.sha(file)==meta['feature_sha256'];d=c.load(file,['ids','images']);ids=d['ids'];assert ids.tolist()==[f'TEST_{j:06d}' for j in range(1,1001)] and len(set(ids))==1000 and d['images'].shape==(1000,1,128,128) and d['images'].dtype==np.float32 and np.isfinite(d['images']).all()
 with (P/'data/sample_submission.csv').open(newline='',encoding='utf-8-sig') as f:
  reader=csv.reader(f);assert next(reader)==['ID','hardness'];sample_ids=[row[0] for row in reader]
 assert sample_ids==ids.tolist();fresh=c.raw_images(ids,'test',meta);np.testing.assert_array_equal(fresh,d['images']);del fresh
 rid,ref=table(P/'artifacts/submission-candidate-021.csv');np.testing.assert_array_equal(rid,ids);assert np.isfinite(ref).all();old=torch.load(_repo_io(c.OLD/'models/CNN-20261003.pt'),map_location='cpu',weights_only=True);models={n:torch.load(_repo_io(R/n/'model.pt'),map_location='cpu',weights_only=True) for n in ['detach','true','true-repeat']};pp={}
 for name,obj in [('old',old),*models.items()]:
  pm=c.predict(obj,d['images'],'mps');path=c.OLD/'models/CNN-20261003.pt' if name=='old' else R/name/'model.pt';repeat=c.predict(torch.load(_repo_io(path),map_location='cpu',weights_only=True),d['images'],'mps');pc=c.predict(obj,d['images'],'cpu');np.testing.assert_array_equal(pm,repeat);diff=float(abs(pm-pc).max());assert diff<.001;pp[name]={'mps':pm,'cpu':pc};c.emit({'test_reload':name,'MPS_repeat_exact':True,'CPU_MPS_max_abs':diff})
 np.testing.assert_array_equal(pp['detach']['mps'],pp['old']['mps']);np.testing.assert_array_equal(pp['true']['mps'],pp['true-repeat']['mps']);np.testing.assert_array_equal(pp['true']['cpu'],pp['true-repeat']['cpu'])
 old018=c.load(P/'experiments/candidate021-178/final-inference/018-mps-replay/predictions.npz');old021=c.load(P/'experiments/candidate021-178/final-inference/predictions.npz');np.testing.assert_array_equal(old018['ids'],ids);np.testing.assert_array_equal(old021['ids'],ids);np.testing.assert_array_equal(old018['CNN_seeds'][0],pp['old']['mps']);np.testing.assert_array_equal(old018['primary'],old021['reference018']);np.testing.assert_array_equal(old021['primary'],ref);np.testing.assert_array_equal(.9*old018['primary']+.1*old021['L_CPU_seeds'].mean(0),ref)
 cal=c.read(c.OLD/'models/calibrator.json');coefficient=.081*cal['slope'];assert coefficient==c.plan['coefficient'];q=ref+coefficient*(pp['true']['mps']-pp['old']['mps']);repeat=ref+coefficient*(pp['true-repeat']['mps']-pp['detach']['mps']);np.testing.assert_array_equal(q,repeat);np.testing.assert_array_equal(ref+coefficient*(pp['detach']['mps']-pp['old']['mps']),ref)
 # Independent full-ensemble reconstruction changes exactly one of the3CNN seeds.
 weights=np.array([0.,.1875,.1875,.125,.125]);weights/=weights.sum();baseP=old018['P'].copy();np.testing.assert_array_equal(baseP[:,2],old018['CNN_seeds'].mean(0));np.testing.assert_array_equal(baseP@weights,old018['raw']);base=.9*(cal['target_center']+cal['slope']*(old018['raw']-cal['prediction_center']))+.1*old018['short'];np.testing.assert_array_equal(base,old018['base']);np.testing.assert_array_equal(base+old018['corrections'].mean(0),old018['primary']);changed=old018['CNN_seeds'].copy();changed[0]=pp['true']['mps'];baseP[:,2]=changed.mean(0);raw=baseP@weights;alt=.9*(.9*(cal['target_center']+cal['slope']*(raw-cal['prediction_center']))+.1*old018['short']+old018['corrections'].mean(0))+.1*old021['L_CPU_seeds'].mean(0);formula_error=float(abs(q-alt).max());assert formula_error<1e-10
 cpu=ref+coefficient*(pp['true']['cpu']-pp['old']['cpu']);cpuerr=float(abs(cpu-q).max());assert cpuerr<.001;assert q.shape==(1000,) and np.isfinite(q).all();lo,hi=cert['target_range_for_fixed_format_guard'];span=hi-lo;assert q.min()>0 and q.min()>=lo-span and q.max()<=hi+span
 out=R/'inference';out.mkdir();csvfile=out/'submission-candidate-022.csv';write_csv(csvfile,ids,q);write_csv(out/'repeated-022.csv',ids,repeat);assert c.sha(csvfile)==c.sha(out/'repeated-022.csv');ri,rp=table(csvfile);np.testing.assert_array_equal(ri,ids);np.testing.assert_array_equal(rp,q);assert len(csvfile.read_bytes().splitlines())==1001
 c.arrays(out/'predictions.npz',ids=ids,reference021=ref,old_CNN_MPS=pp['old']['mps'],new_CNN_MPS=pp['true']['mps'],old_CNN_CPU=pp['old']['cpu'],new_CNN_CPU=pp['true']['cpu'],primary=q,CPU=cpu,independent_full_ensemble=alt)
 rec={'passed':True,'rows':1000,'columns':['ID','hardness'],'ID_order_exact_sample':True,'ID_unique':True,'finite':True,'CSV_roundtrip_exact':True,'repeated_model_CSV_byte_exact':True,'raw_test1000_hash_and_BOX_cache_exact':True,'test_sample_hardness_parsed':False,'test_labels_read':False,'test_fitting_tuning_or_selection':False,'fixed_formula':c.plan['formula'],'coefficient':coefficient,'legacy_float32_denormalization_batch32':True,'first_old_CNN_fresh_reload_exact_cached021_component':True,'detach_replacement_best021_exact':True,'independent_full_ensemble_max_abs':formula_error,'CPU_MPS_composite_max_abs':cpuerr,'CPU_MPS_CNN_max_abs':{n:float(abs(p['mps']-p['cpu']).max()) for n,p in pp.items()},'prediction_range':stats(q),'delta_vs_best021':{'max_abs':float(abs(q-ref).max()),'RMS':float(np.sqrt(np.mean((q-ref)**2))),'mean':float((q-ref).mean())},'training_target_range':[lo,hi],'below_train_min':int((q<lo).sum()),'above_train_max':int((q>hi).sum()),'fixed_range_guard_passed':True,'clipping':False,'CSV_SHA256':c.sha(csvfile),'predictions_SHA256':c.sha(out/'predictions.npz'),'test_cache_SHA256':meta['feature_sha256'],'training_validation_SHA256':c.sha(R/'training-validation.json'),'network_install_download_payment_upload':False}
c.check();c.write(R/'inference-validation.json',rec);c.emit(rec)
