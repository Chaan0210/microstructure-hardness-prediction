"""Same-anchor CNN-minus-KRR feature and pre-score equivalence/lineage audit."""
import sys,time,datetime,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import bootstrap_selector_144 as s
import numpy as np,torch,joblib
x=s.x;a=s.a;e=s.e;OUT=ROOT/'artifacts/disagreement-156'

def node(d,A,B,V,source,extra,path,create,expected):
 old=e.read(source/'report.json');er=e.read(extra/'report.json');ez=a.readnp(extra/'predictions.npz',er['predictions_sha256']);np.testing.assert_array_equal(ez['train'],A);np.testing.assert_array_equal(ez['validation'],B)
 files=[source/'kernel.joblib',source/'cnn.pt',*[extra/f'seed-{seed}.pt' for seed in x.SEEDS[1:]]];hs=[old['files_sha256']['kernel.joblib'],old['files_sha256']['cnn.pt'],*[v['checkpoint_sha256'] for v in er['models']]]
 for f,h in zip(files,hs):assert e.sha(f)==h,f
 if create:
  path.mkdir(exist_ok=False);obj=joblib.load(files[0]);assert obj['train_ids']==[d['ids'][i] for i in A];m=obj['model'];np.testing.assert_array_equal(m.bound_.data_min_,d['K'][A].min(0));np.testing.assert_array_equal(m.bound_.data_max_,d['K'][A].max(0));assert m.ym_==float(d['y'][A].mean()) and m.ys_==float(d['y'][A].std());kb=m.predict(d['K'][B]);kv=m.predict(d['K'][V]);np.testing.assert_allclose(kb,expected[:,1],atol=1e-8,rtol=0);cb=[];cv=[];norm={'x_mean':float(d['images'][A].mean()),'x_std':float(d['images'][A].std()),'y_mean':float(d['y'][A].mean()),'y_std':float(d['y'][A].std())}
  for si,f in enumerate(files[1:]):
   obj=torch.load(f,map_location='cpu',weights_only=True);assert obj['train_ids']==[d['ids'][i] for i in A] and obj['normalization']==norm;bp=x.predict(obj,d,B,'mps');vp=x.predict(obj,d,V,'mps');np.testing.assert_allclose(bp,ez['seeds'][si],atol=1e-4,rtol=0);cb.append(bp);cv.append(vp)
  cb=np.stack(cb);cv=np.stack(cv);np.testing.assert_allclose(cb.mean(0),expected[:,2],atol=1e-4,rtol=0);assert not set(A)&(set(B)|set(V)) and not set(d['g'][A])&(set(d['g'][B])|set(d['g'][V]));db=cb.mean(0)-kb;dv=cv.mean(0)-kv
  with (path/'features.npz').open('xb') as f:np.savez_compressed(f,anchor=A,block=B,outer=V,kernel_B=kb,kernel_V=kv,cnn_B=cb,cnn_V=cv,delta_B=db,delta_V=dv)
  rec={'anchor_train':A.tolist(),'block':B.tolist(),'outer':V.tolist(),'models':[{'path':str(f.relative_to(ROOT)),'sha256':h} for f,h in zip(files,hs)],'train_IDs_groups_scalers_and_targets_exact':True,'same_A_models_for_B_and_V':True,'feature_sha256':e.sha(path/'features.npz')};e.write(path/'report.json',rec)
 else:
  rec=e.read(path/'report.json');assert rec['anchor_train']==A.tolist() and rec['block']==B.tolist() and rec['outer']==V.tolist();assert rec['models']==[{'path':str(f.relative_to(ROOT)),'sha256':h} for f,h in zip(files,hs)]
 z=a.readnp(path/'features.npz',rec['feature_sha256']);np.testing.assert_array_equal(z['anchor'],A);np.testing.assert_array_equal(z['block'],B);np.testing.assert_array_equal(z['outer'],V);np.testing.assert_array_equal(z['cnn_B'].mean(0)-z['kernel_B'],z['delta_B']);np.testing.assert_array_equal(z['cnn_V'].mean(0)-z['kernel_V'],z['delta_V']);return z,rec

def augmented(d,z):
 dd=dict(d);extra=np.zeros(len(d['ids']));extra[z['block']]=z['delta_B'];extra[z['outer']]=z['delta_V'];dd['X']=np.column_stack([d['X'],extra]);dd['blocks']=list(d['blocks'])+['anchor_disagreement'];return dd

def audit():
 s.check();OUT.mkdir(exist_ok=False);(OUT/'features').mkdir();records=[];sizes={k:[] for k in ['142_143','213_214','320']};hashes={};start=time.perf_counter()
 # Equal .3 weights discard the signed difference: preserving K+C preserves mixture.
 p=np.array([[0.,1.,3.,2.,4.],[0.,2.,2.,2.,4.]]);np.testing.assert_array_equal(a.mixture(p,x.ARM)[0],a.mixture(p,x.ARM)[1]);assert (p[0,2]-p[0,1])!=(p[1,2]-p[1,1]);mixed=float(1-2*np.exp(-1)+np.exp(-2));assert mixed>0.39
 for split in e.SPLITS:
  d=s.data(split);(OUT/'features'/split).mkdir();levels={k:[] for k in sizes}
  for k in range(5):
   dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);V=np.flatnonzero(d['fid']==k);levels['320'].extend((dd['P'][V,2]-dd['P'][V,1]).tolist())
   for j in range(3):
    A,B,P,short,_=dd['cells'][k,j];source=e.cal.OUT/split/f'fold-{k}/inner-{j}';extra=x.OUT/split/f'cnn-fold-{k}/middle-{j}';path=OUT/'features'/split/f'fold-{k}-block-{j}';z,r=node(d,A,B,V,source,extra,path,True,P);levels['213_214'].extend(z['delta_V'].tolist());nuisance=np.column_stack([np.ones(len(B)),a.mixture(P,x.ARM),short]);res=z['delta_B']-nuisance@np.linalg.lstsq(nuisance,z['delta_B'],rcond=None)[0];records.append({'split':split,'outer':k,'block':j,'delta_B_RMS':float(np.sqrt(np.mean(z['delta_B']**2))),'RMS_after_projection_on_intercept_mixture_shortB':float(np.sqrt(np.mean(res**2))),'projection_diagnostic_no_B_labels':True,'feature_report_sha256':e.sha(path/'report.json')});hashes[str((path/'report.json').relative_to(ROOT))]=e.sha(path/'report.json')
    source=e.OUT/split/f'fold-{k}/inner-{j}/deep-0';C,D,P0=a.bundle(source);extra=x.OUT/split/f'cnn-fold-{k}/deep-{j}-0';er=e.read(extra/'report.json');ez=a.readnp(extra/'predictions.npz',er['predictions_sha256']);P0=P0.copy();P0[:,2]=ez['mean3'];z,_=node(d,C,D,V,source,extra,OUT/'features'/split/f'deep-size-{k}-{j}',True,P0);levels['142_143'].extend(z['delta_V'].tolist())
  for size,values in levels.items():sizes[size].append({'split':split,'n_overlapping_query_predictions':len(values),'mean':float(np.mean(values)),'SD':float(np.std(values)),'RMS':float(np.sqrt(np.mean(np.array(values)**2)))})
  e.emit({'audit_split_complete':split,'new_neural_fits':0})
 report={'passed':True,'study':'156 equivalence and same-anchor feature audit before candidate scores','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'linear_equivalence':'An unbounded linear beta*(CNN-KRR) using same-scope branches is a weight shift (+beta CNN,-beta KRR), so it is not a novel linear blend. Exclude this added feature from BOTH Ridge candidates; existing585 Ridge models retained. No new global mixture/affine fit.','not_equivalent_to_current_scalar_calibration':'Equal-weight KRR/CNN mixture loses signed difference. Synthetic two-row witness has identical mixture but different delta. In observed honest B rows, projection on intercept/rawmix/shortB retains nonzero delta variance; this checks representation only, not outcome benefit.','nonlinear_distinction':'RBF on [585features,delta] multiplies a spatial/morphology distance factor by a delta-distance factor (at fixed train-derived gamma); nonzero mixed finite difference demonstrates nonadditivity. It can condition neighbor similarity on model disagreement, unlike a global linear component-weight adjustment. No claim of new independent information beyond original images or guaranteed interactions in fitted results.','nonadditive_RBF_fixture':mixed,'primary_design_after_audit':'Only two existing RBF candidates receive scalar delta. none/ridge1/ridge01 stay exactly585. Two regularization strengths1/.1,512groupbootstrap and fixed3blocks remain. Exactly one primary versus018, no all-feature Ridge fallback.','feature_scope':'A-only KRR and three identityCNN checkpoints produce delta for BOTH B and outerV. InnerI/H subsets ofB share models trainedA which excluded allB andV. No globalOOF feature lookup for training; no fullT query branch replacingA.','training_size_diagnostics':sizes,'records':records,'upstream_audit_sha256':e.sha(ROOT/'reports/nested-selector-146-audit.json'),'source_sha256':e.sha(Path(__file__)),'feature_reports_sha256':hashes,'new_neural_fits':0,'candidate_scores_computed':False,'holdout_test_read':False,'seconds':time.perf_counter()-start,'cost':0};e.write(OUT/'audit.json',report);e.write(ROOT/'reports/disagreement-156-audit.json',report);e.emit({'audit_passed':True,'honest_feature_blocks':45,'nonadditive_fixture':mixed,'remaining_delta_RMS_range':[min(r['RMS_after_projection_on_intercept_mixture_shortB'] for r in records),max(r['RMS_after_projection_on_intercept_mixture_shortB'] for r in records)],'seconds':report['seconds']})
if __name__=='__main__':
 torch.set_num_threads(2);audit()
