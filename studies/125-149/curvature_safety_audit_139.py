"""Read-only fitted-curve audit; no test features/outputs, fitting, or model changes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import quadratic_calibration_137 as q
import numpy as np
e=q.e;OUT=_repo_path(ROOT, 'artifacts/curvature-safety-audit-139')

def analyze(name,m,train,query):
 assert np.isfinite(train).all() and np.isfinite(query).all();assert m['n_train']==len(train);mu=float(np.mean(train));sd=float(np.std(train));np.testing.assert_allclose([mu,sd],[m['prediction_center'],m['prediction_scale']],atol=1e-12,rtol=0)
 z=(train-mu)/sd;v=(query-mu)/sd;beta=0. if m['degenerate'] else m['curvature']/m['quadratic_scale'];lin=m['linear_slope_in_standardized_p'];proj=m['square_projection'];sq=m['square_mean'];A=beta;B=lin-beta*proj;C=m['target_center']-beta*sq
 # Independent polynomial evaluation of exactly the existing stored mapping.
 direct=A*v*v+B*v+C;pred=q.predict(m,query);np.testing.assert_allclose(direct,pred,atol=1e-10,rtol=0)
 Q=np.zeros_like(z) if m['degenerate'] else (z*z-sq-proj*z)/m['quadratic_scale'];D=np.column_stack([np.ones(len(z)),z,Q]);H=D.T@D/len(z)+np.diag([0.,0.,1.]);eig=np.linalg.eigvalsh(H)
 def interval(lo,hi):
  zz=np.array([(lo-mu)/sd,(hi-mu)/sd]);points=zz.tolist()
  if A!=0:
   zv=-B/(2*A)
   if zz[0]<=zv<=zz[1]:points.append(zv)
  vals=np.array([A*t*t+B*t+C for t in points]);der=(2*A*zz+B)/sd
  return {'input_range':[float(lo),float(hi)],'calibrated_exact_range':[float(vals.min()),float(vals.max())],'derivative_range':[float(der.min()),float(der.max())],'strictly_increasing':bool(der.min()>0),'negative_output_possible_in_interval':bool(vals.min()<0)}
 lo,hi=float(train.min()),float(train.max());ql,qh=float(query.min()),float(query.max());cor=m['curvature']*((v*v-sq-proj*v)/m['quadratic_scale']) if not m['degenerate'] else np.zeros_like(v);outside=(query<lo)|(query>hi);distance=np.maximum.reduce([lo-query,query-hi,np.zeros_like(query)])/sd
 vertex=None
 if A!=0:
  zv=-B/(2*A);vertex={'raw_input':float(mu+sd*zv),'calibrated_value':float(C-B*B/(4*A)),'is_global_minimum':bool(A>0)}
 roots=np.roots([A,B,C] if A else [B,C]);realroots=[float(mu+sd*t.real) for t in roots if abs(t.imag)<1e-9]
 return {'name':name,'train_rows':len(train),'query_rows':len(query),'stored_prediction_range_exact':m['train_pred_range']==[lo,hi],'prediction_scale':sd,'quadratic_scale':m['quadratic_scale'],'curvature':m['curvature'],'quadratic_coefficient_in_z':A,'linear_coefficient_in_z':B,'constant_in_z':C,'train_orthogonality_max_abs':float(max(abs(Q.mean()),abs(np.mean(Q*z)))),'penalized_normal_matrix_eigenvalues':eig.tolist(),'condition_number':float(eig[-1]/eig[0]),'train_interval':interval(lo,hi),'observed_query_interval':interval(ql,qh),'combined_train_query_interval':interval(min(lo,ql),max(hi,qh)),'query_outside_train_range':int(outside.sum()),'max_extrapolation_distance_in_train_std':float(distance.max()),'query_curvature_correction_before_point9_range':[float(cor.min()),float(cor.max())],'max_abs_query_curvature_contribution_to_base':float(.9*np.max(abs(cor))),'query_prediction_max_abs_error_independent_polynomial':float(abs(pred-direct).max()),'vertex':vertex,'real_zero_crossing_raw_inputs':realroots,'globally_nonnegative_calibration':bool(A>0 and vertex['calibrated_value']>=0) if A else False,'note':'Interval statements concern this calibrator only, not shortB/physics residual terms or arbitrary unseen inputs.'}

def main():
 q.check();OUT.mkdir(exist_ok=False);protected={f'artifacts/submission-candidate-{c}.csv':e.sha(_repo_path(ROOT, f'artifacts/submission-candidate-{c}.csv')) for c in ['015','016']};reports=[]
 for split in e.SPLITS:
  d=q.data(split)
  for k in range(5):
   dd,_=q.x.replace(d,split,k,q.x.OUT/split/f'cnn-fold-{k}',False);fr=e.read(q.OUT/split/f'fold-{k}/report.json');V=np.flatnonzero(d['fid']==k);reports.append(analyze(f'{split}/outer{k}',fr['outer_model'],q.a.mixture(dd['middleP'][k],q.x.ARM),q.a.mixture(dd['P'][V],q.x.ARM)))
   for j in range(3):
    _,_,P,_,_=dd['cells'][k,j];reports.append(analyze(f'{split}/outer{k}/anchor{j}',fr['anchor_models'][j],q.a.mixture(dd['deepP'][k,j],q.x.ARM),q.a.mixture(P,q.x.ARM)))
 # Production audit reads only fitted calibrator and its training OOF arrays: no full500 labels or test arrays.
 prod=_repo_path(ROOT, 'artifacts/final-model-138');cal=e.read(prod/'calibrator.json')
 with np.load(_repo_io(prod/'calibration_training.npz'),allow_pickle=False) as z:p=z['inner_oof'].copy()
 full=analyze('full500/main',cal,p,p)
 for j in range(3):
  rr=e.read(prod/f'block-{j}/report.json')
  with np.load(_repo_io(prod/f'block-{j}/anchor.npz'),allow_pickle=False) as z:D=z['deep_components'].copy();M=z['middle_components'].copy()
  reports.append(analyze(f'full500/anchor{j}',rr['calibrator'],q.a.mixture(D,q.x.ARM),q.a.mixture(M,q.x.ARM)))
 reports.append(full)
 existing=e.read(_repo_path(ROOT, 'reports/candidate016-verification.json'));assert existing['valid'] and existing['CSV_SHA256']==protected['artifacts/submission-candidate-016.csv'];min_der=min(z['combined_train_query_interval']['derivative_range'][0] for z in reports);min_scale=min(z['quadratic_scale'] for z in reports);max_condition=max(z['condition_number'] for z in reports);negative=[z['name'] for z in reports if z['combined_train_query_interval']['negative_output_possible_in_interval']];nonmonotone=[z['name'] for z in reports if not z['combined_train_query_interval']['strictly_increasing']]
 record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'calibrators_checked':len(reports),'new_fits':0,'test_predictions_or_features_decoded':False,'full500_labels_decoded':False,'all_training_moments_and_stored_ranges_checked':True,'min_train_prediction_std':min(z['prediction_scale'] for z in reports),'min_train_quadratic_RMS':min_scale,'max_penalized_normal_matrix_condition_number':max_condition,'min_derivative_over_each_observed_train_query_interval':min_der,'negative_intervals':negative,'nonmonotone_intervals':nonmonotone,'max_extrapolation_distance_train_std':max(z['max_extrapolation_distance_in_train_std'] for z in reports),'max_abs_curvature_base_contribution_observed':max(z['max_abs_query_curvature_contribution_to_base'] for z in reports),'development_outer_queries_outside_meta_training_range':sum(z['query_outside_train_range'] for z in reports if z['name'].count('/')==1 and not z['name'].startswith('full500')),'production_main':full,'all_curves':reports,'mathematical_definition':'q normalization and projection are fit on current training OOF only. Prediction uses the same polynomial at every input, including outside training range. train_pred_range is metadata, not clipping/gating. Penaltylambda1 shrinks the normalized orthogonal quadratic coefficient byhalf but does not bound extrapolation.','comparison_with109':'109 had tiny per-cell feature scales producing enormous standardized rare features. Here observed64 prediction/q scales and penalized-system conditioning are measured explicitly. Finite outputs alone are not the safety argument.','unconditional_limit':'Quadratics grow as p²; lambda1 does not guarantee bounded output, monotonicity for all real inputs, or a globally nonnegative complete ensemble. Fold curves may have negative curvature. No claim of zero extrapolation risk.','existing016_test_verification_used_only_as_prior_evidence':{'path':'reports/candidate016-verification.json','sha256':e.sha(_repo_path(ROOT, 'reports/candidate016-verification.json')),'finite':existing['finite'],'prediction_range':existing['prediction_range'],'outside_training_label_range':existing['outside_training_range'],'CSV_byte_exact':existing['CSV_byte_exact']},'decision':{'numerical_or_scope_bug_found':bool(negative or nonmonotone or min_scale<1e-6 or max_condition>10),'change_model_or_rule':False,'hold_submission_for_bug':bool(negative or nonmonotone or min_scale<1e-6 or max_condition>10),'note':'Flags are audit warnings, not performance/model-selection rules. Any flag needs concrete investigation before calling it a model bug. No test-selected threshold or repair.'},'protected':protected,'cost':0,'uploaded':False,'source_sha256':e.sha(Path(__file__))}
 for name,h in protected.items():assert e.sha(_repo_path(ROOT, name))==h
 e.write(_repo_path(OUT, 'report.json'),record);e.write(_repo_path(ROOT, 'reports/curvature-safety-audit-139-summary.json'),record);e.emit({k:record[k] for k in ['calibrators_checked','min_train_prediction_std','min_train_quadratic_RMS','max_penalized_normal_matrix_condition_number','min_derivative_over_each_observed_train_query_interval','negative_intervals','nonmonotone_intervals','max_extrapolation_distance_train_std','max_abs_curvature_base_contribution_observed','decision']})
if __name__=='__main__':main()
