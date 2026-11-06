from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import math,json,hashlib,csv,statistics
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=_RepoPath(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def mean(x):return math.fsum(float(v) for v in x)/len(x)
def rmse(y,p):return math.sqrt(mean([(float(a)-float(b))**2 for a,b in zip(y,p)]))
def corr(x,y):
 xm=mean(x);ym=mean(y);return math.fsum((a-xm)*(b-ym) for a,b in zip(x,y))/math.sqrt(math.fsum((a-xm)**2 for a in x)*math.fsum((b-ym)**2 for b in y))
p=load(_repo_path(R, 'predictions.npz'));a=load(_repo_path(R, 'attention.npz'));d=read(_repo_path(R, 'report.json'));ref=load(_repo_path(P, 'experiments/query-normalization-197/OOF.npz'));np.testing.assert_array_equal(p['ids'],a['ids']);np.testing.assert_array_equal(p['ids'],ref['ids']);y=ref['y'];b=ref['best021'];components={}
for s in ['196','197']:
 n=p[s+'_normal'];u=p[s+'_uniform'];delta=u-n;st=d['label_free'][s]['delta_uniform_minus_original'];assert abs(math.sqrt(mean(delta**2))-st['RMS'])<1e-12;assert abs(mean(delta)-st['mean'])<1e-12;assert abs(mean(abs(delta))-st['mean_absolute'])<1e-12;assert abs(statistics.median(delta)-st['q0_5_25_50_75_95_100'][3])<1e-12
 r=d['performance_diagnostic_only'][s]
 for arm,x in [('normal',n),('uniform',u)]:
  assert abs(rmse(y,x)-r[arm+'_RMSE'])<1e-12;assert abs(rmse(y,.9*b+.1*x)-r[arm+'_blend_RMSE'])<1e-12
  ess=a[s+'_'+arm+'_cross_ESS'];KL=a[s+'_'+arm+'_cross_KL_uniform'];H=a[s+'_'+arm+'_cross_entropy'];assert ess.shape==(400,4,4) and ess.min()>=1-1e-8 and ess.max()<=1296+1e-8;np.testing.assert_allclose(KL+H,math.log(1296),atol=1e-12,rtol=0);weights=a[s+'_'+arm+'_self_weights'];assert weights.shape==(400,4,4,4);np.testing.assert_allclose(weights.sum(-1),1.,atol=1e-12,rtol=0)
  if arm=='uniform':np.testing.assert_allclose(ess,1296,atol=1e-8,rtol=0);assert abs(a[s+'_'+arm+'_cross_TV_uniform']).max()<1e-12
 for q in r['by_fold']:
  ix=p['fold']==q['fold'];assert sum(ix)==80
  assert abs(rmse(y[ix],u[ix])-rmse(y[ix],n[ix])-q['uniform_minus_normal_RMSE'])<1e-12
 eb=b-y;en=n-y;eu=u-y;term_component=.01*(mean(eu**2)-mean(en**2));term_cross=.18*mean(eb*(eu-en));direct=mean((.9*eb+.1*eu)**2)-mean((.9*eb+.1*en)**2);assert abs(term_component+term_cross-direct)<1e-10
 components[s]={'fixed_weights':[.9,.1],'component_MSE_change_contribution':term_component,'cross_error_product_change_contribution':term_cross,'blend_MSE_change':direct,'normal_component_error_correlation_with021':corr(en,eb),'uniform_component_error_correlation_with021':corr(eu,eb),'note':'Exact descriptive algebra on original400; nooptimizedweight orPublicscore usage. NegativeblendMSEchangeisdiagnostic,notselection.'}
cert=read(_repo_path(R, 'inference-validation.json'));assert cert['passed'] and len(cert['all_records'])==10
for row in cert['all_records']:assert row['original_MPS_saved_prediction_exact'] and row['state_and_RNG_both_backends_unchanged'] and row['self_algorithm_parameters_unchanged'] and row['normal_after_intervention_repeat_exact'] and row['uniform_repeat_exact']
for stage in ['infer','analyze']:assert read(_repo_path(R, stage+'-exit.json'))['exit_code']==0
r={'passed':True,'independent_fsum_RMSE_delta_and_all_fold_metrics':True,'attention_shape_range_probability_and_entropy_identity':True,'original_reference_OOF_exact':True,'CPU_MPS_and_independent_float64_certificates_verified':True,'mixture_decomposition':components,'new_model_fit_or_inference_in_validator':False,'new_model_selection':False,'report_SHA256':sha(_repo_path(R, 'report.json')),'predictions_SHA256':sha(_repo_path(R, 'predictions.npz')),'source_SHA256':sha(Path(__file__))}
with (_repo_path(R, 'validation.json')).open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
print(json.dumps(r),flush=True)
