import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import np,O,SPLITS
c.check();s=c.J(O/'summary.json');z=c.Z(O/'ensemble-OOF.npz');d=c.Z(O/'bootstrap-draws.npz')['draws'];saved=c.Z(O/'bootstrap-results.npz');np.testing.assert_array_equal(d,np.random.default_rng(202610204).integers(0,400,(10000,400),dtype=np.int64));counts=np.stack([np.bincount(row,minlength=400) for row in d]);errors={a:(z[a]-z['y'])**2 for a in ['R','F','N']};maxerr=0.
for a in errors:
 scalar=sum(float((z[a][i,j]-z['y'][j])**2) for i in range(3) for j in range(400))/1200;assert abs(scalar**.5-s['pooled_RMSE'][a])<1e-12
for ref,new in [('R','N'),('F','N'),('R','F')]:
 key=new+'_vs_'+ref;er=errors[ref];en=errors[new];bg=np.sqrt(counts@er.mean(0)/400)-np.sqrt(counts@en.mean(0)/400);err=float(abs(bg-saved[key]).max());assert err<1e-12;maxerr=max(maxerr,err);np.testing.assert_allclose(np.quantile(bg,[.025,.975]),s['contrasts'][key]['pooled']['conditional_CI95'],atol=1e-12,rtol=0)
 for si,split in enumerate(SPLITS):
  g=float(np.sqrt(er[si].mean())-np.sqrt(en[si].mean()));assert abs(g-s['contrasts'][key]['by_partition'][split]['gain'])<1e-12
 for a,b,rec in [(er.mean(0),en.mean(0),s['contrasts'][key]['pooled']['influence']),*[(er[i],en[i],s['contrasts'][key]['by_partition'][sp]['influence']) for i,sp in enumerate(SPLITS)]]:
  order=sorted(range(400),key=lambda j:(-float(a[j]-b[j]),j))
  for top in [1,5,10]:
   keep=order[top:];gain=(sum(float(a[j]) for j in keep)/len(keep))**.5-(sum(float(b[j]) for j in keep)/len(keep))**.5;assert abs(gain-rec['top_benefit_removal'][str(top)]['gain_after_removal'])<1e-12
vp={a:np.array([sum(float((z[a][i,j]-sum(z[a][:,j])/3)**2) for i in range(3))/3 for j in range(400)]) for a in ['R','F','N']}
for a in vp:assert abs(vp[a].mean()-s['partition_variance']['mean_prediction_variance'][a])<1e-12
vg=counts@(vp['R']-vp['N'])/400;assert abs(vg-saved['partition_variance_gain']).max()<1e-12
assert s['strong_development_gate']==all(s['criteria'].values());c.W(O/'score-validation.json',{'passed':True,'independent_scalar_all1200_squared_errors':True,'independent_all10000_clusterbootstrap_count_replay':True,'max_bootstrap_error':maxerr,'all_influence_removals_and_partitionvariance_verified':True,'summary_sha256':c.H(O/'summary.json'),'ensemble_sha256':c.H(O/'ensemble-OOF.npz')});c.access_report('score-validation-access.json');c.emit({'score_validation_passed':True,'max_bootstrap_error':maxerr})
