import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import np,O,PLAN
c.check();s=c.J(O/'summary.json');z=c.Z(O/'ensemble-OOF.npz');d=c.Z(O/'bootstrap-draws.npz')['draws'];b=c.Z(O/'bootstrap-results.npz');counts=np.array([np.bincount(row,minlength=400) for row in d],dtype=np.int64)
E=(z['primary']-z['y'])**2;F=(z['reference']-z['y'])**2
boot=np.sqrt(counts@F.mean(0)/400)-np.sqrt(counts@E.mean(0)/400)
assert float(abs(boot-b['gain']).max())<1e-10;np.testing.assert_allclose(np.quantile(boot,[.025,.975]),s['pooled']['conditional_CI95'],atol=1e-10,rtol=0)
Q=z['reference'];P=z['primary'];vq=sum((Q[j]-Q[k])**2 for j in range(3) for k in range(j+1,3))/9;vp=sum((P[j]-P[k])**2 for j in range(3) for k in range(j+1,3))/9
var=counts@(vq-vp)/400;assert abs(var-b['variance_gain']).max()<1e-10
maxerr=0.
for si,split in enumerate(PLAN['splits']):
 np.testing.assert_allclose(np.quantile(np.sqrt(counts@F[si]/400)-np.sqrt(counts@E[si]/400),[.025,.975]),s['by_partition'][split]['conditional_CI95'],atol=1e-10,rtol=0)
 for name,arr in [('old3',z['old_seeds'][si]),('new3',z['new_seeds'][si]),('all6',z['all_seeds'][si])]:
  n=len(arr);sample=sum(sum((float(arr[j,i])-sum(arr[:,i])/n)**2 for j in range(n))/(n-1) for i in range(400))/400
  weighted=sum(z['coefficient'][si,i]**2*sum((float(arr[j,i])-sum(arr[:,i])/n)**2 for j in range(n))/(n-1)/n for i in range(400))/400
  row=s['seed_variance'][split][name];maxerr=max(maxerr,abs(sample-row['sample_seed_variance']),abs(weighted-row['conditional_final_direct_mean_variance_proxy']))
  assert maxerr<1e-9
for a,b_,rec in [(F.mean(0),E.mean(0),s['pooled']['influence'])]+[(F[i],E[i],s['by_partition'][split]['influence']) for i,split in enumerate(PLAN['splits'])]:
 order=sorted(range(400),key=lambda i:-(a[i]-b_[i]));assert sum(a>b_)==rec['positive_IDs']
 for n in [1,5,10]:
  ix=order[n:];gain=(sum(float(a[i]) for i in ix)/len(ix))**.5-(sum(float(b_[i]) for i in ix)/len(ix))**.5
  assert abs(gain-rec['top_benefit_removal'][str(n)]['gain_after_removal'])<1e-10
c.check();c.W(O/'score-validation.json',{'passed':True,'all10000_IDcluster_bootstrap_count_replay':True,'bootstrap_max_abs_error':float(abs(boot-c.Z(O/'bootstrap-results.npz')['gain']).max()),'independent_seed_variance_max_abs_error':maxerr,'influence_all3partitions_andpooled_top1_5_10_verified':True,'acrosspartition_variance_pairwise_identity_verified':True,'same400_not1200':True,'all_preexisting_hashes_unchanged':True,'summary_sha256':c.H(O/'summary.json')});c.emit({'score_validation_passed':True,'independent_seed_variance_error':maxerr})
