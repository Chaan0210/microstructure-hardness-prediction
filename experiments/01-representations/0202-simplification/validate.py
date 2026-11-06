"""Independent202 verification: direct scalar reconstruction and count bootstrap."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import csv,hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
R=_RepoPath(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
plan=json.loads((_repo_path(O, 'plan.json')).read_text());readpaths=set();blocked=[]
def audit(event,args):
 if event.startswith('socket.'):raise RuntimeError('No network')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(R):
   s=str(p.relative_to(R));flags=args[2];mode=args[1]
   writing=(isinstance(mode,str) and any(k in mode for k in 'wa+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
   if (writing and not p.is_relative_to(O)) or (not p.is_relative_to(O) and not (p.is_relative_to(_repo_path(R, '.venv')) and not writing) and s not in plan['input_sha256'] and s not in plan['protected_sha256']):blocked.append(s);raise RuntimeError('Unexpected project file')
   readpaths.add(s)
  elif str(p).startswith('/Users/'):raise RuntimeError('Outside project')
sys.addaudithook(audit)
def J(p):return json.loads((_repo_path(R, p)).read_text())
def Z(p):
 with np.load(_repo_path(R, p),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def H(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert e<1e-9,e;return e
start=time.perf_counter()
for p,h in {**plan['input_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
lock=J('experiments/simplification-202/code-lock.json')
assert lock['plan_sha256']==H(_repo_path(O, 'plan.json'))
for name,h in lock['sources'].items():assert H(_repo_path(O, name))==h
z=Z('experiments/simplification-202/predictions.npz');d=Z('experiments/simplification-202/bootstrap-draws.npz')['draws'];b=Z('experiments/simplification-202/bootstrap-results.npz')
sumry=J('experiments/simplification-202/summary.json');seed=J('experiments/simplification-202/seed-variance.json')
assert d.shape==(10000,400) and z['R'].shape==(3,400) and z['fold'].shape==(3,400)
assert len(set(z['ids'].tolist()))==len(set(z['groups'].tolist()))==400
np.testing.assert_array_equal(d,np.random.default_rng(202610202).integers(0,400,(10000,400),dtype=np.int64))
maxformula=0.;maxidentity=0.;by_part={};manual={a:[] for a in ['R','A_no_residual','B_fixed']}
for si,s in enumerate(plan['data']['splits']):
 c=Z(f'artifacts/component-ablation-121/{s}/predictions.npz');old=Z(f'artifacts/cnn-seed-mean-133/audit/{s}.npz');L=Z(f'experiments/dinol-177/{s}/predictions.npz');B=Z(f'artifacts/dinob-complement-064/{s}/development_predictions.npz')
 with (_repo_path(R, f'artifacts/seed-stability-052/{s}/development_oof.csv')).open() as f:rows=list(csv.DictReader(f))
 S=np.array([[float(row[f'seed_{ss}']) for row in rows] for ss in [20261128,20261205,20261219]])
 for name,pp in [('CNN',old['seeds']),('S',S),('B',B['Bseeds']),('L',L['L_MLP_seeds'])]:close(z[name+'_seeds'][si],pp)
 for arr in [c,old,L,B]:
  np.testing.assert_array_equal(arr['ids'],z['ids']);np.testing.assert_array_equal(arr['y'],z['y']);np.testing.assert_array_equal(arr['fold'],z['fold'][si])
 m={a:np.empty(400) for a in manual}
 for f in range(5):
  rr=J(f'artifacts/cnn-seed-mean-133/{s}/pipeline-fold-{f}/report.json')['outer_calibrator']
  vs=np.flatnonzero(z['fold'][si]==f);assert len(vs)==80
  blocks=[]
  for j in range(3):
   zz=Z(f'artifacts/bootstrap-selector-144/{s}/fold-{f}-block-{j}/predictions.npz');np.testing.assert_array_equal(vs,zz['outer_indices'])
   assert not set(vs)&set(zz['block_indices']);blocks.append(zz)
  for local,i in enumerate(vs):
   cnn=sum(old['seeds'][:,i])/3;sm=sum(S[:,i])/3;bm=sum(B['Bseeds'][:,i])/3;lm=sum(L['L_MLP_seeds'][:,i])/3
   raw=(3*c['components'][i,1]+3*cnn+2*sm+2*bm)/10
   affine=rr['intercept']+rr['slope']*raw
   correction=sum(sum(bb['weights'][k]*bb['full_candidate_predictions'][k,local] for k in range(5)) for bb in blocks)/3
   m['A_no_residual'][i]=.81*affine+.09*c['short'][i]+.1*lm
   m['R'][i]=m['A_no_residual'][i]+.9*correction
   m['B_fixed'][i]=.243*c['components'][i,1]+.243*cnn+.162*sm+.162*bm+.09*c['short'][i]+.1*lm
 for a in manual:
  maxformula=max(maxformula,close(m[a],z[a][si]));manual[a].append(m[a])
 y=z['y'];by_part[s]={a:float(np.sqrt(sum((float(p)-float(yy))**2 for p,yy in zip(m[a],y))/400)) for a in manual}
 for a in manual:
  for name in ['CNN','S','B','L']:
   v=z[name+'_seeds'][si];w=np.full(400,{'CNN':.243,'S':.162,'B':.162,'L':.1}[name])
   if a!='B_fixed' and name!='L':w*=z['slopes'][si]
   rawvar=sum(sum((v[t,i]-sum(v[:,i])/3)**2 for t in range(3))/3 for i in range(400))/400
   weightedvar=sum(w[i]**2*sum((v[t,i]-sum(v[:,i])/3)**2 for t in range(3))/3 for i in range(400))/400
   ss=seed[a]['components'][name]
   maxidentity=max(maxidentity,close(rawvar,ss['raw_population_seed_variance_by_partition'][si]),close(weightedvar,ss['weighted_population_seed_variance_by_partition'][si]))
   # sample variance=1.5*population variance for n=3, divided by3 for mean3.
   maxidentity=max(maxidentity,close(weightedvar/2,ss['conditional_iid_proxies_by_partition'][si]['conditional_mean3_MSE_variance_proxy']),close(weightedvar/3,ss['conditional_iid_proxies_by_partition'][si]['conditional_3_to9_MSE_reduction_proxy']))
for s,rec in by_part.items():
 for a,v in rec.items():close(v,sumry['RMSE_by_partition'][s][a])
counts=np.zeros((10000,400),dtype=np.int64)
for i,row in enumerate(d):counts[i]=np.bincount(row,minlength=400)
errs={a:np.square(np.array(v)-z['y'][None,:]) for a,v in manual.items()};maxboot=0.
for a in ['A_no_residual','B_fixed']:
 ref=errs['R'].mean(0);new=errs[a].mean(0)
 boot=np.sqrt(counts@ref/400)-np.sqrt(counts@new/400)
 maxboot=max(maxboot,close(boot,b[a]),close(np.quantile(boot,[.025,.975]),sumry['contrasts'][a]['conditional_IDcluster_CI95']))
 for si,s in enumerate(plan['data']['splits']):
  br=np.sqrt(counts@errs['R'][si]/400)-np.sqrt(counts@errs[a][si]/400)
  close(np.quantile(br,[.025,.975]),sumry['contrasts'][a]['per_partition'][s]['conditional_CI95'])
 def removal(r,n,k):
  rank=sorted(range(400),key=lambda i:-(r[i]-n[i]));rest=rank[k:]
  return float((sum(r[i] for i in rest)/len(rest))**.5-(sum(n[i] for i in rest)/len(rest))**.5)
 for k in [1,5,10]:close(removal(ref,new,k),sumry['contrasts'][a]['pooled_influence']['top_benefit_removal'][str(k)]['gain_after_removal'])
 delta=np.array(manual[a])-np.array(manual['R']);cross=2*np.mean((np.array(manual['R'])-z['y'])*delta)
 close(cross,sumry['contrasts'][a]['exact_MSE_change']['cross_term'])
for a in manual:
 pp=np.array(manual[a]);v=sum((pp[j]-pp[k])**2 for j in range(3) for k in range(j+1,3))/9
 close(v.mean(),sumry['mean_across_partition_prediction_variance'][a])
P=np.array(manual['B_fixed']);Q=np.array(manual['R']);vb=sum((P[j]-P[k])**2 for j in range(3) for k in range(j+1,3))/9;vq=sum((Q[j]-Q[k])**2 for j in range(3) for k in range(j+1,3))/9
close(np.quantile(counts@(vq-vb)/400,[.025,.975]),sumry['contrasts']['B_fixed']['conditional_partition_instability_gain_CI95'])
assert all(np.isfinite(v).all() for v in errs.values());assert not blocked
for p,h in {**plan['input_sha256'],**plan['protected_sha256']}.items():assert H(_repo_path(R, p))==h,p
report={'passed':True,'independent_scalar_prediction_max_error':maxformula,'independent_count_bootstrap_max_error':maxboot,'independent_seed_variance_identity_max_error':maxidentity,'all10000_paired_cluster_draws_checked':True,'all3x400_ID_label_fold_order_verified':True,'groups_unique':400,'all15_folds80rows':True,'all45_residual_block_reconstructions':True,'fixed_base_coefficients_checked':True,'formula_no_fit_or_new_scaler':True,'all209_input_hashes_and50_protected_hashes_unchanged':True,'access_blocked':blocked,'read_paths':sorted(readpaths),'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.perf_counter()-start,'plan_sha256':H(_repo_path(O, 'plan.json')),'summary_sha256':H(_repo_path(O, 'summary.json')),'source_sha256':H(Path(__file__))}
assert report['RSS_bytes']<6e9
with (_repo_path(O, 'validation.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in report.items() if k!='read_paths'}),flush=True)
