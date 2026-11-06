"""202: fixed, prediction-only simplification and conditional seed variance."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[key]='2'
import csv,datetime,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
ROOT=_RepoPath(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
assert Path.cwd()==ROOT
PLAN=json.loads((_repo_path(OUT, 'plan.json')).read_text()); SPLITS=PLAN['data']['splits']; ARMS=['R','A_no_residual','B_fixed']; COMPONENTS=['CNN','S','B','L']
READS=set(); BLOCKED=[]
def sha(p):
 with open(_repo_io(p),'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):
 with (_repo_path(OUT, name)).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def guard(event,args):
 if event.startswith('socket.'):raise RuntimeError('Network forbidden')
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(os.fsdecode(args[0])).absolute()
  if p.is_relative_to(ROOT):
   rel=str(p.relative_to(ROOT));mode=args[1];flags=args[2]
   writing=(isinstance(mode,str) and any(x in mode for x in 'wa+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
   if writing and not p.is_relative_to(OUT):BLOCKED.append(rel);raise RuntimeError('Write outside202: '+rel)
   if not p.is_relative_to(OUT) and not (p.is_relative_to(_repo_path(ROOT, '.venv')) and not writing) and rel not in PLAN['input_sha256'] and rel not in PLAN['protected_sha256']:
    BLOCKED.append(rel);raise RuntimeError('Unplanned project read: '+rel)
   READS.add(rel)
  elif str(p).startswith('/Users/'):
   BLOCKED.append('outside_project');raise RuntimeError('Other user directory forbidden')
sys.addaudithook(guard)
def read(rel):
 with (_repo_path(ROOT, rel)).open() as f:return json.load(f)
def npz(rel):
 with np.load(_repo_path(ROOT, rel),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=0)
def mse(y,p):return float(np.mean(np.square(p-y)))
def rmse(y,p):return float(np.sqrt(mse(y,p)))
def verify_hashes():
 for rel,h in {**PLAN['input_sha256'],**PLAN['protected_sha256']}.items():assert sha(_repo_path(ROOT, rel))==h,rel
 for name,h in read('experiments/simplification-202/code-lock.json')['sources'].items():assert sha(_repo_path(OUT, name))==h,name

def reported(folder,split):
 report=read(f'{folder}/{split}/report.json');path=f'{folder}/{split}/predictions.npz'
 assert sha(_repo_path(ROOT, path))==report['predictions_sha256'];return npz(path),report

def load_split(split):
 old,rec121=reported('artifacts/component-ablation-121',split)
 cnn,rec133=reported('artifacts/cnn-seed-mean-133',split)
 resid,rec144=reported('artifacts/bootstrap-selector-144',split)
 large,rec177=reported('experiments/dinol-177',split)
 audit=npz(f'artifacts/cnn-seed-mean-133/audit/{split}.npz')
 with (_repo_path(ROOT, f'artifacts/seed-stability-052/{split}/development_oof.csv')).open() as f:rows=list(csv.DictReader(f))
 ids=large['ids'];y=large['y'];fid=large['fold'];groups=large['g']
 assert ids.shape==y.shape==fid.shape==groups.shape==(400,)
 assert len(set(ids.tolist()))==len(set(groups.tolist()))==400
 for arr in [old,cnn,resid,audit]:
  np.testing.assert_array_equal(arr['ids'],ids);np.testing.assert_array_equal(arr['y'],y);np.testing.assert_array_equal(arr['fold'],fid)
 assert [r['ID'] for r in rows]==ids.tolist()
 close(np.array([float(r['hardness']) for r in rows]),y,0)
 np.testing.assert_array_equal(np.array([int(r['fold']) for r in rows]),fid)
 sv=read('reports/seed-052-validation.json');assert sv['valid'];assert sha(_repo_path(ROOT, f'artifacts/seed-stability-052/{split}/development_oof.csv'))==sv['results'][split]['oof_sha256']
 B=npz(f'artifacts/dinob-complement-064/{split}/development_predictions.npz');br=read(f'artifacts/dinob-complement-064/{split}/report.json')
 assert sha(_repo_path(ROOT, f'artifacts/dinob-complement-064/{split}/development_predictions.npz'))==br['predictions_sha256']
 for k in ['ids','y','fold']:np.testing.assert_array_equal(B[k],large[k])
 seeds={'CNN':audit['seeds'],'S':np.array([[float(r[f'seed_{s}']) for r in rows] for s in [20261128,20261205,20261219]]),'B':B['Bseeds'],'L':large['L_MLP_seeds']}
 for name,p in seeds.items():assert p.shape==(3,400) and np.isfinite(p).all()
 close(seeds['CNN'].mean(0),audit['mean3'],0);close(seeds['S'].mean(0),B['Smean']);close(seeds['B'].mean(0),B['Bmean'],0);close(seeds['L'].mean(0),large['L_MLP'],0)
 close(old['components'][:,2],seeds['CNN'][0]);close(old['components'][:,3],seeds['S'].mean(0));close(old['components'][:,4],seeds['B'].mean(0))
 close(large['B_MLP_seeds'],seeds['B'],1e-8)
 raw=.3*old['components'][:,1]+.3*seeds['CNN'].mean(0)+.2*seeds['S'].mean(0)+.2*seeds['B'].mean(0)
 close(raw,cnn['raw']);base=cnn['base'];short=old['short'];L=seeds['L'].mean(0)
 slopes=np.zeros(400);centers=np.zeros(400);targets=np.zeros(400);correction=np.zeros(400)
 for k in range(5):
  V=np.flatnonzero(fid==k);T=np.flatnonzero(fid!=k)
  assert len(V)==80 and not set(groups[T])&set(groups[V])
  c=read(f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json')['outer_calibrator']
  slopes[V]=c['slope'];centers[V]=c['prediction_center'];targets[V]=c['target_center']
  path=f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}'
  rr=read(path+'/report.json');zz=npz(path+'/predictions.npz');assert sha(_repo_path(ROOT, path+'/predictions.npz'))==rr['predictions_sha256']
  np.testing.assert_array_equal(zz['outer_indices'],V);close(zz['base'],base[V])
  cc=[]
  for j in range(3):
   path=f'artifacts/bootstrap-selector-144/{split}/fold-{k}-block-{j}';rj=read(path+'/report.json');zj=npz(path+'/predictions.npz')
   assert sha(_repo_path(ROOT, path+'/predictions.npz'))==rj['predictions_sha256']
   assert sha(_repo_path(ROOT, path+'/report.json'))==rec144['block_report_sha256'][f'fold-{k}-block-{j}/report.json']
   np.testing.assert_array_equal(zj['outer_indices'],V);assert not set(zj['block_indices'])&set(V)
   close(zj['weights']@zj['full_candidate_predictions'],zj['primary'],1e-12);cc.append(zj['primary'])
  correction[V]=np.mean(cc,axis=0)
 affine=targets+slopes*(raw-centers)
 close(base,.9*affine+.1*short);close(resid['primary'],base+correction)
 close(large['reference018'],resid['primary'],0)
 pred={'R':.9*resid['primary']+.1*L,'A_no_residual':.9*base+.1*L,'B_fixed':.9*(.9*raw+.1*short)+.1*L}
 close(pred['R'],large['blend_L_MLP'],0)
 close(pred['A_no_residual'],pred['R']-.9*correction)
 close(pred['B_fixed'],pred['A_no_residual']-.81*(affine-raw))
 for folder,rep in [('artifacts/component-ablation-121',rec121),('artifacts/cnn-seed-mean-133',rec133),('artifacts/bootstrap-selector-144',rec144)]:
  cert=read(f'{folder}/validation-{split}.json');assert cert['passed']
  if 'report_sha256' in cert:assert cert['report_sha256']==sha(_repo_path(ROOT, f'{folder}/{split}/report.json'))
 cert=read(f'experiments/dinol-177/{split}/validation.json');assert cert['passed'] and cert['report_sha256']==sha(_repo_path(ROOT, f'experiments/dinol-177/{split}/report.json'))
 assert all(np.isfinite(p).all() for p in pred.values())
 return {'ids':ids,'y':y,'groups':groups,'fold':fid,'raw':raw,'short':short,'L':L,'affine':affine,'base':base,'correction':correction,'slopes':slopes,'seeds':seeds,'pred':pred}

def influence(refse,newse):
 benefit=refse-newse;order=np.argsort(-benefit,kind='stable');net=float(benefit.sum());out={}
 for n in [1,5,10]:
  keep=np.ones(len(benefit),bool);keep[order[:n]]=False
  out[str(n)]={'benefit_sum':float(benefit[order[:n]].sum()),'net_SSE_fraction':float(benefit[order[:n]].sum()/net) if net>0 else None,'gain_after_removal':float(np.sqrt(refse[keep].mean())-np.sqrt(newse[keep].mean()))}
 return {'positive_image_count':int((benefit>0).sum()),'net_SSE_gain':net,'top_benefit_removal':out}

def seed_analysis(ds):
 results={};pred=[d['pred'] for d in ds]
 for arm in ARMS:
  pieces={};sd=[]
  for name in COMPONENTS:
   variance=[];weighted=[];identity=[];means=[]
   for d in ds:
    coeff=np.full(400,{'CNN':.243,'S':.162,'B':.162,'L':.1}[name])
    if name!='L' and arm!='B_fixed':coeff*=d['slopes']
    P=d['seeds'][name];v=P.var(axis=0,ddof=0);sv=P.var(axis=0,ddof=1)
    changed=d['pred'][arm][None,:]+coeff[None,:]*(P-P.mean(0)[None,:])
    diff=np.mean((changed-d['y'][None,:])**2)-mse(d['y'],d['pred'][arm])
    rhs=float(np.mean(coeff**2*v));assert abs(diff-rhs)<1e-10
    variance.append(v);weighted.append(coeff**2*sv);identity.append({'mean_individual_substitution_MSE_minus_current':float(diff),'weighted_population_seed_variance':rhs,'identity_abs_error':abs(float(diff)-rhs)})
    means.append({'conditional_mean3_MSE_variance_proxy':float(np.mean(coeff**2*sv/3)),'conditional_3_to9_MSE_reduction_proxy':float(np.mean(coeff**2*sv*(1/3-1/9)))})
   v=np.stack(variance);w=np.stack(weighted);sd.append(np.sqrt(w/3))
   pieces[name]={'raw_population_seed_variance_by_partition':[float(x.mean()) for x in v],
    'weighted_population_seed_variance_by_partition':[x['weighted_population_seed_variance'] for x in identity],
    'identity_by_partition':identity,'conditional_iid_proxies_by_partition':means,
    'pooled_conditional_mean3_MSE_variance_proxy':float(np.mean(w/3)),
    'pooled_conditional_3_to9_MSE_reduction_proxy':float(np.mean(w*(1/3-1/9)))}
  ss=np.stack(sd);upper=ss.sum(0)**2;lower=np.maximum(2*ss.max(0)-ss.sum(0),0)**2
  results[arm]={'components':pieces,'independent_component_sum_mean3_MSE_variance_proxy':sum(x['pooled_conditional_mean3_MSE_variance_proxy'] for x in pieces.values()),
   'covariance_agnostic_mean3_variance_bounds':[float(lower.mean()),float(upper.mean())],
   'independent_component_sum_3_to9_MSE_reduction_proxy':sum(x['pooled_conditional_3_to9_MSE_reduction_proxy'] for x in pieces.values()),
   'scope':'Conditional direct-head variance only, fixed downstream fits,3seed realizations,unknown cross-component covariance,not a prediction of selected-model risk improvement;no future-seed training selected.'}
 return results

def main():
 start=time.perf_counter();verify_hashes();ds=[load_split(s) for s in SPLITS]
 for d in ds[1:]:
  for k in ['ids','y','groups']:np.testing.assert_array_equal(ds[0][k],d[k])
 ids=ds[0]['ids'];y=ds[0]['y'];groups=ds[0]['groups'];fold=np.stack([d['fold'] for d in ds])
 P={a:np.stack([d['pred'][a] for d in ds]) for a in ARMS};SE={a:(v-y[None,:])**2 for a,v in P.items()}
 draws=np.random.default_rng(202610202).integers(0,400,(10000,400),dtype=np.int64)
 with (_repo_path(OUT, 'bootstrap-draws.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),draws=draws)
 with (_repo_path(OUT, 'predictions.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),ids=ids,y=y,groups=groups,fold=fold,**P,slopes=np.stack([d['slopes'] for d in ds]),raw=np.stack([d['raw'] for d in ds]),short=np.stack([d['short'] for d in ds]),affine=np.stack([d['affine'] for d in ds]),correction=np.stack([d['correction'] for d in ds]),**{n+'_seeds':np.stack([d['seeds'][n] for d in ds]) for n in COMPONENTS})
 pooled={a:float(np.sqrt(v.mean())) for a,v in SE.items()};by_split={s:{a:float(np.sqrt(SE[a][i].mean())) for a in ARMS} for i,s in enumerate(SPLITS)}
 contrasts={};boots={};inst={a:np.var(P[a],axis=0,ddof=0) for a in ARMS}
 for arm in ARMS[1:]:
  ref=SE['R'].mean(0);new=SE[arm].mean(0)
  gain=np.sqrt(ref[draws].mean(1))-np.sqrt(new[draws].mean(1));boots[arm]=gain
  row={'pooled_RMSE_gain':pooled['R']-pooled[arm],'conditional_IDcluster_CI95':np.quantile(gain,[.025,.975]).tolist(),'pooled_influence':influence(ref,new),'per_partition':{},'partition_instability_gain':float(np.mean(inst['R']-inst[arm])),'conditional_partition_instability_gain_CI95':np.quantile((inst['R']-inst[arm])[draws].mean(1),[.025,.975]).tolist()}
  delta=P[arm]-P['R'];cross=float(np.mean(2*(P['R']-y[None,:])*delta));square=float(np.mean(delta**2));actual=float(np.mean(SE[arm]-SE['R']));assert abs(cross+square-actual)<1e-10
  row['exact_MSE_change']={'cross_term':cross,'delta_square_term':square,'sum':cross+square,'direct':actual}
  for i,s in enumerate(SPLITS):
   fg=[float(np.sqrt(SE['R'][i,fold[i]==k].mean())-np.sqrt(SE[arm][i,fold[i]==k].mean())) for k in range(5)]
   g=np.sqrt(SE['R'][i][draws].mean(1))-np.sqrt(SE[arm][i][draws].mean(1))
   row['per_partition'][s]={'gain':by_split[s]['R']-by_split[s][arm],'fold_gains':fg,'positive_folds':sum(x>0 for x in fg),'conditional_CI95':np.quantile(g,[.025,.975]).tolist(),'influence':influence(SE['R'][i],SE[arm][i])}
  contrasts[arm]=row
 c=contrasts['B_fixed'];gate={'pooled_gain_at_least_002':c['pooled_RMSE_gain']>=.02,'all_partition_gains_positive':all(v['gain']>0 for v in c['per_partition'].values()),'conditional_CI_lower_positive':c['conditional_IDcluster_CI95'][0]>0,'pooled_top5_removal_positive':c['pooled_influence']['top_benefit_removal']['5']['gain_after_removal']>0,'partition_variance_not_increased':c['partition_instability_gain']>=0}
 seed=seed_analysis(ds)
 with (_repo_path(OUT, 'bootstrap-results.npz')).open('xb') as f:np.savez_compressed(_repo_io(f),**boots)
 write('seed-variance.json',seed)
 summary={'study':202,'RMSE_by_partition':by_split,'pooled_RMSE':pooled,'contrasts':contrasts,'mean_across_partition_prediction_variance':{a:float(v.mean()) for a,v in inst.items()},'primary_only':'B_fixed','operational_gate_before_independent_validation':gate,'passes_numeric_gate':all(gate.values()),'A_diagnostic_not_eligible':True,'uncertainty':PLAN['metrics']['uncertainty'],'same400_adaptive_history':True,'new_fits':0,'holdout_or_test_access':False,'cost':0,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.perf_counter()-start,'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'source_sha256':sha(Path(__file__))}
 assert summary['RSS_bytes']<6e9;verify_hashes();write('summary.json',summary)
 write('access-audit.json',{'planned_project_files_opened':sorted(READS),'blocked_accesses':BLOCKED,'network':False,'old_rawdata_or_image_paths_opened':False,'mode':'Only explicit209 inputs,protected50 hashes and own202 outputs allowed.'})
 print(json.dumps({'study':202,'RMSE':by_split,'pooled':pooled,'gate':gate,'RSS_bytes':summary['RSS_bytes'],'seconds':summary['seconds']},ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  name='failure-'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')+'.json'
  write(name,{'error':repr(exc),'type':type(exc).__name__,'blocked':BLOCKED});raise
