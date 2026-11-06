"""Comparison-only process: reads old outputs/states only after fresh nodes frozen."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time
from pathlib import Path
import numpy as np,torch
import train as t
R=t.PROJECT;OUT=t.OUT

def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {n:z[n].copy() for n in z.files}
def diff(a,b):return {'max_abs':float(abs(a-b).max()),'RMS':float(np.sqrt(np.mean((a-b)**2))),'exact':bool(np.array_equal(a,b))}
def state(file,key):return torch.load(_repo_io(file),map_location='cpu',weights_only=True)[key]
def sdif(a,b):
 assert set(a)==set(b)
 return {'exact':all(torch.equal(a[n],b[n]) for n in a),'max_abs':max(float((a[n]-b[n]).abs().max()) for n in a)}
def main(k):
 t.check();d=t.input_data();path=OUT/'original'/f'fold-{k}';assert (path/'report.json').exists();source={};nodes={};failures=[]
 def remember(p):source[str(p.relative_to(R))]=t.sha(p);return p
 outerS=load(remember(R/'artifacts/dinob-mlp-063/original/development_predictions.npz'));outerC=load(remember(R/'artifacts/cnn-seed-mean-133/audit/original.npz'));outerShort=load(remember(R/'artifacts/frozen-complement-085/original/development_predictions.npz'))
 for z in [outerS,outerC,outerShort]:assert z['ids'].tolist()==d['ids'].tolist()
 for name in ['outer']+[n for j in range(3) for n in [f'middle-{j}']+[f'deep-{j}-{ell}' for ell in range(3)]]:
  fresh=path/name;rr=t.read(fresh/'report.json');U=np.array(rr['U']);Q=np.array(rr['Q']);new=load(fresh/'predictions.npz');old={};ptpaths={}
  if name=='outer':
   oldK=load(remember(R/f'artifacts/krr-scope-174/original/fold-{k}/all/kernels/outer/predictions.npz'));np.testing.assert_array_equal(U,oldK['U']);np.testing.assert_array_equal(Q,oldK['Q']);old['KRR']=oldK['predictions'];old['CNN']=outerC['mean3'][Q];old['S']=outerS['controlS'][Q];old['B']=outerS['baseB'][Q];old['short']=outerShort['frozen'][Q]
   for i,seed in enumerate(t.CNN_SEEDS):
    old[f'CNN_{seed}']=outerC['seeds'][i,Q];ptpaths[f'CNN-{seed}.pt']=(R/(f'artifacts/run-009/fold-{k}.pt' if i==0 else f'artifacts/run-018/cnn-seed{seed}-fold{k}.pt'),'state_dict')
   for arm,oldarm in [('S','controlS'),('B','baseB')]:
    for i,seed in enumerate(t.HEAD_SEEDS):old[f'{arm}_{seed}']=outerS[oldarm+'_seeds'][i,Q];ptpaths[f'{arm}-{seed}.pt']=(R/f'artifacts/dinob-mlp-063/original/fold-{k}/{oldarm}-seed-{seed}.pt','model')
   ptpaths['short.pt']=(R/f'artifacts/dinob-joint-084/original/fold-{k}/frozen/model.pt','state')
  else:
   if name.startswith('middle'):
    j=int(name.split('-')[1]);base=R/f'artifacts/nested-calibration-074/original/fold-{k}/inner-{j}';sh=load(remember(R/f'artifacts/nested-fusion-112/original/fold-{k}/inner-{j}/calibration.npz'));old['short']=sh['shortB'];ptpaths['short.pt']=(R/f'artifacts/nested-fusion-112/original/fold-{k}/inner-{j}/shortB/model.pt','state')
   else:
    _,j,ell=name.split('-');base=R/f'artifacts/nested-fusion-112/original/fold-{k}/inner-{j}/deep-{ell}'
   z=load(remember(base/'predictions.npz'));np.testing.assert_array_equal(U,z['train_indices']);np.testing.assert_array_equal(Q,z['validation_indices']);old.update(KRR=z['kernel'],S=z['Smean'],B=z['Bmean']);cnn=load(remember(R/f'artifacts/cnn-seed-mean-133/original/cnn-fold-{k}/{name}/predictions.npz'));np.testing.assert_array_equal(cnn['train'],U);np.testing.assert_array_equal(cnn['validation'],Q);old['CNN']=cnn['mean3']
   for i,seed in enumerate(t.CNN_SEEDS):
    old[f'CNN_{seed}']=cnn['seeds'][i];ptpaths[f'CNN-{seed}.pt']=(base/'cnn.pt' if i==0 else R/f'artifacts/cnn-seed-mean-133/original/cnn-fold-{k}/{name}/seed-{seed}.pt','state_dict')
   for arm in ['S','B']:
    for seed in t.HEAD_SEEDS:old[f'{arm}_{seed}']=z[f'{arm}_{seed}'];ptpaths[f'{arm}-{seed}.pt']=(base/f'{arm}-{seed}.pt','model')
  differences={n:diff(new[n],p) for n,p in old.items()};states={}
  for fn,(oldpath,key) in ptpaths.items():
   a=state(fresh/fn,key);b=state(remember(oldpath),key)
   if fn=='short.pt':b={n:b[n] for n in a}
   states[fn]=sdif(a,b)
  for n,df in differences.items():
   tol=1e-8 if n=='KRR' or n.startswith(('S','B')) else 1e-3
   if df['max_abs']>tol:failures.append({'node':name,'component':n,**df,'tolerance':tol})
  nodes[name]={'prediction_differences':differences,'state_differences':states,'all_states_exact':all(v['exact'] for v in states.values())}
 # Entire018 downstream comparison against174 control, independently certified exact018.
 oldroot=R/f'artifacts/krr-scope-174/original/fold-{k}/all/pipeline';new=load(path/'predictions.npz');old=load(remember(oldroot/'predictions.npz'));down={key:diff(new[key],old[key]) for key in ['base','corrections','primary']};blocks=[]
 for j in range(3):
  nn=load(path/f'residual-{j}/predictions.npz');oo=load(remember(oldroot/f'block-{j}/predictions.npz'));blocks.append({key:diff(nn[key],oo[key]) for key in ['anchor','residual','inner','weights','full','primary']})
 if down['primary']['max_abs']>1e-3:failures.append({'component':'whole_pipeline',**down['primary']})
 result={'passed':not failures,'fold':k,'nodes':nodes,'downstream':down,'residual_blocks':blocks,'failures':failures,'all_neural_states_exact':all(v['all_states_exact'] for v in nodes.values()),'new_RMSE':t.b.rmse(d['y'][new['V']],new['primary']),'old_RMSE':t.b.rmse(d['y'][new['V']],old['primary']),'source_hashes':source,'source_files_used_for_comparison_only':True,'no_fitting_performed':True,'comparison_source_sha256':t.sha(Path(__file__))}
 t.write(OUT/f'comparison-fold-{k}.json',result);t.emit({key:result[key] for key in ['passed','fold','downstream','failures','all_neural_states_exact','new_RMSE','old_RMSE']})
if __name__=='__main__':
 for k in map(int,sys.argv[1:]):main(k)
