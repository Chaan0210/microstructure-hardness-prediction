"""Project-only comparison after fresh176 fitting; never enters training driver."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import numpy as np,torch,joblib
import train as t
import infer as inf
R=t.PROJECT;O=t.OUT

def load(p):
 with np.load(_repo_io(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def diff(a,b):return {'exact':bool(np.array_equal(a,b)),'max_abs':float(abs(a-b).max()),'RMS':float(np.sqrt(np.mean((a-b)**2)))}
def states(a,b):
 assert set(a)==set(b);return {'exact':all(torch.equal(a[k],b[k]) for k in a),'max_abs':max(float((a[k]-b[k]).abs().max()) for k in a)}
def main():
 t.check();assert t.read(O/'training-validation.json')['passed'];d=t.input_data();root=O/'full';reference={};records={};failures=[];inf.guards()
 def remember(p):reference[str(p.relative_to(R))]=t.sha(p);return p
 def checkpoint(p):return torch.load(_repo_io(remember(p)),map_location='cpu',weights_only=True)
 for name in ['outer']+[n for j in range(3) for n in [f'middle-{j}']+[f'deep-{j}-{ell}' for ell in range(3)]]:
  path=root/name;nr=t.read(path/'report.json');new=load(path/'predictions.npz');U=new['U'];Q=new['Q'];old={};files={}
  if name=='outer':
   km=joblib.load(_repo_io(remember(R/'artifacts/final-model-017/kernel.joblib')));assert km['train_ids']==d['ids'].tolist();old['KRR']=km['model'].predict(d['K']);source=R/'artifacts/final-model-024'
   for seed in t.CNN_SEEDS:
    file=source/f'cnn-seed{seed}.pt';obj=checkpoint(file);old[f'CNN_{seed}']=inf.cnn(d['images'],obj,'mps');files[f'CNN-{seed}.pt']=(file,'state_dict')
   for arm in ['S','B']:
    for seed in t.HEAD_SEEDS:
     file=R/f'artifacts/final-model-065/{arm}-mlp-seed-{seed}.pt';obj=checkpoint(file);old[f'{arm}_{seed}']=inf.head(d[arm],obj);files[f'{arm}-{seed}.pt']=(file,'model')
   file=R/'artifacts/final-model-086/head/model.pt';obj=checkpoint(file);old['short']=inf.short(d['B'],obj,'mps');files['short.pt']=(file,'state')
   for key,seeds in [('CNN',t.CNN_SEEDS),('S',t.HEAD_SEEDS),('B',t.HEAD_SEEDS)]:old[key]=np.mean([old[f'{key}_{s}'] for s in seeds],0)
  else:
   if name.startswith('middle'):
    j=int(name.split('-')[1]);base=R/f'artifacts/final-model-075/inner-{j}';anchor=load(remember(R/f'artifacts/final-model-114/block-{j}/anchor.npz'));old['short']=anchor['shortB'];files['short.pt']=(R/f'artifacts/final-model-114/block-{j}/shortB/model.pt','state')
   else:
    _,j,ell=name.split('-');base=R/f'artifacts/final-model-114/block-{j}/deep-{ell}'
   z=load(remember(base/'predictions.npz'));np.testing.assert_array_equal(U,z['train_indices']);np.testing.assert_array_equal(Q,z['validation_indices']);old.update(KRR=z['kernel'],S=z['Smean'],B=z['Bmean']);cc=load(remember(R/f'artifacts/final-model-134/cnn-nested/{name}/predictions.npz'));np.testing.assert_array_equal(U,cc['train']);np.testing.assert_array_equal(Q,cc['validation']);old['CNN']=cc['mean3']
   for i,seed in enumerate(t.CNN_SEEDS):old[f'CNN_{seed}']=cc['seeds'][i];files[f'CNN-{seed}.pt']=(base/'cnn.pt' if i==0 else R/f'artifacts/final-model-134/cnn-nested/{name}/seed-{seed}.pt','state_dict')
   for arm in ['S','B']:
    for seed in t.HEAD_SEEDS:old[f'{arm}_{seed}']=z[f'{arm}_{seed}'];files[f'{arm}-{seed}.pt']=(base/f'{arm}-{seed}.pt','model')
  changes={key:diff(new[key],v) for key,v in old.items()};st={}
  for fn,(oldpath,key) in files.items():
   a=torch.load(_repo_io(path/fn),map_location='cpu',weights_only=True)[key];b=checkpoint(oldpath)[key]
   if fn=='short.pt':b={key:b[key] for key in a}
   st[fn]=states(a,b)
  for key,df in changes.items():
   tol=1e-8 if key=='KRR' or key.startswith(('S','B')) else 1e-3
   if df['max_abs']>tol:failures.append({'node':name,'component':key,**df})
  records[name]={'predictions':changes,'states':st,'all_states_exact':all(x['exact'] for x in st.values())}
 new=load(root/'predictions.npz');oldmid=load(remember(R/'artifacts/final-model-134/calibration_training.npz'));np.testing.assert_array_equal(d['ids'],oldmid['ids']);middle=diff(new['middle'][:,1:],oldmid['components'][:,1:]);cal=t.read(remember(R/'artifacts/final-model-134/calibrator.json'));oldbase=.9*(cal['target_center']+cal['slope']*(new['raw']-cal['prediction_center']))+.1*new['short'];base=diff(new['base'],oldbase);blocks=[];oldcor=[]
 for j in range(3):
  nn=load(root/f'residual-{j}/predictions.npz');oldinner=load(remember(R/f'artifacts/final-model-141/block-{j}/residual/predictions.npz'));oldfull=load(remember(R/f'artifacts/final-model-145/block-{j}/predictions.npz'));oldcor.append(oldfull['correction']);np.testing.assert_array_equal(nn['B'],oldinner['block_indices']);blocks.append({'anchor':diff(nn['anchor'],oldinner['anchor_predictions']),'residual':diff(nn['residual'],oldinner['residual']),'inner':diff(nn['inner'],oldinner['inner_predictions']),'weights':diff(nn['weights'],oldfull['weights']),'full':diff(nn['full'],oldfull['candidate_predictions']),'correction':diff(nn['primary'],oldfull['correction'])})
 full=diff(new['primary'],oldbase+np.mean(oldcor,0));count=sum(len(v['states']) for v in records.values());assert count==121
 if full['max_abs']>1e-3:failures.append({'stage':'complete_train_query',**full})
 r={'passed':not failures,'all_neural_states_exact':all(v['all_states_exact'] for v in records.values()),'neural_states_compared':count,'nodes':records,'middle_OOF':middle,'base':base,'residual_blocks':blocks,'complete_train_query_difference':full,'failures':failures,'no_training_performed_in_comparison':True,'no_subgroup_score':True,'test_read':False,'references':reference,'source_sha256':t.sha(Path(__file__))};t.write(O/'training-comparison.json',r);t.emit({k:r[k] for k in ['passed','all_neural_states_exact','neural_states_compared','middle_OOF','base','complete_train_query_difference','failures']})
if __name__=='__main__':main()
