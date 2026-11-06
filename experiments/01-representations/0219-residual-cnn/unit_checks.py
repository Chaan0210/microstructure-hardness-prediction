"""CPU-only fixtures, cached model replay, labels isolation. NO main fit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,time,json,traceback,hashlib,resource,shutil,copy
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
O=Path(__file__).resolve().parent;R=_RepoPath(__file__).resolve().parents[3];sys.path.insert(0,str(O))
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits,threadpool_info
from sklearn.model_selection import GroupKFold
from contracts import Labels,Features,AnchorGraph,require,sha,lock_manifest
from anchor import fit_complete_anchor,coefficient,apply,bootstrap_weights
from bundle import freeze_anchor,residual_targets,save_bundle,load_bundle
from backend import ProductionBackend,TinyCNN,training_guard,cnn_normalization
from data import load_development,graph_for
from evaluation import evaluate_all
import residual_models
threadpool_limits(1);torch.set_num_threads(1)
class FixturePredictor:
 def __init__(self,X,y,U,arm='base',anchor_sha256=''):
  self.train=tuple(U);self.arm=arm;self.coef=np.linalg.lstsq(np.column_stack([np.ones(len(X)),X]),y,rcond=None)[0];self.anchor_sha256=anchor_sha256
 def predict(self,f,Q):return np.column_stack([np.ones(len(Q)),f.take('X',Q)])@self.coef
class FixtureBackend:
 def __init__(self):self.calls=[]
 def make(self,f,l,offset,arm):
  self.calls.append({'train':l.ids.tolist(),'arm':arm,'offset':offset});return FixturePredictor(f.take('X',l.ids),l.take(l.ids),l.ids,arm)
 def base(self,f,l,offset):return self.make(f,l,offset,'base')
 def long(self,f,l,offset,arm):return self.make(f,l,offset,arm)
 def short(self,f,l,offset):return self.make(f,l,offset,'short')

def expect_error(fn,match):
 try:fn()
 except (ValueError,RuntimeError) as e:
  require(match in str(e),'Wrong failure: '+str(e));return
 raise AssertionError('Expected rejection: '+match)

def fixture():
 rng=np.random.default_rng(219);X=rng.normal(size=(90,6));y=150+X@np.arange(1.,7.)+rng.normal(size=90)*3;f=Features(np.array([f'fixture-{i}' for i in range(90)]),np.arange(90),{'X':X},['a']*3+['b']*3,[],scope='synthetic_unit_fixture');A=np.arange(54);B=np.arange(54,72);V=np.arange(72,90);cells=[]
 for j,(ti,vi) in enumerate(GroupKFold(3,shuffle=True,random_state=20270110).split(A,groups=f.groups[A])):
  C,D=A[ti],A[vi];deep=[]
  for ell,(it,iv) in enumerate(GroupKFold(3,shuffle=True,random_state=20270210+j).split(C,groups=f.groups[C])):deep.append({'E':C[it].tolist(),'F':C[iv].tolist(),'offset':300+j*100+ell})
  cells.append({'C':C.tolist(),'D':D.tolist(),'offset':200+j,'deep':deep})
 return f,y,A,B,V,AnchorGraph(tuple(A),100,tuple(cells))

def isolation_checks(folder):
 f,y,A,B,V,g=fixture();backend=FixtureBackend();a=fit_complete_anchor(f,Labels(A,y[A]),g,backend);p=a.predict(f,np.r_[B,V]);require(all(set(c['train'])<=set(A) for c in backend.calls),'Anchor backend received forbidden label IDs')
 for poison in (12345.,float('nan')):
  yp=y.copy();yp[np.r_[B,V]]=poison;l=Labels(A,yp[A]);other=fit_complete_anchor(f,l,g,FixtureBackend());np.testing.assert_array_equal(other.predict(f,np.r_[B,V]),p)
 expect_error(lambda:Labels(A,y[A]).take(B),'Forbidden label read');expect_error(lambda:Labels(A,y[A]).restrict(np.r_[A,B]),'Forbidden label read')
 bad=copy.deepcopy(g.cells);bad[0]['deep'][0]['E'][0]=int(V[0]);expect_error(lambda:AnchorGraph(g.A,g.offset,bad).validate(f),'Partition membership')
 expect_error(lambda:fit_complete_anchor(f,Labels(np.r_[A,B],y[np.r_[A,B]]),g,FixtureBackend()),'exactly A')
 path=folder/'anchor.joblib';protocol=sha(_repo_path(O, 'protocol.json'));binding=freeze_anchor(a,path,protocol);target=residual_targets(f,B,Labels(B,y[B]),path,binding);model=FixturePredictor(f.take('X',B),target.take(B),B,'fixture-residual',binding['anchor_sha256'])
 for poison in (23456.,float('nan')):
  yp=y.copy();yp[V]=poison;changed=residual_targets(f,B,Labels(B,yp[B]),path,binding);np.testing.assert_array_equal(changed.take(B),target.take(B));other=FixturePredictor(f.take('X',B),changed.take(B),B,'fixture-residual',binding['anchor_sha256']);np.testing.assert_array_equal(other.predict(f,V),model.predict(f,V))
 manifest=save_bundle(folder/'bundle',path,binding,model,f,B,protocol,'unit-split');loaded=load_bundle(folder/'bundle',protocol,'unit-split');np.testing.assert_array_equal(loaded.predict(f,V),a.predict(f,V)+model.predict(f,V));expect_error(lambda:load_bundle(folder/'bundle',protocol,'different-split'),'Recipe/split mismatch')
 changed=copy.deepcopy(model);changed.anchor_sha256='wrong';expect_error(lambda:save_bundle(folder/'bad-bundle',path,binding,changed,f,B,protocol,'unit-split'),'different anchor')
 tamper=folder/'tamper';shutil.copytree(_repo_io(folder/'bundle'),_repo_io(tamper));(tamper/'anchor.joblib').write_bytes(b'changed');expect_error(lambda:load_bundle(tamper,protocol,'unit-split'),'Bundle checksum')
 # A different frozen anchor cannot receive residuals trained against the first.
 a2=copy.deepcopy(a);a2.affine['slope']+=.1;path2=folder/'anchor2.joblib';binding2=freeze_anchor(a2,path2,protocol);expect_error(lambda:save_bundle(folder/'swapped',path2,binding2,model,f,B,protocol,'unit-split'),'different anchor')
 expect_error(training_guard,'implementation/CPU-unit-test only')
 xx=np.arange(16,dtype=np.float32).reshape(2,1,2,4);rr=np.array([0.,2.]);stats=(float(rr.mean()),max(float(rr.std()),1e-6));require(cnn_normalization(xx,rr,True,stats)==cnn_normalization(xx,rr[::-1],True,stats),'Residual arms normalization mismatch');require(cnn_normalization(xx,np.zeros(2),True,(0.,1e-6))[-1]==1e-6,'Zero residual handling');expect_error(lambda:cnn_normalization(xx,rr,True),'shared true-B normalization')
 expect_error(lambda:evaluate_all(None,None,None,None,None,None,[0]),'All five folds')
 # Each classical inner learner is unaffected by its own validation labels. Selection itself may use them.
 D=np.array(g.cells[0]['D']);rr=y[D]-a.predict(f,D);ti,vi=next(GroupKFold(3,shuffle=True,random_state=11).split(D,groups=f.groups[D]))
 for kind in residual_models.KINDS:
  m=residual_models.fit(f.take('X',D[ti]),rr[ti],f.blocks,kind);p0=residual_models.predict(m,f.take('X',D[vi]));poison=rr.copy();poison[vi]=np.nan;m2=residual_models.fit(f.take('X',D[ti]),poison[ti],f.blocks,kind);np.testing.assert_array_equal(p0,residual_models.predict(m2,f.take('X',D[vi])))
 return {'passed':True,'synthetic_complete_anchors_fit':3,'synthetic_nonnull_classical_fits':152,'synthetic_backend_calls':len(backend.calls)*3,'finite_and_nan_outer_poison_exact':True,'B_and_V_anchor_poison_exact':True,'bundle_reload_exact':True,'swap_split_tamper_rejected':True,'main_training_guard_rejected':True,'no_competition_model_fits':True}

def real_contract_checks(f,labels,contract):
 for k,g in enumerate(contract['graphs']):
  f.partition(np.arange(400),g['T'],g['V']);f.partition(g['T'],g['A'],g['B']);graph_for(k).validate(f)
 backend=ProductionBackend(R,contract['registry'],'cpu');errors=[];cnnerrors=[];shorterrors=[];count=0
 for e in contract['registry']:
  lab=labels.restrict(e['train']);require(backend.entry(lab,e['offset'])==e,'Registry lookup')
  with np.load(_repo_path(R, 'experiments/cnn-nested-204/cache', f"{e['cache_name']}.npz"),allow_pickle=False) as z:Q=z['validation'][:2];expected=z['P'][:2].copy();expected[:,2]=z['oldseeds'][:,:2].mean(0);expected=expected@np.array([0,.3,.3,.2,.2]);cnnexpected=z['oldseeds'][:,:2].mean(0);sexpected=z['short'][:2] if 'short' in z else None
  model=backend.cached_base(f,lab,e);pred=model.predict(f,Q);err=float(abs(pred-expected).max());require(err<2e-4,'Cached base replay mismatch');errors.append(err);ce=float(abs(model.cnn.predict(f,Q)-cnnexpected).max());require(ce<2e-4,'Cached CNN replay mismatch');cnnerrors.append(ce)
  if 'short' in e['files']:
   small=backend.short(f,lab,e['offset']);sp=small.predict(f,Q);err=float(abs(sp-sexpected).max());require(err<2e-4,'Cached short replay mismatch');shorterrors.append(err)
  count+=1
 bad=labels.restrict(contract['registry'][0]['train']);expect_error(lambda:backend.entry(Labels(bad.ids,bad.values+1),contract['registry'][0]['offset']),'target membership/version mismatch')
 # TinyCNN forward/backward only: no optimizer, no model update, no validation labels.
 B=contract['graphs'][0]['B'];x=f.take('images',B[:2]);torch.manual_seed(20264000);m=TinyCNN();before={k:v.detach().clone() for k,v in m.state_dict().items()};xx=torch.tensor((x-x.mean())/max(float(x.std()),1e-6));out=m(xx);loss=(out-torch.tensor([-.5,.5])).square().mean();loss.backward();require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),'Bad CNN gradients');require(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()),'Unit check changed weights');params=sum(p.numel() for p in m.parameters());require(params==91905,'Architecture changed')
 return {'passed':True,'outer_graphs':5,'deep_nodes':45,'cached_base_nodes_replayed':count,'CPU_rows_per_node':2,'base_replay_max_abs_error':max(errors),'cnn_replay_max_abs_error':max(cnnerrors),'short_replay_max_abs_error':max(shorterrors),'tolerance':2e-4,'TinyCNN_parameters':params,'CPU_forward_backward_batches':1,'optimizer_steps':0,'competition_model_fits':0}

def regression204(f,contract):
 with np.load(_repo_path(R, 'experiments/cnn-nested-204/cache/data.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in ['y','R','short','L','fold','groups']}
 records=[];full_models=0
 for si,split in enumerate(['original','20261031','20261101']):
  for k in range(5):
   p=_repo_path(R, 'experiments/cnn-nested-204/R', split, f'fold-{k}');fr=json.loads((p/'report.json').read_text());require(sha(p/'predictions.npz')==fr['predictions_sha256'],'204 fold hash')
   with np.load(_repo_io(p/'predictions.npz'),allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
   T,V=a['T'],a['V'];cal=coefficient(a['raw_middle'],d['y'][T]);base=.9*apply(cal,a['raw_outer'])+.1*d['short'][si,V];np.testing.assert_allclose(base,a['base'],atol=1e-9,rtol=0);corrections=[]
   for j in range(3):
    bpath=p/f'block-{j}';br=json.loads((bpath/'report.json').read_text());require(sha(bpath/'report.json')==fr['block_reports_sha256'][f'block-{j}/report.json'],'204 block report hash');require(sha(bpath/'predictions.npz')==br['predictions_sha256'],'204 block predictions hash')
    with np.load(_repo_io(bpath/'predictions.npz'),allow_pickle=False) as z:b={key:z[key].copy() for key in z.files}
    A,B=b['A'],b['B_hold'];cal=coefficient(b['raw_deep'],d['y'][A]);assert cal==br['anchor_calibrator'];np.testing.assert_array_equal(b['residual'],d['y'][B]-b['anchor']);loss=(b['inner_predictions']-b['residual'])**2;w,draw,wins=bootstrap_weights(loss,d['groups'][B],br['bootstrap_seed']);np.testing.assert_array_equal(w,b['weights']);np.testing.assert_array_equal(draw,b['draws']);np.testing.assert_array_equal(wins,b['winners']);full=[]
    for kind in residual_models.KINDS:
     name=f'full-{kind}.joblib';require(sha(bpath/name)==br['models_sha256'][name],'204 saved residual model hash');obj=joblib.load(_repo_io(bpath/name));np.testing.assert_array_equal(obj['train'],B);pred=residual_models.predict(obj['model'],f.take('X',V));full.append(pred);full_models+=1
    full=np.stack(full);np.testing.assert_array_equal(full,b['full_predictions']);corr=w@full;np.testing.assert_array_equal(corr,b['correction']);corrections.append(corr)
   primary=.9*(base+np.mean(corrections,axis=0))+.1*d['L'][si,V];np.testing.assert_allclose(primary,a['primary'],atol=1e-9,rtol=0);err=float(abs(primary-d['R'][si,V]).max());require(err<1e-9,'204 reference changed');records.append({'split':split,'fold':k,'max_abs_error':err})
 return {'passed':True,'folds_replayed':len(records),'saved_full_residual_models_replayed':full_models,'bootstrap_blocks_replayed':45,'max_abs_error':max(v['max_abs_error'] for v in records),'checks':records,'new_fits':0,'limit':'Saved204 inference/arithmetic regression only, not end-to-end204 retraining'}

def main():
 start=time.perf_counter();suffix=time.strftime('%Y%m%dT%H%M%S');folder=_repo_path(O, 'unit-tmp', suffix);folder.mkdir();result={'stage':'CPU_UNIT_ONLY','started':suffix};code={str(p.relative_to(O)):sha(p) for p in O.glob('*.py')};lock_manifest(folder/'code-lock.json',{'sources':code,'protocol_sha256':sha(_repo_path(O, 'protocol.json')),'data_contract_sha256':sha(_repo_path(O, 'data-contract.json'))})
 # Restrict writes and avoid all project data outside planned development/cache/model files.
 contract=json.loads((_repo_path(O, 'data-contract.json')).read_text());reads=set()
 def guard(event,args):
  if event.startswith('socket.'):raise RuntimeError('No network in219 unit checks')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).absolute();mode=args[1];flags=args[2];write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
   if write and not p.is_relative_to(O):raise RuntimeError('219-only write guard')
   if p.is_relative_to(R):
    rel=str(p.relative_to(R));allowed=p.is_relative_to(O) or p.is_relative_to(_repo_path(R, '.venv')) or rel in contract['inputs'] or rel in contract['protected'] or p.is_relative_to(_repo_path(R, 'experiments/cnn-nested-204/R')) or any(rel==f['path'] for e in contract['registry'] for f in e['files'].values())
    if not allowed:raise RuntimeError('Unplanned219 project read:'+rel)
    reads.add(rel)
   elif str(p).startswith('/Users/'):raise RuntimeError('Other personal directory forbidden')
 sys.addaudithook(guard)
 try:
  result['isolation']=isolation_checks(folder);print(json.dumps({'stage':'fixture_isolation_passed','seconds':time.perf_counter()-start}),flush=True)
  f,labels,contract=load_development();result['contracts']=real_contract_checks(f,labels,contract);print(json.dumps({'stage':'real_cache_and_graph_passed',**result['contracts']}),flush=True)
  result['regression204']=regression204(f,contract);result['protected']={rel:sha(_repo_path(R, rel))==digest for rel,digest in contract['protected'].items()};require(all(result['protected'].values()),'Protected artifact changed');require(all(sha(_repo_path(O, p))==v for p,v in code.items()),'Source changed during tests');result['passed']=True
 except Exception as e:result.update(passed=False,error=repr(e),traceback=traceback.format_exc());raise
 finally:
  result['seconds']=time.perf_counter()-start;result['RSS_peak_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result['threadpools']=threadpool_info();result['torch_threads']=torch.get_num_threads();result['MPS_updates']=0;result['CPU_neural_optimizer_steps']=0;result['main_training_started']=False;result['read_files']=sorted(reads);lock_manifest(_repo_path(O, 'results', f'unit-{suffix}.json'),result);print(json.dumps({'passed':result.get('passed'), 'seconds':result['seconds'],'result':str(_repo_path(O, 'results', f'unit-{suffix}.json'))}),flush=True)
if __name__=='__main__':main()
