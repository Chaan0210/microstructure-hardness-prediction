"""Independent CPU1 post-completion verification. No training and no scoring.
Run --self-test before completion; --run only after parent confirms all five folds.
All output and writes are confined to this directory. Never import run.py.
"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os,sys,json,hashlib,time,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
ROOT=_RepoPath(__file__).resolve().parents[4]
MAIN=_repo_path(SOURCE, 'main-20261007')
assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv'),'Project .venv required'
sys.path.insert(0,str(SOURCE))
READS=set()
CONTRACT_META=json.loads((_repo_path(SOURCE, 'data-contract.json')).read_text())
INPUTS=CONTRACT_META['inputs']
REGISTRY_FILES={entry['path'] for node in CONTRACT_META['registry'] for entry in node['files'].values()}
def audit(event,args):
    if event.startswith('socket.') or event in ('subprocess.Popen','os.system'):
        raise RuntimeError('Independent verifier forbids network and subprocesses')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        wr=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if wr and not p.is_relative_to(HERE):raise RuntimeError('Verifier-only writes: '+str(p))
        if p.is_relative_to(ROOT):
            rel=str(p.relative_to(ROOT))
            if not (p.is_relative_to(SOURCE) or p.is_relative_to(_repo_path(ROOT, '.venv')) or rel in INPUTS or rel in REGISTRY_FILES):
                raise RuntimeError('Unplanned project read: '+rel)
            READS.add(rel)
        elif str(p).startswith('/Users/'):
            raise RuntimeError('Other personal files forbidden: '+str(p))
sys.addaudithook(audit)
import numpy as np,torch,joblib
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(1);torch.set_num_threads(1)
from contracts import sha
from anchor import CompleteAnchor
from bundle import load_bundle

def require(condition,message):
    if not condition:raise AssertionError(message)
def readj(path):return json.loads(path.read_text())
def exact(a,b,message):
    require(np.array_equal(np.asarray(a),np.asarray(b)),message)
def near(a,b,message,tol=1e-8):
    np.testing.assert_allclose(a,b,rtol=0,atol=tol,err_msg=message)
def independent_affine(raw,y,query):
    raw=np.asarray(raw,dtype=float);y=np.asarray(y,dtype=float)
    coef=np.linalg.lstsq(np.column_stack((np.ones(len(raw)),raw)),y,rcond=None)[0]
    return coef[0]+coef[1]*np.asarray(query,dtype=float),coef

def self_test():
    class Fixed:
        def __init__(self,v):self.v=np.asarray(v,dtype=float)
        def predict(self,f,q):return self.v[q]
    raw=np.array([2.,4.,8.]);short=np.array([10.,20.,30.]);long=np.array([7.,11.,19.])
    blocks=[np.array([1.,2.,3.]),np.array([4.,5.,6.]),np.array([-2.,7.,1.])]
    a=CompleteAnchor(Fixed(raw),Fixed(short),Fixed(long),{'prediction_center':3.,'target_center':5.,'slope':2.,'intercept':-1.},[Fixed(x) for x in blocks],(),{})
    near(a.predict(None,np.arange(3)),.81*(2*raw-1)+.09*short+.3*sum(blocks)+.1*long,'distinct-component formula',1e-14)
    prediction,coef=independent_affine(raw,2*raw-1,np.array([3.,5.]))
    near(prediction,[5.,9.],'independent affine fixture',1e-13)
    from evaluation import evaluate_all
    try:evaluate_all(None,None,None,None,None,None,[0,1,2,3])
    except ValueError as ex:require('All five folds' in str(ex),'Wrong partial-fold rejection')
    else:raise AssertionError('Missing partial-fold gate')
    return {'distinct_component_formula':True,'independent_lstsq_affine':True,'partial_fold_gate_rejected':True,'real_labels_read':False,'real_predictions_read':False}

def check_sources():
    lock=readj(_repo_path(SOURCE, 'code-lock.json'));mainlock=readj(_repo_path(MAIN, 'main-code-lock.json'))
    require(sha(_repo_path(SOURCE, 'code-lock.json'))==mainlock['original219_code_lock_sha256'],'Original code-lock changed')
    require(sha(_repo_path(MAIN, 'authorized-source-fix.json'))==mainlock['authorized_source_fix_sha256'],'Authorization fix record changed')
    fixes=readj(_repo_path(MAIN, 'authorized-source-fix.json'))
    require(set(fixes)=={'evaluation.py'},'Unexpected authorized source changes')
    for name,fix in fixes.items():
        require(lock['sources'][name]==fix['old_sha256'],'Source fix original mismatch')
        require(sha(_repo_path(MAIN, fix['snapshot']))==fix['old_sha256'],'Original source snapshot mismatch')
        lock['sources'][name]=fix['new_sha256']
    for name,digest in lock['sources'].items():require(sha(_repo_path(SOURCE, name))==digest,'Source hash: '+name)
    for name,digest in mainlock['main_sources'].items():require(sha(_repo_path(MAIN, name))==digest,'Runner source hash: '+name)
    for name,key in [('protocol.json','protocol_sha256'),('data-contract.json','data_contract_sha256')]:
        require(sha(_repo_path(SOURCE, name))==mainlock[key],'Contract hash: '+name)
    return {'source_count':len(lock['sources']),'authorized_fix':'evaluation.py only','main_sources':mainlock['main_sources']}

def readiness():
    # Reject incompleteness before loading development labels or prediction arrays.
    required=[_repo_path(MAIN, 'fit-completion.json'),_repo_path(MAIN, 'CNN-records.json'),_repo_path(MAIN, 'oof-unscored.npz'),_repo_path(MAIN, 'verification/validation.json')]
    for k in range(5):
        folder=_repo_path(MAIN, f'fold-{k}')
        required.extend(folder/name for name in ('validation.json','training-report.json','frozen-anchor.json','frozen-anchor.joblib','predictions.npz'))
        for arm in ('true','permuted'):
            required.extend(folder/arm/name for name in ('manifest.json','anchor.joblib','residual.joblib'))
    require(all(p.is_file() for p in required),'All five completed packages and fit-completion required; no partial evaluation')
    completion=readj(_repo_path(MAIN, 'fit-completion.json'))
    require(completion['passed'] and completion['original400_only'] and not completion['forbidden_reads'],'Fit completion not valid')
    require(readj(_repo_path(MAIN, 'verification/validation.json'))['passed'],'Main poison/replay checks incomplete')
    require(sha(_repo_path(MAIN, 'verification/validation.json'))==completion['poison_validation_sha256'],'Poison validation hash')
    require(sha(_repo_path(MAIN, 'oof-unscored.npz'))==completion['oof_sha256'],'OOF hash')
    expected={'main_CNN':145,'main_CNN_updates':35600,'verification_CNN':34,'verification_CNN_updates':8480}
    for key,value in expected.items():require(completion['counts'][key]==value,'Incomplete fits: '+key)
    for k in range(5):require(sha(_repo_path(MAIN, f'fold-{k}', 'validation.json'))==completion['fold_validation_sha256'][str(k)],'Fold validation hash')
    return completion,{str(p.relative_to(MAIN)):sha(p) for p in required}

def validate_block(block,record,cell,features,y,offset,ell):
    C=np.asarray(cell['C']);D=np.asarray(cell['D'])
    exact(block.train,D,'Classical train D');exact(record['C'],C,'Stored C');exact(record['D'],D,'Stored D')
    _,coef=independent_affine(record['deep_oof'],y[C],record['deep_oof'])
    near([record['calibrator']['intercept'],record['calibrator']['slope']],coef,'Cell affine coefficient')
    exact(record['residual_D'],y[D]-record['anchor_D'],'Cell residual arithmetic')
    br=block.record
    require(br['split_seed']==20271013+100*offset+ell and br['bootstrap_seed']==20271204+100*offset+ell,'Residual selector seeds')
    exact(br['loss'],(br['inner_predictions']-record['residual_D'])**2,'Residual OOF loss')
    from sklearn.model_selection import GroupKFold
    splits=list(GroupKFold(3,shuffle=True,random_state=br['split_seed']).split(D,groups=features.groups[D]))
    for split,(ti,vi) in zip(br['splits'],splits):
        exact(split['train'],D[ti],'Residual inner train');exact(split['validation'],D[vi],'Residual inner validation')
    groups=features.groups[D];unique=np.unique(groups)
    draws=np.random.default_rng(br['bootstrap_seed']).integers(0,len(unique),(512,len(unique)))
    exact(br['draws'],draws,'Independent bootstrap draws')
    # Sum loss by group and resample groups, independently of the original counts implementation.
    grouped=np.column_stack([br['loss'][:,groups==g].sum(axis=1) for g in unique])
    winners=np.array([np.argmin(grouped[:,row].sum(axis=1)) for row in draws])
    exact(br['winners'],winners,'Independent bootstrap winners')
    exact(block.weights,np.array([(winners==j).mean() for j in range(5)]),'Independent bootstrap weights')
    require(len(block.models)==5 and np.isfinite(block.weights).all() and block.weights.sum()==1,'Classical candidates')

def replay_nested(k,a,f,y,graph,contract,report):
    from contracts import Labels
    from backend import ProductionBackend
    backend=ProductionBackend(ROOT,contract['registry'],'cpu')
    events={event['artifact']:event for event in report['backend_events'] if 'artifact' in event}
    def retained(kind,U,offset):
        key=hashlib.sha256(np.asarray(U,dtype=np.int64).tobytes()).hexdigest()[:16]
        path=_repo_path(MAIN, f'nodes-fold-{k}', f'{kind}-{offset}-{key}.joblib')
        require(str(path) in events,'Retained node event missing')
        event=events[str(path)];require(sha(path)==event['sha256'],'Retained node hash')
        obj=joblib.load(_repo_io(path));exact(obj['train'],U,'Retained training indices');exact(obj['model'].train,U,'Retained model training indices')
        require(obj['target_sha256']==hashlib.sha256(y[U].tobytes()).hexdigest(),'Retained target hash')
        return obj['model']
    A=np.asarray(graph['A']);whereA={int(i):j for j,i in enumerate(A)};raw_oof=np.full(len(A),np.nan)
    for cell,record in zip(graph['cells'],a.record['cells']):
        C=np.asarray(cell['C']);D=np.asarray(cell['D']);lab=Labels(C,y[C]);entry=backend.entry(lab,cell['offset'])
        require(entry is not None,'Expected cached middle base')
        middle=backend.cached_base(f,lab,entry);raw=middle.predict(f,D)
        if 'short' in entry['files']:small=backend.short(f,lab,cell['offset'])
        else:small=retained('short',C,cell['offset'])
        deep=np.full(len(C),np.nan);whereC={int(i):j for j,i in enumerate(C)}
        for node in cell['deep']:
            E=np.asarray(node['E']);F=np.asarray(node['F']);model=retained('base',E,node['offset'])
            deep[[whereC[int(i)] for i in F]]=model.predict(f,F)
        exact(deep,record['deep_oof'],'Fresh deep E/F OOF predictions')
        calibrated,_=independent_affine(deep,y[C],raw)
        near(.9*calibrated+.1*small.predict(f,D),record['anchor_D'],'Independent nested honest anchor equation')
        raw_oof[[whereA[int(i)] for i in D]]=raw
    exact(raw_oof,a.record['raw_oof'],'Fresh middle C/D root OOF predictions')
    return {'deep_nodes_replayed':9,'middle_nodes_replayed':3,'nested_honest_anchor_equations':True}

def validate_fold(k,f,y,contract,cnn_records):
    graph=contract['graphs'][k];A,B,T,V=(np.asarray(graph[x],dtype=np.int64) for x in ('A','B','T','V'))
    require(tuple(map(len,(A,B,T,V)))==(213,107,320,80),'Fold sample counts')
    require(set(A).isdisjoint(B) and set(A)|set(B)==set(T),'A/B partition')
    require(set(T).isdisjoint(V) and set(T)|set(V)==set(range(400)),'T/V partition')
    require(not set(f.groups[A])&set(f.groups[B]) and not set(f.groups[T])&set(f.groups[V]),'Group overlap')
    folder=_repo_path(MAIN, f'fold-{k}');validation=readj(folder/'validation.json');report=readj(folder/'training-report.json');binding=readj(folder/'frozen-anchor.json')
    require(validation['passed'] and validation['reload_exact'],'Internal validation not passed')
    exact(validation['V'],V,'Validation V');exact(binding['train'],A,'Binding train')
    for filename,key in [('frozen-anchor.joblib','anchor_sha256'),('predictions.npz','predictions_sha256'),('training-report.json','training_report_sha256')]:
        require(sha(folder/filename)==validation[key],'Validation artifact hash: '+filename)
    digest=sha(folder/'frozen-anchor.joblib');protocol=sha(_repo_path(SOURCE, 'protocol.json'));split=sha(_repo_path(SOURCE, 'data-contract.json'))
    require(binding['anchor_sha256']==digest and binding['protocol_sha256']==protocol,'Frozen binding')
    require(report['protocol_sha256']==protocol and report['split_sha256']==split and report['anchor_binding']==binding,'Training report binding')
    require(report['outer']==k and report['predictions_sha256']==sha(folder/'predictions.npz'),'Training report prediction hash')
    a=joblib.load(_repo_io(folder/'frozen-anchor.joblib'));exact(a.train,A,'Anchor train indices')
    for child in (a.root,a.short,a.L):exact(child.train,A,'Anchor component training indices')
    require(len(a.residuals)==3 and len(a.record['cells'])==3,'Complete classical correction structure')
    nested=replay_nested(k,a,f,y,graph,contract,report)
    for ell,(block,record,cell) in enumerate(zip(a.residuals,a.record['cells'],graph['cells'])):
        validate_block(block,record,cell,f,y,graph['offset'],ell)
    pB=a.predict(f,B);pV=a.predict(f,V);raw=a.root.predict(f,V)
    calibrated,coef=independent_affine(a.record['raw_oof'],y[A],raw)
    near([a.affine['intercept'],a.affine['slope']],coef,'Root independent affine coefficients')
    base=.9*calibrated+.1*a.short.predict(f,V)
    correction=np.mean([m.predict(f,V) for m in a.residuals],axis=0)
    manual=.9*(base+correction)+.1*a.L.predict(f,V)
    near(manual,pV,'Independent complete anchor algebra')
    exact(pV+np.zeros(len(V)),pV,'Zero correction equivalence')
    residual=y[B]-pB;X=f.take('images',B)
    norm={'x_mean':float(X.mean()),'x_std':float(X.std()),'y_mean':float(residual.mean()),'y_std':max(float(residual.std()),1e-6)}
    exact(report['residual_shared_normalization'],[norm['y_mean'],norm['y_std']],'Shared true B target stats')
    permutation=np.random.default_rng(20264050+k).permutation(len(B));exact(report['permutation'],permutation,'Locked residual permutation')
    with np.load(_repo_io(folder/'predictions.npz'),allow_pickle=False) as z:saved={name:z[name].copy() for name in z.files}
    exact(saved['V'],V,'Prediction exact V');exact(saved['R0'],pV,'Fresh CPU1 anchor replay')
    for arm,key in [('true','R1'),('permuted','Rperm')]:
        pkg=load_bundle(folder/arm,protocol,split);manifest=pkg.manifest;r=pkg.residual
        require(sha(folder/arm/'manifest.json')==validation['bundles'][arm],'Manifest locked hash')
        require(manifest['anchor_sha256']==digest and r.anchor_sha256==digest,'Cross-arm anchor SHA linkage')
        exact(manifest['anchor_train'],A,'Manifest A');exact(manifest['residual_train'],B,'Manifest B')
        exact(manifest['anchor_ids'],f.ids[A],'Persisted anchor sample IDs');exact(manifest['residual_ids'],f.ids[B],'Persisted residual sample IDs')
        exact(r.train,B,'Residual train');require(r.kind=='CNN' and r.seed==20264000+k,'Residual architecture/seed')
        require(r.normalization==norm,'CNN normalization must be from true B residual and B images')
        require(len(report['curves'][arm])==80 and [x['epoch'] for x in report['curves'][arm]]==list(range(1,81)),'Fixed80 epochs')
        exact(pkg.anchor.predict(f,B),pB,'Bundle anchor B identical');exact(pkg.anchor.predict(f,V),pV,'Bundle anchor V identical')
        exact(pkg.predict(f,V),saved[key],'Fresh CPU1 bundle replay: '+arm)
        exact(pV+r.predict(f,V),saved[key],'Signed unscaled correction: '+arm)
        target=residual if arm=='true' else residual[permutation]
        targethash=hashlib.sha256(target.tobytes()).hexdigest()
        recs=[rec for rec in cnn_records if rec['phase']=='main' and rec['residual'] and rec['seed']==20264000+k and rec['target_sha256']==targethash]
        require(len(recs)==1,'Unique residual CNN target record')
        rec=recs[0];artifact=_repo_path(MAIN, rec['path']);require(sha(artifact)==rec['sha256'],'Residual CNN training artifact hash')
        obj=joblib.load(_repo_io(artifact));meta=obj['meta'];exact(meta['train'],B,'CNN persisted train');exact(meta['train_ids'],f.ids[B],'CNN persisted sample IDs')
        require(meta['normalization']==norm,'CNN training artifact normalization')
        for name,value in r.state.items():require(torch.equal(value,obj['model'].state[name]),'CNN artifact state equality')
    print(json.dumps({'fold':k,'independent_validation_passed':True,'scores_computed':False}),flush=True)
    return {'fold':k,'V':V.tolist(),'anchor_sha256':digest,'fresh_CPU1_predictions_exact':True,'manual_anchor_max_error':float(np.max(np.abs(manual-pV))),'normalization_true_B_exact':True,'sample_ID_mapping_exact':True,'bootstrap_blocks':3,'nested':nested},saved

def run():
    completion,before=readiness();source=check_sources()
    from data import load_development
    f,labels,contract=load_development();y=labels.values
    require(len(y)==400,'Development400 labels only')
    records=readj(_repo_path(MAIN, 'CNN-records.json'));reports=[];seen=[];allpred={name:np.full(400,np.nan) for name in ('R0','R1','Rperm')};fold=np.full(400,-1)
    for k in range(5):
        report,pred=validate_fold(k,f,y,contract,records);reports.append(report);V=np.array(report['V']);seen.extend(V);fold[V]=k
        for name in allpred:allpred[name][V]=pred[name]
    exact(sorted(seen),np.arange(400),'All five V sets exactly cover400 once')
    with np.load(_repo_path(MAIN, 'oof-unscored.npz'),allow_pickle=False) as z:
        exact(z['ids'],f.ids,'OOF sample IDs');exact(z['fold'],fold,'OOF fold assignments')
        for name in allpred:require(np.isfinite(allpred[name]).all(),'Finite OOF');exact(z[name],allpred[name],'OOF assembled prediction '+name)
    require(all(sha(_repo_path(MAIN, rel))==digest for rel,digest in before.items()),'Artifact changed during independent verification')
    check_sources()
    return {'source_audit':source,'folds':reports,'complete_unique400':True,'all_artifact_hashes_unchanged':True,'main_completion_sha256':sha(_repo_path(MAIN, 'fit-completion.json')),'scores_computed':False,'labels_scope':'development400 only','main_and_verification_counts':completion['counts']}

def main():
    require(sys.argv[1:] in (['--self-test'],['--run']),'Use --self-test or --run')
    mode=sys.argv[1][2:];start=time.perf_counter();result={'mode':mode,'passed':False,'script_sha256':sha(Path(__file__))}
    try:
        result['checks']=self_test() if mode=='self-test' else run();result['passed']=True
    except Exception as ex:
        result['error']=repr(ex);result['traceback']=traceback.format_exc()
    finally:
        result.update(seconds=time.perf_counter()-start,torch_threads=torch.get_num_threads(),threadpools=threadpool_info(),read_files=sorted(READS),training_steps=0,MPS_calls=0,scores_computed=False)
        suffix=time.strftime('%Y%m%dT%H%M%S')
        path=_repo_path(HERE, f'{mode}-{suffix}.json')
        with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        if mode=='run':
            with (_repo_path(HERE, 'results.json')).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({'passed':result['passed'],'mode':mode,'report':str(path),'seconds':result['seconds'],'error':result.get('error')}),flush=True)
    if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
