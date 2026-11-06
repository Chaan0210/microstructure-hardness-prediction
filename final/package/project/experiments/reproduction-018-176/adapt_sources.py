from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;R=D.parents[1]
def replace_function(text,name,replacement):
 a=ast.parse(text);n=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name);lines=text.splitlines(keepends=True);return ''.join(lines[:n.lineno-1])+replacement+'\n\n'+''.join(lines[n.end_lineno:])
p=D/'base_train.py';s=p.read_text();s=s.replace('def kernel_fold(X, y, groups, blocks, tr, va, fold):','def kernel_fold(X, y, groups, blocks, tr, va, fold, selection=None):\n    selection = tr if selection is None else np.asarray(selection)')
a=ast.parse(s);n=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fold');seg=ast.get_source_segment(s,n);part1,part2=seg.split('    chosen = min(scores',1);part1=part1.replace('X[tr]','X[selection]').replace('y[tr]','y[selection]').replace('groups[tr]','groups[selection]').replace('len(tr)','len(selection)').replace('tr[ti]','selection[ti]').replace('tr[vi]','selection[vi]');s=s.replace(seg,part1+'    chosen = min(scores'+part2);p.write_text(s)
p=D/'train.py';s=p.read_text();s=s.replace('Fresh018 training using only local dev400 input; never loads historic learners.','Fresh018 full500 reproduction; no historic fitted learners loaded.')
s=replace_function(s,'input_data', '''def input_data():
 meta=read(OUT/'train-feature-plan.json');v=read(OUT/'train-feature-validation.json');assert v['passed'] and sha(OUT/'train-features.npz')==v['feature_sha256']
 with np.load(OUT/'train-features.npz',allow_pickle=False) as z:d={name:z[name].copy() for name in z.files}
 assert d['ids'].tolist()==meta['IDs'] and len(d['ids'])==500
 schema=read(ROOT/'config/schema.json');d['Kblocks']=schema['compact_blocks'];d['blocks']=schema['physics_blocks']+['joint_token']*528
 assert len(d['selection'])==400 and d['ids'][d['selection']].tolist()==read(ROOT/'config/development-ids.json');return d''')
s=replace_function(s,'prepare', '''def prepare():
 d=input_data();recipe=read(ROOT/'config/recipe.json');weights=read(ROOT/'licenses/current-model-provenance.json')
 for w in weights['weights']:assert sha(ROOT/w['path'])==w['sha256']
 sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in str(p) and 'run/' not in str(p.relative_to(ROOT))}
 plan={'study':176,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Reproduce historic018 full500 finaltraining; no subgroup holdout evaluation or new selection','features':'Fresh500rawfeatures; reuse official frozen pretrained DINO weights only. No fitted regression/calibration/neural/residual artifacts reused.','feature_sha256':sha(OUT/'train-features.npz'),'feature_plan_sha256':sha(OUT/'train-feature-plan.json'),'sources':sources,'config':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'config').glob('*.json')},'ids':d['ids'].tolist(),'groups':d['g'].tolist(),'selection_ids':d['ids'][d['selection']].tolist(),'target_array_sha256':hashlib.sha256(d['y'].tobytes()).hexdigest(),'recipe':recipe,'tolerances':recipe['tolerances'],'fresh_counts':{'base_nodes':13,'CNN':39,'long_MLP':78,'short_B':4,'KRR_final':13,'KRR_inner':351,'affines':4,'residual_objects':60,'residual_nonnull':48},'test':'Only inference after training validation. Test CSV values ignored; fitting guard during inference.','resource':{'CPU_threads':2,'RSS_cap_bytes':4000000000,'MPS_driver_cap_bytes':2000000000,'estimated_train_and_validation_minutes':[18,30]},'root_query':'Root query=train500 only for reload/inference comparison. Never used for calibration or residual targets. No root in-sample performance reported as validation.'}
 write(OUT/'plan.json',plan)
 import importlib.metadata,platform
 write(OUT/'environment.json',{'python':sys.version,'machine':platform.machine(),'packages':{dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},'torch_threads':torch.get_num_threads(),'MPS_available':torch.backends.mps.is_available()})
 emit({'PLAN_LOCKED':sha(OUT/'plan.json')})''')
s=replace_function(s,'check', '''def check():
 p=read(OUT/'plan.json')
 for name,hsh in {**p['sources'],**p['config']}.items():assert sha(ROOT/name)==hsh,name
 assert sha(OUT/'train-features.npz')==p['feature_sha256'];return p''')
old="assert not set(U)&set(Q) and not set(d['g'][U])&set(d['g'][Q]);pred={};logs={};ids=d['ids'].tolist();y=d['y']"
new="root_fit=path.name=='outer' and len(U)==500 and np.array_equal(U,Q);assert root_fit or (not set(U)&set(Q) and not set(d['g'][U])&set(d['g'][Q]));pred={};logs={};ids=d['ids'].tolist();y=d['y']"
assert old in s;s=s.replace(old,new);s=s.replace("d['Kblocks'],U,Q,offset)","d['Kblocks'],U,Q,offset,selection=d['selection'] if root_fit else U)")
s=s.replace("'with_short':with_short,'all_learners_fresh':True", "'with_short':with_short,'KRR_selection':(d['selection'] if root_fit else U).tolist(),'root_in_sample_query':root_fit,'all_learners_fresh':True")
s=replace_function(s,'fold', '''def fit_full(d):
 root=OUT/'full'
 if (root/'report.json').exists():return read(root/'report.json')
 start=time.perf_counter();T=np.arange(500);outer=node(d,T,T,0,root/'outer',True);middle=np.full((500,5),np.nan);cells=[]
 splits=list(GroupKFold(3,shuffle=True,random_state=20270110).split(d['S'],d['y'],d['g']))
 for j,(A,B) in enumerate(splits):
  offset=100*(j+1);mid=node(d,A,B,offset,root/f'middle-{j}',True);middle[B]=mid['P'];deep=np.full((len(A),5),np.nan)
  inner=list(GroupKFold(3,shuffle=True,random_state=20270110+offset).split(d['S'][A],d['y'][A],d['g'][A]))
  for ell,(C,D) in enumerate(inner):deep[D]=node(d,A[C],A[D],offset+100*(ell+1),root/f'deep-{j}-{ell}')['P']
  anchor,cal=affine(mixture(deep),d['y'][A],mixture(mid['P']));anchor=.9*anchor+.1*mid['short'];cells.append((A,B,anchor,cal,deep))
 assert np.isfinite(middle).all();cp,cal=affine(mixture(middle),d['y'],mixture(outer['P']));base=.9*cp+.1*outer['short'];cor=[]
 for j,(A,B,anchor,ac,deep) in enumerate(cells):
  cor.append(residual(d,B,anchor,T,0,j,root/f'residual-{j}'));arrays(root/f'anchor-{j}.npz',A=A,B=B,deep=deep,anchor=anchor)
 pred=base+np.mean(cor,0);assert np.isfinite(pred).all();arrays(root/'predictions.npz',T=T,V=T,middle=middle,outer=outer['P'],raw=mixture(outer['P']),short=outer['short'],base=base,corrections=np.stack(cor),primary=pred)
 r={'rows':500,'calibrator':cal,'anchor_calibrators':[v[3] for v in cells],'seconds':time.perf_counter()-start,'memory':cap(),'fresh_fit_counts':read(OUT/'plan.json')['fresh_counts'],'prediction_sha256':sha(root/'predictions.npz'),'node_reports':{str(p.relative_to(root)):sha(p) for p in root.glob('*/report.json')},'no_holdout_subgroup_metrics':True,'test_read':False};write(root/'report.json',r);emit({'FULL500_COMPLETE':{k:v for k,v in r.items() if k not in ['node_reports']}});return r''')
s=replace_function(s,'run', '''def run():
 check();d=input_data();lock=OUT/'training.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 touched=filesystem_guard();start=time.perf_counter();trace=[];stop=threading.Event()
 def watch():
  while not stop.wait(1):trace.append({'seconds':time.perf_counter()-start,**memory()})
 worker=threading.Thread(target=watch,daemon=True);worker.start()
 try:fit_full(d);assert time.perf_counter()-start<1800;cap()
 finally:
  stop.set();worker.join();write(OUT/f'execution-{os.getpid()}.json',{'seconds':time.perf_counter()-start,'file_accesses':sorted(touched),'resource_trace':trace,'historical_project_files_blocked_by_audit_hook':True});lock.unlink()''')
s=s[:s.index("if __name__=='__main__':")]+"if __name__=='__main__':\n if sys.argv[1]=='prepare':prepare()\n elif sys.argv[1]=='fit':run()\n else:raise ValueError('Use prepare or fit')\n";p.write_text(s)
old=json.loads((R/'experiments/reproduction-018-175/run/plan.json').read_text());recipe={'model':'018','fixed_before176':True,'CNN_seeds':[20261003,20261023,20261043],'head_seeds':[20261128,20261205,20261219],'head_config':old['head_config'],'effective_aux_weight':0.,'CNN':'TinyCNN91905,80epochs,batch32,D4train,AdamW.002cosine.0001,wd.01,clip5,last5weightsaverage,identityinference','short_B':'hidden32,dropout.5,12epochs,effectivebatch16,microbatch2,AdamW.0003cosine.00001,wd.1,clip1,seed20270215+offset','offsets':'root0,middle100(j+1),deep100(j+1)+100(ell+1)','top_KRR':'9grid (.25,1,4)x(.01,.1,1),GroupKFold3seed20261005 onfixeddevelopment400;refit500','nested_KRR':'same9grid onallcurrentnodeU,seed20261005+offset','nested_split':'GroupKFold3seed20270110+offset','raw_weights':[0.,.3,.3,.2,.2],'base':'.9affine(raw)+.1shortB','residual':'585features,3honestblocks,none/ridge1/ridge01/rbf1/rbf01,inner3seed20271013+j,512groupbootstrapseed20271204+j,mean3','tolerances':old['tolerances'],'selection_ids_source':'fixed originaldevelopment400 IDs only; no oldlabels/predictions/models reused','no_output_clipping':True};(D/'config/recipe.json').write_text(json.dumps(recipe,indent=2))
for file in ['base_train.py','train.py']:compile((D/file).read_text(),file,'exec')
print('FULL500_TRAINER_READY')
