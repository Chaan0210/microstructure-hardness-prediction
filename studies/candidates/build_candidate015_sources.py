from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[1]
old=(_repo_path(R, 'candidate014_pipeline.py')).read_text()
header=old[:old.index('def inputs():')].replace('import component_single_ablation_122 as study','import candidate014_pipeline as previous\nimport cnn_seed_mean_133 as x\nimport component_single_ablation_122 as study').replace('final-model-123','final-model-134').replace('submission-candidate-014','submission-candidate-015')
custom='''
CNN_ROOT=ROOT/'artifacts/final-model-024'
def cnnpath(seed):return CNN_ROOT/f'cnn-seed{seed}.pt'
def prepare():
 previous.check();x.check();assert not CSV.exists();OUT.mkdir(exist_ok=False)
 dev=e.read(x.OUT/'confirmation-decision.json');assert dev['all_independent_validation_passed'] and dev['lower_pooled_MSE']
 names=['candidate015_pipeline.py','validate_candidate015.py','candidate014_pipeline.py','validate_candidate014.py','cnn_seed_mean_133.py','validate_cnn_seed_mean_133.py','candidate004_pipeline.py','frozen_validation.py','reports/cnn-seed-mean-133-summary.json','reports/cnn-seed-mean-133-preservation.json','artifacts/cnn-seed-mean-133/confirmation-decision.json','artifacts/cnn-seed-mean-133/pooled-decision-policy.json','artifacts/final-model-024/report.json','artifacts/submission-candidate-014.csv','artifacts/submission-candidate-014.manifest.json']
 names += [str(cnnpath(seed).relative_to(ROOT)) for seed in x.SEEDS]
 p={'candidate':'015','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Parent explicitly authorized fixed133 recipe full500 fitting and test1000 inference, no upload.','fixed_seeds':list(x.SEEDS),'formula':'raw=.3compact26KRR+.3mean3identityCNN+.2Smean3+.2Bmean3; base=.9*newaffine(raw)+.1shortB; prediction=base+mean3newhonestphysics57residuals','weights_changed':False,'seed_selection':False,'inference_augmentation':'identity only; historical024 D4 inference is not reused','development':dev,'pooled_RMSE_gain':0.0355203045085446,'conditional95_gain_CI':[-0.02680458603553579,0.09978930199665262],'limits':['Same400 adaptively reused; overlapping partitions are not independent.','Fixed-fit CI excludes model refitting and research-selection uncertainty; includes zero.','Final repartition worsened.','Pooled policy requested after first resplit result; no actual early stop or candidate change.'],'training':'Reuse exact full500 CNN seed checkpoints024; train missing24 nestedCNNs via133 node; refit4affines and48 residual models including none. Existing all500 independent KRR/S/B/shortB retained.','test_scope':'Infer only after training validation; no test fit/selection/calibration/clipping.','prior_inspection':'During cache inspection historical004 report displayed prior test aggregates. They did not affect the already-fixed recipe.','holdout_scope':'All500 now authorized for training; no formerholdout subgroup scoring.','hashes':{f:e.sha(ROOT/f) for f in names},'protected':previous.check()['protected'],'cost':0,'uploaded':False};e.write(OUT/'decision.json',p);e.emit({'candidate015_locked':True,'new_nested_CNN_fits':24,'reused_full500_CNNs':3})
def check():
 previous.check();x.check();p=e.read(OUT/'decision.json')
 for name,h in {**p['hashes'],**p['protected']}.items():assert e.sha(ROOT/name)==h,name
 return p
def inputs(train=False):
 d,P,cells=previous.inputs();P=P.copy();new=[];root=OUT/'cnn-nested'
 if train:root.mkdir(exist_ok=False)
 for j,(A,B,M,D,short) in enumerate(cells):
  M=M.copy();D=D.copy();M[:,2]=x.node(d,A,B,100*(j+1),M[:,2],root/f'middle-{j}',train);P[B,2]=M[:,2];lookup={int(i):k for k,i in enumerate(A)}
  for ell in range(3):
   C,E,DP=a.bundle(c.OUT/f'block-{j}/deep-{ell}');assert set(C)|set(E)==set(A) and not(set(C)|set(E))&set(B)
   D[[lookup[int(i)] for i in E],2]=x.node(d,C,E,100*(j+1)+100*(ell+1),DP[:,2],root/f'deep-{j}-{ell}',train)
  new.append((A,B,M,D,short));e.emit({'full500_nested_block':j,'training':train})
 return d,P,new
'''
fit=old[old.index('def fit():'):old.index("if __name__=='__main__':")]
fit=fit.replace('def fit():\n check();d,P,cells=inputs();start=time.perf_counter();','def fit():\n check();start=time.perf_counter();d,P,cells=inputs(train=True);')
fit=fit.replace("'new_neural_fits':0","'new_neural_fits':24,'reused_full500_CNNs':3")
fit=fit.replace("cnn=c.cp.predict_cnn(z['cnn_images'],ROOT/'artifacts/final-model-017/cnn.pt',device)","cnn=np.mean([c.cp.predict_cnn(z['cnn_images'],cnnpath(seed),device) for seed in x.SEEDS],axis=0)")
fit=fit.replace('candidate014','candidate015').replace("'candidate':'014'","'candidate':'015'").replace("'013_preserved':True","'014_preserved':True")
fit=fit.replace('Freshreloadremainingfull500KRR,CNN,S/Bheads,shortB,andnewcalibrator/residuals;no795baselineused.','Fresh full500 KRR, three identity CNN seeds equally averaged, S/B heads, shortB, and new calibrator/residuals; no795baseline.')
main=old[old.index("if __name__=='__main__':"):]
(_repo_path(R, 'candidate015_pipeline.py')).write_text(header+custom+'\n'+fit+main)
v=(_repo_path(R, 'validate_candidate014.py')).read_text().replace('import candidate014_pipeline as p','import candidate015_pipeline as p\nimport validate_cnn_seed_mean_133 as v133')
v=v.replace("p.check();d,P,cells=p.inputs();fit=", "p.check();d,P,cells=p.inputs();cnnchecks=cnn_audit(d);nestedchecks=[v133.nodecheck(d,path) for path in sorted((p.OUT/'cnn-nested').iterdir())];fit=")
v=v.replace("'passed':True,'rows':500", "'passed':True,'full500_CNN_checks':cnnchecks,'nested_CNN_checks':nestedchecks,'new_nested_CNN_models_checked':24,'rows':500")
v=v.replace("for j,name in enumerate(['kernel','cnn','S_mean3','B_mean3'],1):", "for j,name in [(1,'kernel'),(3,'S_mean3'),(4,'B_mean3')]:")
v=v.replace('candidate014','candidate015').replace("'candidate':'014'","'candidate':'015'")
v=v.replace("'013_preserved':True","'014_preserved':True")
v=v.replace("artifacts/submission-candidate-013.csv","artifacts/submission-candidate-014.csv").replace('difference_from013_RMS','difference_from014_RMS')
audit='''
def cnn_audit(d):
 report=e.read(p.CNN_ROOT/'report.json');assert report['n_train']==500 and report['training_id_order']==d['ids'];assert report['train_csv_sha256']==e.sha(ROOT/'data/train.csv');out=[]
 for ident,h in report['training_image_sha256'].items():assert e.sha(ROOT/'data/train'/f'{ident}.png')==h
 for seed in p.x.SEEDS:
  file=p.cnnpath(seed);assert e.sha(file)==report['model_sha256'][file.name];m=torch.load(file,map_location='cpu',weights_only=True);assert m['train_ids']==d['ids'] and m['seed']==seed and m['epochs']==80 and m['average_last_epochs']==5
  norm={'x_mean':float(d['images'].mean()),'x_std':float(d['images'].std()),'y_mean':float(d['y'].mean()),'y_std':float(d['y'].std())};assert m['normalization']==norm
  V=np.arange(500);pm=p.x.predict(m,d,V,'mps');pc=p.x.predict(m,d,V,'cpu');err=float(abs(pm-pc).max());assert err<1e-3
  out.append({'seed':seed,'checkpoint_sha256':e.sha(file),'train_rows':500,'identity_CPU_MPS_max_error':err,'train_stats_and_image_hashes_verified':True})
 assert e.sha(p.cnnpath(p.x.SEEDS[0]))==e.sha(ROOT/'artifacts/final-model-017/cnn.pt')
 return out
'''
v=v.replace('def independent_ols(',audit+'\ndef independent_ols(')
(_repo_path(R, 'validate_candidate015.py')).write_text(v)
print('created candidate015_pipeline.py and validate_candidate015.py')
