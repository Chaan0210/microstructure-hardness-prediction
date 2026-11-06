from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
R=_RepoPath(__file__).resolve().parents[1]
old=(_repo_path(R, 'candidate015_pipeline.py')).read_text();body=old[old.index('def fit():'):old.index("if __name__=='__main__':")]
body=body.replace('d,P,cells=inputs(train=True)','d,P,cells=inputs()').replace("_,cal=a.affine(ip,d['y'],ip)","cal=q.fit(ip,d['y'])").replace("pred,coef=a.affine(a.mixture(D,ARM),d['y'][A],a.mixture(M,ARM))","coef=q.fit(a.mixture(D,ARM),d['y'][A]);pred=q.predict(coef,a.mixture(M,ARM))")
body=body.replace("calpred=cal['target_center']+cal['slope']*(mixture-cal['prediction_center'])","calpred=q.predict(cal,mixture)")
body=body.replace("'new_affine_fits':4","'new_quadratic_calibrators':4").replace("'new_neural_fits':24","'new_neural_fits':0").replace('candidate015','candidate016').replace("'candidate':'015'","'candidate':'016'").replace("'014_preserved':True","'015_preserved':True")
head='''"""Frozen137 quadratic extension of015; full500 downstream-only refit."""
import sys,csv,time,datetime,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import candidate015_pipeline as previous
import quadratic_calibration_137 as q
import numpy as np,torch,joblib
x=previous.x;a=previous.a;e=previous.e;n=previous.n;r=previous.r;c=previous.c;p8=previous.p8;vn=previous.vn;ARM=previous.ARM
OUT=ROOT/'artifacts/final-model-138';CSV=ROOT/'artifacts/submission-candidate-016.csv'
cnnpath=previous.cnnpath
inputs=previous.inputs

def prepare():
 previous.check();q.check();assert not CSV.exists();OUT.mkdir(exist_ok=False);decision=e.read(q.OUT/'decision.json');assert decision['eligible'] and decision['independent_validation_passed'];names=['candidate016_pipeline.py','validate_candidate016.py','quadratic_calibration_137.py','validate_quadratic_calibration_137.py','reports/quadratic-calibration-137-summary.json','artifacts/quadratic-calibration-137/decision.json','reports/candidate015-public-submission.json','artifacts/submission-candidate-015.csv','artifacts/submission-candidate-015.manifest.json','artifacts/final-model-134/fit.json','artifacts/final-model-134/training-validation.json']
 for node in (previous.OUT/'cnn-nested').rglob('*'):
  if node.is_file():names.append(str(node.relative_to(ROOT)))
 p={'candidate':'016','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formula':'same015 fixedraw mixture; .9*train-only quadratic137 calibration(raw)+.1shortB+mean3newhonestphysics57residuals','lambda':1.,'extra_curvature_df_per_calibrator':.5,'development_decision':decision,'training':'Exactly137:4 quadratic calibrators and48 residual models includingnone. Reuse verifiedfull500 independentbaselearners andall133/134 3seednestedCNNmeans. No newneuralfits.','holdout':'Included onlyinlockedfinal500fit; no subgroup score orselection.','test':'Onlyafterfit andindependenttrainingvalidation; no fit/tuning/clipping.','hashes':{f:e.sha(ROOT/f) for f in names},'protected':q.check()['protected'],'best015_retained':True,'cost':0,'uploaded':False};e.write(OUT/'decision.json',p);e.emit({'candidate016_locked':True,'new_neural_fits':0,'new_quadratic_calibrators':4})
def check():
 previous.check();q.check();p=e.read(OUT/'decision.json')
 for f,h in {**p['hashes'],**p['protected']}.items():assert e.sha(ROOT/f)==h,f
 return p

'''
main="""if __name__=='__main__':
 torch.set_num_threads(2);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','predict']);args=parser.parse_args();globals()[args.mode]()
"""
(_repo_path(R, 'candidate016_pipeline.py')).write_text(head+body+main)
v=(_repo_path(R, 'validate_candidate014.py')).read_text().replace('import candidate014_pipeline as p','import candidate016_pipeline as p\nimport validate_quadratic_calibration_137 as qv')
start=v.index('def independent_ols(');end=v.index('\ndef training():',start)
v=v[:start]+'''def independent_ols(P,y,Q):
 return qv.independent(P,y,Q),None
'''+v[end:]
v=v.replace("_,coef=independent_ols(ip,d['y'],ip);assert abs(coef[0]-cal['target_center'])<1e-10 and abs(coef[1]-cal['slope'])<1e-10","manual,_=independent_ols(ip,d['y'],ip);np.testing.assert_allclose(manual,p.q.predict(cal,ip),atol=1e-9,rtol=0)")
v=v.replace("for j,name in enumerate(['kernel','cnn','S_mean3','B_mean3'],1):","for j,name in [(1,'kernel'),(3,'S_mean3'),(4,'B_mean3')]:")
v=v.replace('candidate014','candidate016').replace("'candidate':'014'","'candidate':'016'").replace("'013_preserved':True","'015_preserved':True").replace('artifacts/submission-candidate-013.csv','artifacts/submission-candidate-015.csv').replace('difference_from013_RMS','difference_from015_RMS').replace('affine_fits_independently_verified','quadratic_fits_independently_verified').replace('independent_OLS_kernel_final_max_error','independent_quadratic_kernel_final_max_error')
v=v.replace("'cached_nested_base_training_certificate':e.sha(p.c.OUT/'training-validation.json')","'cached_nested_base_training_certificate':e.sha(p.c.OUT/'training-validation.json'),'fixed3seed_full500_and_nested_CNN_certificate':e.sha(p.previous.OUT/'training-validation.json')")
(_repo_path(R, 'validate_candidate016.py')).write_text(v)
print('created candidate016 code and independent validator')
