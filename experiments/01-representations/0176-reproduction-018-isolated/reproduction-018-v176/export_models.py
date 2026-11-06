"""Export only the freshly trained176 production models after fulltraining audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse,json,shutil
from pathlib import Path
import numpy as np
import train as t

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,default=_repo_path(t.ROOT, 'models'));args=ap.parse_args();t.check();v=t.read(t.OUT/'training-validation.json');assert v['passed'];root=t.OUT/'full';assert v['training_report_sha256']==t.sha(root/'report.json');out=args.models.resolve();out.mkdir(exist_ok=False)
 for file in (root/'outer').iterdir():
  if file.suffix in ['.pt','.joblib']:shutil.copyfile(_repo_io(file),_repo_io(out/file.name))
 r=t.read(root/'report.json');t.write(out/'calibrator.json',r['calibrator'])
 for j in range(3):
  path=root/f'residual-{j}'
  with np.load(_repo_io(path/'predictions.npz')) as z:w=z['weights'].tolist()
  t.write(out/f'weights-{j}.json',{'kinds':t.KINDS,'weights':w})
  for kind in t.KINDS:shutil.copyfile(_repo_io(path/f'full-{kind}.joblib'),_repo_io(out/f'residual-{j}-{kind}.joblib'))
 shutil.copyfile(_repo_io(t.OUT/'training-validation.json'),_repo_io(out/'training-validation.json'));manifest={'candidate':'018','study':176,'training_validated':True,'all_fitted_models_fresh176':True,'source_training_plan_sha256':t.sha(t.OUT/'plan.json'),'source_fit_report_sha256':t.sha(root/'report.json'),'files':{p.name:t.sha(p) for p in out.iterdir() if p.is_file()}};t.write(out/'manifest.json',manifest);t.emit({'exported_new_models':str(out),'manifest_sha256':t.sha(out/'manifest.json')})
if __name__=='__main__':main()
