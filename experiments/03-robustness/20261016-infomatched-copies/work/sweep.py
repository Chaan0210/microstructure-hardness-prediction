"""Run calib.evaluate for a list of (name, params) given as a JSON file; append one line per setting to work/calib/<LOG> (default sweep.log) and
save each setting's pixel statistics to work/calib/<name>.npz. Label-free. usage: sweep.py LIST.json [LOG] [PROCS]"""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib
if __name__ == '__main__':
    L = json.load(open(_repo_io(sys.argv[1]))); C = _repo_path(Path(__file__).resolve().parent, 'calib'); log = open(_repo_io(C / (sys.argv[2] if len(sys.argv) > 2 else 'sweep.log')), 'a'); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    for name, prm in L:
        res, (ids, X) = calib.evaluate(prm, procs=procs); np.savez(_repo_io(C / f'{name}.npz'), ids=ids, X=X, keys=np.array(calib.PKEYS)); s = f"{name:22s} logit {res['logit']:.3f} gbm {res['gbm']:.3f} | " + ' '.join(f'{k}:{v:+.2f}' for k, v in list(res['smd'].items())[:7])
        print(s, flush=True); log.write(s + '  ' + json.dumps(prm) + '\n'); log.flush()
