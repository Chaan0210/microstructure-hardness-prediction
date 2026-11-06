"""Read-only incumbent inventory. Never run baseline builders in place."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

HERE=Path(__file__).resolve().parent
ROOT=_RepoPath(__file__).resolve().parents[3]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def write(name,x):
    with (_repo_path(HERE, name)).open('x') as f: json.dump(x,f,indent=2)

def main():
    start=time.monotonic()
    # Conservatively protect entire ancestors, including unused siblings, rather
    # than attempting to infer every dynamically imported historical dependency.
    roots=['data','src','vendor','artifacts','reports','requirements.in','requirements.lock',
           'requirements-cnn.in','requirements-cnn.lock','.venv',
           'deliverables/reproduction-018-v176','experiments/candidate021-178',
           'experiments/reproduction-018-176','experiments/dinol-177',
           'experiments/grain-cellpose-220','experiments/probabilistic-grains-900',
           'experiments/tabpfn-v2','experiments/tabpfn-v3','experiments/rmse-research-20261015']
    roots += [f'experiments/push-{n}' for n in range(300,1400,100)]
    paths=set(ROOT.glob('*.py'))|set(ROOT.glob('*.md'))
    for name in roots:
        p=_repo_path(ROOT, name)
        if p.is_file(): paths.add(p)
        elif p.is_dir(): paths.update(q for q in p.rglob('*') if q.is_file())
    # Active 076-teacher work is not an incumbent ancestor and is not frozen.
    # Its already complete, read-only baseline is an explicit evaluation input.
    for name in ['baseline076.npz','baseline076.json','probe.py']:
        paths.add(_repo_path(ROOT, 'experiments/076-complement-20261016', name))
    paths={p for p in paths if '__pycache__' not in p.parts and not p.name.endswith('.pyc') and p.name!='.DS_Store'}
    records={}
    for j,p in enumerate(sorted(paths)):
        stat=p.stat(); digest=sha(p)
        assert p.stat().st_mtime_ns==stat.st_mtime_ns, f'Input changing: {p}'
        records[str(p.relative_to(ROOT))]=dict(sha256=digest,size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,symlink=str(p.readlink()) if p.is_symlink() else None)
        if j%3000==0: print('protected',j,'elapsed',round(time.monotonic()-start),flush=True)
    write('protected.json',records)
    git=lambda *a:subprocess.check_output(['git',*a],cwd=_repo_io(ROOT),text=True)
    write('repository.json',dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        status=git('status','--porcelain=v1'),diff=git('diff','--binary'),staged_diff=git('diff','--cached','--binary')))
    envs={}
    code='import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version,executable=sys.executable,packages={d.metadata["Name"]:d.version for d in m.distributions()})))'
    for name in ['.venv','experiments/tabpfn-v2/.venv','experiments/grain-cellpose-220/.venv']:
        envs[name]=json.loads(subprocess.check_output([str(_repo_path(ROOT, name, 'bin/python')),'-B','-c',code],text=True))
    write('environments.json',envs)
    scores=[]
    for p in sorted((_repo_path(ROOT, 'reports')).rglob('*public*submission*.json')):
        try:
            z=json.loads(p.read_text())
            if isinstance(z.get('public_rmse'),(float,int)): scores.append(dict(file=str(p.relative_to(ROOT)),score=z['public_rmse']))
        except (ValueError,TypeError): pass
    scores.sort(key=lambda d:d['score'])
    write('incumbent.json',dict(candidate='076',recorded_public_rmse=8.9646534604,
        csv='artifacts/submission-candidate-076.csv',csv_sha256=sha(_repo_path(ROOT, 'artifacts/submission-candidate-076.csv')),
        official_promotion_authorized=False,score_records=scores,
        formula='076 = 074 + .384*(H9_Tab - prior_Tab); 074 = 073 + high_noise*.128*((B_Ridge-DTA_Ridge)+(B_logRidge-DTA_logRidge))',
        component_weights=dict(G021=.16,Tab=.384,Ridge=.128,logRidge=.128,grain_set=.05,GMM=.15),
        noise_threshold_gray=12,failed_fallback='G021 for n_obj<20',
        tab_seeds=[0,100,200],tab_variants=['u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z'],
        original_training_inference_commands=['See frozen experiments/push-300/REPRO-030.md',
            'See frozen experiments/push-400/REPRO-035.md','See frozen experiments/push-500/REPRO-038.md',
            'See frozen experiments/push-800/REPRO-045.md',
            'push-1300/final_nb_tab.py; final_nb_partners.py; final_srcwB.py; final_h9_tab.py (run from historical cwd; NOT executed in place)',
            'push-1300/make_073.py; make_074.py; make_076.py (original assembly; NOT executed in place)'],
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_files=len(records),total_bytes=sum(r['size'] for r in records.values()),seconds=time.monotonic()-start,
        active_other_job='PID 76231 retarget_teachers.py --folds 2 3 4 observed read-only; not modified'))
    print('FROZEN',len(records),round(time.monotonic()-start,1),flush=True)

if __name__=='__main__': main()
