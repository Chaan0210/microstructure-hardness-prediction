"""Complete only predeclared, unstarted arms after the original timing limit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import ast,json,sys
def main():
    lock=json.loads((s.HERE/'code-lock.json').read_text());assert s.sha(s.HERE/'study.py')==lock['source_sha256'];assert not (s.HERE/'training-complete.json').exists()
    for path in (s.HERE/'models').iterdir():
        assert all((path/n).exists() for n in ['training.json','predictions.npz','model.pt']),f'Partial output is not resumed: {path}'
    s.CAP=55*60
    tree=ast.parse((s.HERE/'study.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    # The immutable code-lock write is omitted; all training code is unchanged.
    node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='save' and n.value.args and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='code-lock.json')]
    skip=ast.parse("""
prior_folder=HERE/f'models/fold{k}-{stage}-{arm}'
if (prior_folder/'training.json').exists():
    previous=json.loads((prior_folder/'training.json').read_text())
    assert previous['normalization']==meta and previous['seed']==seed and previous['arm']==arm
    elapsed+=sum(previous[n]['seconds'] for n in ['pretraining','first_supervised','supervised'] if n in previous)
    outputs.append(str(prior_folder.relative_to(HERE)))
    print('REUSED_COMPLETE',k,stage,arm,'accounted_neural_seconds',round(elapsed,1),flush=True)
    continue
""").body
    loops=[n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='arm'];assert len(loops)==1;loops[0].body=skip+loops[0].body
    mod=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));namespace=dict(vars(s));namespace['CAP']=s.CAP
    s.save(_repo_io('resumption-record.json'),dict(training_source_sha256=s.sha(s.HERE/'study.py'),runner_sha256=s.sha(__file__),budget_amendment_sha256=s.sha(s.HERE/'BUDGET-AMENDMENT.md'),cap_seconds=s.CAP,changes='Skip completed arms and existing code-lock write; identical numerical training function and fixed matrix.'))
    exec(compile(mod,str(s.HERE/'study.py'),'exec'),namespace);namespace['train']()
if __name__=='__main__':
    sys.addaudithook(s.guard);s.torch.set_num_threads(2);s.torch.set_num_interop_threads(1);s.torch.use_deterministic_algorithms(True)
    with s.threadpool_limits(limits=2):main()
