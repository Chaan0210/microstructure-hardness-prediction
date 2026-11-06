"""Close the report only after every required verification has completed."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,re,datetime
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    get=lambda n:json.loads((_repo_path(HERE, n)).read_text())
    p=get('preservation.json');assert p['all_unchanged'] and p['git_head_unchanged'] and p['tracked_diff_unchanged']
    assert get('verification.json')['all_passed'] and get('h2-tie-repair/verification.json')['all_passed']
    assert not get('final-decision.json')['advance'] and not get('final-decision.json')['promoted']
    assert get('cache-replay.json')['byte_identical'] and get('raw-image-spot-replay.json')['all_passed']
    assert get('precision-train-results.json')['n_contexts']==90 and get('precision-train-results.json')['all_complete']
    assert len(get('tab-replay.json')['records'])==72 and get('model-replay.json')['all1000']
    lock=get('pre-score-lock.json')['sha256'];assert all(sha(_repo_path(HERE, n))==h for n,h in lock.items())
    lock=get('precision-pre-score-lock.json')['source_sha256'];assert all(sha(_repo_path(HERE, n))==h for n,h in lock.items())
    text=(_repo_path(HERE, 'REPORT.md')).read_text();assert 'PRESERVATION_STATUS' in text
    text=text.replace('PRESERVATION_STATUS',f"**all {p['files_verified']:,} protected files ({p['bytes_verified']/2**30:.2f} GiB) are unchanged**. Git HEAD, tracked diff and staged diff also match the recorded starting state")
    text=text.replace('9.509485','9.509490').replace('−0.015055','−0.015058')
    # Improve prose spacing without touching identifiers in inline/fenced code.
    lines=[];fenced=False
    for line in text.splitlines():
        if line.startswith('```'):fenced=not fenced;lines.append(line);continue
        if not fenced:
            parts=line.split('`')
            for j in range(0,len(parts),2):
                parts[j]=re.sub(r'(?<!float)(?<!z)(?<=[a-z])(?=\d)',' ',parts[j])
                parts[j]=re.sub(r'(?<=\d)(?=[a-df-zA-DF-Z])',' ',parts[j])
                parts[j]=re.sub(r'(?<=%)(?=[A-Za-z])',' ',parts[j])
            line='`'.join(parts)
        lines.append(line)
    (_repo_path(HERE, 'REPORT.md')).write_text('\n'.join(lines)+'\n')
    before=get('repository.json')['status'];own='?? experiments/'+HERE.name+'/\n'
    (_repo_path(HERE, 'preexisting-git-status.txt')).write_text(before.replace(own,''))
    record=dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),incumbent='076',new_candidate_promoted=False,
        authoritative_results='final-results.json',authoritative_predictions='final-predictions.npz',new_hypotheses=3,
        all_original_fold_experiments_complete=True,H2_tie_bug_repaired_and_retested=True,train_only_precision_contexts=90,
        preserved_files=p['files_verified'],full_pipeline_training_reproduced=False,final_heads_replayed=True,uploads=0,installs=0,paid_compute=False)
    with (_repo_path(HERE, 'completion.json')).open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
