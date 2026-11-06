"""Close the negative experiment only after its promised audits pass."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,re,datetime
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    get=lambda n:json.loads((_repo_path(HERE, n)).read_text())
    decision=get('decision-0.json');assert not decision['advance']
    assert get('training-complete-seed0.json')['complete'] and get('verification-0.json')['all_passed']
    rep=get('reproduction/082-inference-verification.json');assert rep['CSV_byte_identical'] and rep['incumbent082_max_difference']==0.
    assert get('reproduction/cache-replay.json')['byte_identical'];pr=get('preservation.json');assert pr['all_unchanged']
    for filename in ['analysis-code-lock.json','mechanism-code-lock.json']:
        assert all(sha(_repo_path(HERE, p))==h for p,h in get(filename)['sha256'].items())
    assert sha(_repo_path(HERE, 'experiment.py'))==get('code-lock.json')['source_sha256']
    report=(_repo_path(HERE, 'REPORT.md')).read_text();assert 'PRESERVATION_STATUS' in report
    report=report.replace('PRESERVATION_STATUS',f"**all {pr['files']:,} protected files are unchanged**; Git HEAD, tracked diff and staged diff also remain unchanged")
    report=report.replace('|REL addition|','|REL ensemble|').replace('-076','- 076')
    for a,b in [('forMSE','for MSE'),('forREL','for REL'),('forSHUFFLED','for SHUFFLED'),('versusSHUFFLED','versus SHUFFLED'),('versusMSE','versus MSE'),('andMSE','and MSE')]:report=report.replace(a,b)
    lines=[];fenced=False
    for line in report.splitlines():
        if line.startswith('```'):fenced=not fenced;lines.append(line);continue
        if not fenced:
            parts=line.split('`')
            for j in range(0,len(parts),2):
                parts[j]=re.sub(r'(?<!float)(?<!z)(?<=[a-z])(?=\d)',' ',parts[j])
                parts[j]=re.sub(r'(?<=\d)(?=[A-Za-z])(?![eE][+-]?\d)',' ',parts[j])
                parts[j]=re.sub(r'(?<=%)(?=[A-Za-z])',' ',parts[j])
            line='`'.join(parts)
        lines.append(line)
    (_repo_path(HERE, 'REPORT.md')).write_text('\n'.join(lines)+'\n')
    record=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),incumbent='082',rollback='076',main_image_fits=30,resource_probe_fits=3,complete_training_replays=1,
        nested_original_five_folds_complete=True,advance=False,new_official_best=False,new_test_candidate=False,public_submissions=0,installations=0,paid_compute=False,
        protected_files=pr['files'],incumbent082_and076_CSV_byte_identical=True,full_historical_training_reproduced=False,
        source_sha256={n:sha(_repo_path(HERE, n)) for n in ['experiment.py','evaluate.py','verify.py','mechanisms.py','reproduce.py','PLAN.md']})
    with (_repo_path(HERE, 'completion.json')).open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
