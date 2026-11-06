"""Reproduce the fixed076+CNN CSV using Python's standard library only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=_RepoPath(__file__).resolve().parents[3]


def main(output):
    protected=json.loads((_repo_path(HERE, 'protected.json')).read_text())
    data={};source_hashes={};order=None
    for tag in ('076','078','080'):
        name=f'artifacts/submission-candidate-{tag}.csv';path=_repo_path(ROOT, name)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==protected[name],f'Input changed: {name}'
        source_hashes[name]=sha
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        assert list(rows[0])==['ID','hardness']
        ids=[r['ID'] for r in rows]
        if order is None:order=ids
        assert ids==order and len(ids)==len(set(ids))==1000
        assert all(i.startswith('TEST_') for i in ids)
        data[tag]=[float(r['hardness']) for r in rows]
    with (_repo_path(ROOT, 'data/sample_submission.csv')).open(newline='') as f:
        assert [r['ID'] for r in csv.DictReader(f)]==order
    text=io.StringIO(newline='');writer=csv.writer(text);writer.writerow(['ID','hardness'])
    n_changed=0
    for i,a,b,c in zip(order,data['076'],data['078'],data['080']):
        assert all(math.isfinite(v) for v in (a,b,c))
        assert (b-a)*(c-b)==0.,f'Changes no longer have disjoint support: {i}'
        pred=a+(c-b);assert math.isfinite(pred)
        writer.writerow([i,repr(pred)]);n_changed+=int(pred!=a)
    encoded=text.getvalue().encode('utf-8');digest=hashlib.sha256(encoded).hexdigest()
    expected=json.loads((_repo_path(HERE, 'attribution.json')).read_text())['CSV_sha256']
    assert digest==expected
    output=Path(output)
    if output.exists():assert output.read_bytes()==encoded,'Will not overwrite a different file'
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as f:f.write(encoded)
    print(json.dumps(dict(file=str(output),sha256=digest,rows=1000,changed_rows=n_changed,
                          formula='076+(080-078)',source_sha256=source_hashes,uploaded=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=_repo_path(HERE, 'candidate-076-plus-frozen-cnn.csv'))
    main(p.parse_args().output)
