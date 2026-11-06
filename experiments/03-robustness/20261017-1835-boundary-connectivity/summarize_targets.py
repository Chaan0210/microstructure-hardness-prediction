"""Reproduce source-level weak-target diagnostics without hardness labels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import numpy as np,json,sys
def main():
    r=json.loads((c.HERE/'target-audit.json').read_text());z=np.load(_repo_io(c.HERE/'targets.npz'));d=c.inputs();assert len(z['dev'])==len(np.unique(z['dev']))==400 and np.array_equal(z['dev'],d['dev']);assert all(str(x).startswith('TRAIN_') for x in z['ids']);results={}
    for name,sel in [('all',np.ones(400,bool)),('high_eligible',(d['noise'][d['dev']]>=12)&~d['failed']),('lower',d['noise'][d['dev']]<12)]:
        rows=[x for x,b in zip(r['sources'],sel) if b];rec={}
        for field in ['node_support','boundary_positive_fraction','edge_known_fraction','pair_positive_count','pair_negative_count','boundary_evidence_AUC']:
            v=np.array([x[field] for x in rows]);rec[field]=dict(median=float(np.median(v)),q10=float(np.quantile(v,.1)),q90=float(np.quantile(v,.9)))
        agree=[x['withheld_view_balanced_agreement'] for x in rows if x.get('withheld_view_balanced_agreement') is not None];rec.update(n=len(rows),four_view_sources=sum(x['views']==4 for x in rows),withheld_view_sources=len(agree),withheld_view_median=float(np.median(agree)) if agree else None,no_pair_positive_or_negative=sum(min(x['pair_positive_count'],x['pair_negative_count'])==0 for x in rows));results[name]=rec
    result=dict(source_groups=results,ids_unique_and_train_only=True,all_development_sources_exact=True,hardness_used=False,interpretation='Withheld-view agreement is conditional on the confident common interiors and a correlated auxiliary teacher. It is not segmentation accuracy. Physical-evidenceAUC compares pseudo-boundary versus pseudo-interior raw ridge/gradient strength, not true boundaries. Positive support weights vary within each image; per-class normalization and equal image averaging do not remove all-image teacher bias.')
    path=c.HERE/'target-summary.json'
    if path.exists():assert json.loads(path.read_text())==result;print('Existing target summary exactly verified')
    else:c.save(_repo_io('target-summary.json'),result);print('Target summary written')
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
