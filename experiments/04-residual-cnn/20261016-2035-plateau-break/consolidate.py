"""Combine unchanged H1/H3 with the independently preserved H2 tie repair."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import evaluate as e
import numpy as np,json,csv,sys

def main():
    p=dict(np.load(_repo_io(s.HERE/'all-predictions.npz')));new=dict(np.load(_repo_io(s.HERE/'h2-tie-repair/evaluated-predictions.npz')))
    assert np.array_equal(p['ids'],new['ids']) and np.array_equal(p['y'],new['y']) and np.array_equal(p['fold'],new['fold'])
    for k in ['linear','marginal','joint']:p['H2_'+k]=new[k]
    results=json.loads((s.HERE/'results.json').read_text());rep=json.loads((s.HERE/'h2-tie-repair/results.json').read_text())
    for k in ['linear','marginal','joint']:results['H2_'+k]=rep[k]
    bands={'clean':p['noise']<9.5,'middle':(p['noise']>=9.5)&(p['noise']<12),'high':p['noise']>=12}
    for k in ['linear','marginal','joint']:
        name='H2_'+k;results[name]['bands']={name_:dict(n=int(m.sum()),baseline_rmse=e.rmse(p['y'][m],p['baseline'][m]),candidate_rmse=e.rmse(p['y'][m],p[name][m]),gain=e.rmse(p['y'][m],p['baseline'][m])-e.rmse(p['y'][m],p[name][m])) for name_,m in bands.items()}
    decision=json.loads((s.HERE/'decision.json').read_text());decision['decisions']['H2_joint']=json.loads((s.HERE/'h2-tie-repair/decision.json').read_text());decision['advance']=[k for k,v in decision['decisions'].items() if v['advance']];assert not decision['advance']
    decision['authoritative_H2']='h2-tie-repair';decision['original_H2_retained']=True
    s.save(_repo_io('final-results.json'),results);s.save(_repo_io('final-decision.json'),decision);s.arrays('final-predictions.npz',**p)
    with (s.HERE/'final-comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','rmse','gain','conditional_ci95_low','conditional_ci95_high','improved_folds','top10_removed_gain','matched_control_gain'])
        for n,r in results.items():w.writerow([n,r['rmse'],r['gain'],*r['conditional_ci95'],r['improved_folds'],r['removed_top_gain']['10']['gain'],r.get('matched_control',{}).get('gain')])
    names=list(results)
    with (s.HERE/'final-per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','noise','target','baseline',*names,*[n+'_SSE_gain' for n in names]])
        for i in range(400):w.writerow([p['ids'][i],p['fold'][i],p['noise'][i],p['y'][i],p['baseline'][i],*[p[n][i] for n in names],*[(p['baseline'][i]-p['y'][i])**2-(p[n][i]-p['y'][i])**2 for n in names]])
    print((s.HERE/'final-comparison.csv').read_text(),flush=True)

if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=1):main()
