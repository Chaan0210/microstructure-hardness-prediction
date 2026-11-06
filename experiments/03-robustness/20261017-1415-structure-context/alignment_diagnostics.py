"""Descriptive source-level and matched-training-size diagnostics; no selection."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import encoder as e
import evaluate as ev
import numpy as np,json,csv,sys
OUT=e.HERE/'alignment-followup'
def main():
    z=np.load(_repo_io(OUT/'oof.npz'));d=np.load(_repo_io(e.OLD/'inputs.npz'));res=json.loads((OUT/'results.json').read_text());graphs=json.loads((e.OLD/'graphs.json').read_text());base=z['baseline082'];y=z['y'];p=base.copy();rows=[]
    for k,g in enumerate(graphs):
        an=np.load(_repo_io(e.OLD/f'anchors/fold{k}.npz'));rr=res['selection'][k]['arms']['spatial'];tag=rr['choice'];record=dict(fold=k,choice=tag,beta=rr['beta'])
        if tag:
            root=e.HERE if tag=='PCA' else OUT;s=np.load(_repo_io(root/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(root/f'readouts/fold{k}-refit/predictions.npz'));ds=(s['spatial']-s['base']).mean(0);dv=(v['spatial']-v['base']).mean(0);ab=ds[:len(an['B'])];av=ds[len(an['B']):];p[an['ix']]=base[an['ix']]+an['highV']*rr['beta']*av
            record.update(B_direction_rms=float(np.sqrt(np.mean(ab[an['highB']]**2))),V_A_direction_rms=float(np.sqrt(np.mean(av[an['highV']]**2))),V_T_direction_rms=float(np.sqrt(np.mean(dv[an['highV']]**2))),V_A_to_T_shift_rms=float(np.sqrt(np.mean((dv[an['highV']]-av[an['highV']])**2))))
        rows.append(record)
    worst={}
    for a in ['spatial','permuted','random']:
        diff=(base-y)**2-(z[a]-y)**2;order=np.argsort(diff)[:10]
        worst[a]=[dict(ID=str(z['ids'][j]),fold=int(z['fold'][j]),high_noise=bool(z['high'][j]),y=float(y[j]),baseline082=float(base[j]),candidate=float(z[a][j]),SSE_gain=float(diff[j])) for j in order]
    with (OUT/'per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','high_noise','y','082','spatial','permuted','random','spatial_SSE_gain'])
        for j in range(len(y)):w.writerow([z['ids'][j],z['fold'][j],z['high'][j],y[j],base[j],z['spatial'][j],z['permuted'][j],z['random'][j],(base[j]-y[j])**2-(z['spatial'][j]-y[j])**2])
    result=dict(worst_changes=worst,selected_direction_size_diagnostics=rows,fixed_A_branch_diagnostic=ev.metrics(y,base,p,z['fold']),limits='All post-score descriptions. No coefficient, routing, feature or candidate selection changes. Fixed-A holds the new branch at213 training sources; baseline082 remains the frozen source-nested reference.')
    e.save(_repo_io('alignment-followup/diagnostics.json'),result);print('Alignment descriptive diagnostics saved')
if __name__=='__main__':
    sys.addaudithook(e.guard)
    with e.threadpool_limits(limits=2):main()
