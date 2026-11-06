"""Matched-size and influential-source descriptions for the frozen support policy."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import stats as s
import numpy as np,json,csv,sys
ARMS=['topology','local','hardness','random','raw']
def main():
    p=c.HERE/'quality-policy';o=np.load(_repo_io(p/'oof.npz'));r=json.loads((p/'results.json').read_text());d=c.inputs(labels=True);base=o['baseline082'];y=o['y'];fixed={a:base.copy() for a in ARMS};source_pos={int(v):j for j,v in enumerate(d['dev'])};rows=[]
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));a=np.load(_repo_io(c.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(c.HERE/f'readouts/fold{k}-refit/predictions.npz'));ix=an['ix'];support=o['support'][ix]
        for arm in ARMS:
            rec=r['selection'][k]['arms'][arm];route=rec['route'];entry=dict(fold=k,arm=arm,route=route,beta=rec['beta'])
            if route:
                da=(a[arm]-a['base']).mean(0)[len(an['B']):];dt=(v[arm]-v['base']).mean(0);mask=an['highV']&(support if route=='supported' else True);fixed[arm][ix]=base[ix]+mask*rec['beta']*da;entry.update(routed_V_sources=int(mask.sum()),same_V_A_to_T_direction_shift_RMS=float(np.sqrt(np.mean((dt[mask]-da[mask])**2))) if mask.any() else 0.)
            rows.append(entry)
    size={a:s.metrics(y,base,v,o['fold'],seed=2026101797) for a,v in fixed.items()};gain=(base-y)**2-(o['topology']-y)**2;worst=np.argsort(gain)[:10];best=np.argsort(-gain)[:10];targets={row['ID']:row for row in json.loads((c.HERE/'target-audit.json').read_text())['sources']}
    def cases(indices):return [dict(ID=str(o['ids'][j]),fold=int(o['fold'][j]),support=bool(o['support'][j]),source_evidence_AUC=targets[str(o['ids'][j])]['boundary_evidence_AUC'],y=float(y[j]),baseline082=float(base[j]),candidate=float(o['topology'][j]),SSE_gain=float(gain[j])) for j in indices]
    with (p/'per-image.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','fold','high','support','y','082',*ARMS,'topology_SSE_gain'])
        for j in range(len(y)):w.writerow([o['ids'][j],o['fold'][j],o['high'][j],o['support'][j],y[j],base[j],*[o[a][j] for a in ARMS],gain[j]])
    c.save(_repo_io('quality-policy/diagnostics.json'),dict(matched_A_branch=size,selected_direction_size_checks=rows,worst=cases(worst),best=cases(best),no_parameter_or_protocol_changes=True,scope='The new branch is held atA213 for both B calibration and V prediction; frozen082 stays its usual source-nested baseline. Post-score descriptions are not alternative winners.'));print('QUALITY DIAGNOSTICS SAVED',flush=True)
if __name__=='__main__':
    sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
