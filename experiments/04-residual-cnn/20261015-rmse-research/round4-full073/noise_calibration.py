"""Partially pooled noise-regime coefficients; tuning and fitting only on B."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import json
from pathlib import Path
import numpy as np
import cnn_experiment as e
HERE=e.HERE;r=e.r
STRENGTHS=('global',100,30,10,0)


def fit_weights(direction,target,band,strength):
    den=float(direction@direction)
    global_weight=float(np.clip(direction@target/den,0,1)) if den>1e-12 else 0.
    if strength=='global':return np.full(2,global_weight)
    penalty=float(strength)*float(np.mean(direction**2));weights=[]
    for group in (False,True):
        mask=band==group;p=direction[mask];t=target[mask];dd=float(p@p)+penalty
        weights.append(float(np.clip((p@t+penalty*global_weight)/dd,0,1)) if dd>1e-12 else global_weight)
    return np.array(weights)


def select(direction,target,band):
    errors=[]
    for strength in STRENGTHS:
        pred=np.zeros(len(target))
        for i in range(len(target)):
            train=np.arange(len(target))!=i
            w=fit_weights(direction[train],target[train],band[train],strength)
            pred[i]=w[int(band[i])]*direction[i]
        errors.append(float(np.mean((pred-target)**2)))
    chosen=STRENGTHS[int(np.argmin(errors))]
    return fit_weights(direction,target,band,chosen),dict(strength=chosen,loo_mse=errors,grid=list(STRENGTHS))


def main():
    d=dict(np.load(_repo_io(HERE.parent/'inputs.npz')));y=d['y'];dev=d['dev'];root=HERE/'noise-calibration';root.mkdir(exist_ok=True)
    all_reports={}
    for family in ('cnn','aux','stats','context'):
        folder=HERE/f'{family}-mean2';reference=np.load(_repo_io(folder/'oof.npz'))
        out={key:np.full(400,np.nan) for key in ('addition','replacement')};records=[]
        for k in range(5):
            inner=np.load(_repo_io(folder/f'fold{k}-selection.npz'));outer=np.load(_repo_io(folder/f'fold{k}-outer.npz'))
            B=inner['B'];V=outer['V'];good=inner['eligible'];assert set(B).isdisjoint(V)
            band=d['noise'][B][good]>=12;vband=(d['noise'][V]>=12).astype(int)
            record=dict(fold=k,n_by_regime=[int((band==i).sum()) for i in (0,1)],weight_validation_ids=d['ids'][B].tolist())
            for arm,key in (('addition','base073'),('replacement','non021')):
                target=(y[B]-inner[key])[good];direction=inner['residual'][good]
                weights,selection=select(direction,target,band)
                altered=y.copy();altered[V]+=10000
                check,check_meta=select(direction,(altered[B]-inner[key])[good],band)
                assert np.array_equal(check,weights) and check_meta==selection
                base=outer['base'] if arm=='addition' else outer['non021']
                out[arm][outer['dev_index']]=base+weights[vband]*outer['residual']
                record[arm]=dict(weights=weights.tolist(),**selection,outer_label_poison_exact=True)
            records.append(record)
        report={arm:r.compare(reference['base'][None],pred[None],y[dev],d['fold'][:1]) for arm,pred in out.items()}
        for arm in out:report[arm+'_vs_global_weight']=r.compare(reference[arm][None],out[arm][None],y[dev],d['fold'][:1])
        report['replacement_vs_deletion']=r.compare(reference['non021'][None],out['replacement'][None],y[dev],d['fold'][:1])
        report['selection']=records;all_reports[family]=report
        np.savez_compressed(_repo_io(root/f'{family}-oof.npz'),ids=reference['ids'],fold=reference['fold'],baseline=reference['base'],non021=reference['non021'],**out)
        r.save_json(root/f'{family}-verification.json',dict(all_passed=True,threshold=12,calibration_selection_inside_B=True,records=records))
        print(family,{a:dict(rmse=report[a]['candidate']['pooled'],gain_vs_global=report[a+'_vs_global_weight']['gain']) for a in out},flush=True)
    r.save_json(root/'summary.json',all_reports)
    paths=[Path(__file__),HERE/'AMENDMENT-05.md',HERE.parent/'inputs.npz']
    paths+=list((HERE/'cnn-mean2').glob('fold*-selection.npz'))+list((HERE/'aux-mean2').glob('fold*-selection.npz'))
    r.save_json(root/'provenance.json',{str(p.relative_to(r.ROOT)):r.digest(p) for p in paths})


if __name__=='__main__':main()
