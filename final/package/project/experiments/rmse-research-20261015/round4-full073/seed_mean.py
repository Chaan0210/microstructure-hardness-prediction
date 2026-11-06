"""Fixed equal-seed predictions, followed by fresh internal-only calibration."""
import json
from pathlib import Path
import numpy as np
import cnn_experiment as e
HERE=e.HERE;r=e.r


def main():
    d=dict(np.load(HERE.parent/'inputs.npz'));labels=d['y'];reports={}
    for family in ('cnn','aux','stats','context'):
        sources=[HERE/f'{family}-seed{s}' for s in range(2)]
        if not all((p/'verification.json').exists() for p in sources):continue
        root=HERE/f'{family}-mean2';root.mkdir(exist_ok=True)
        lock={str(p.relative_to(r.ROOT)):r.digest(p) for p in [Path(__file__),HERE/'AMENDMENT-04.md',HERE/'AMENDMENT-06.md']}
        records=[]
        for k in range(5):
            inner=[np.load(p/f'fold{k}-selection.npz') for p in sources]
            outer=[np.load(p/f'fold{k}-outer.npz') for p in sources]
            for key in ('B','base073','non021','eligible'):assert np.array_equal(inner[0][key],inner[1][key])
            for key in ('V','dev_index','ids','base','non021'):assert np.array_equal(outer[0][key],outer[1][key])
            B=inner[0]['B'];V=outer[0]['V'];assert set(B).isdisjoint(V)
            rb=np.mean([z['residual'] for z in inner],axis=0)
            rv=np.mean([z['residual'] for z in outer],axis=0)
            a,sa=e.coefficient(inner[0]['base073'],rb,labels[B],inner[0]['eligible'])
            b,sb=e.coefficient(inner[0]['non021'],rb,labels[B],inner[0]['eligible'])
            changed=labels.copy();changed[V]+=10000
            aa,_=e.coefficient(inner[0]['base073'],rb,changed[B],inner[0]['eligible'])
            bb,_=e.coefficient(inner[0]['non021'],rb,changed[B],inner[0]['eligible']);assert (aa,bb)==(a,b)
            z=outer[0]
            np.savez_compressed(root/f'fold{k}-outer.npz',V=V,dev_index=z['dev_index'],ids=z['ids'],base=z['base'],
                                non021=z['non021'],residual=rv,addition=z['base']+a*rv,
                                replacement=z['non021']+b*rv,lambda_add=a,lambda_replace=b)
            np.savez_compressed(root/f'fold{k}-selection.npz',B=B,base073=inner[0]['base073'],
                                non021=inner[0]['non021'],residual=rb,eligible=inner[0]['eligible'])
            r.save_json(root/f'fold{k}-selection.json',dict(add=sa,replace=sb,seed_weights=[.5,.5],
                       weight_validation_ids=d['ids'][B].tolist(),outer_validation_ids=d['ids'][V].tolist()))
            records.append(dict(fold=k,seed_weights=[.5,.5],outer_label_poison_exact=True,
                                residual_prediction_correlation=float(np.corrcoef(outer[0]['residual'],outer[1]['residual'])[0,1])))
            for p in sources:
                for filename in (f'fold{k}-selection.npz',f'fold{k}-outer.npz','verification.json'):
                    lock[str((p/filename).relative_to(r.ROOT))]=r.digest(p/filename)
        r.save_json(root/'provenance.json',lock)
        r.save_json(root/'verification.json',dict(all_passed=True,kind='deterministic ensemble of verified CNN fits',folds=records,
                                                fixed_seed_mean=True,weights_fit_on_B_only=True))
        e.summarize(root,d,labels)
        reports[family]=records
    assert reports,'No verified two-seed result available'
    r.save_json(HERE/'seed-mean-verification.json',reports)


if __name__=='__main__':main()
