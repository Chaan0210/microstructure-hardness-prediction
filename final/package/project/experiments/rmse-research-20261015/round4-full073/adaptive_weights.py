"""Two-parameter convex stacking, trained only on the internal B predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import cnn_experiment as e
HERE=e.HERE;r=e.r


def simplex_weight(design,target):
    def objective(w):return float(np.mean((design@w-target)**2))
    def gradient(w):return 2*design.T@(design@w-target)/len(target)
    result=minimize(objective,np.zeros(2),jac=gradient,method='SLSQP',bounds=[(0.,1.),(0.,1.)],
                    constraints=[{'type':'ineq','fun':lambda w:1.-w.sum(),'jac':lambda w:-np.ones(2)}],
                    options={'ftol':1e-11,'maxiter':500})
    assert result.success,result.message
    weights=np.maximum(result.x,0.);weights/=max(1.,weights.sum())
    # Independent boundary/interior check for this convex quadratic.
    candidates=[np.zeros(2)]
    for j in range(2):
        p=design[:,j];w=np.zeros(2);w[j]=np.clip(p@target/(p@p),0,1) if p@p>1e-12 else 0
        candidates.append(w)
    p=design[:,0]-design[:,1];t=target-design[:,1]
    w=np.clip(p@t/(p@p),0,1) if p@p>1e-12 else 0;candidates.append(np.array([w,1-w]))
    ols=np.linalg.lstsq(design,target,rcond=None)[0]
    if np.min(ols)>=0 and ols.sum()<=1:candidates.append(ols)
    assert objective(weights)<=min(map(objective,candidates))+1e-7
    return weights


def run():
    assert not (HERE/'cnn-seed0').exists() or (HERE/'cnn-seed0/summary.json').exists(),'Wait for complete CNN results'
    d=dict(np.load(HERE.parent/'inputs.npz'));y=d['y'][d['dev']];fold=d['fold'][:1]
    root=HERE/'adaptive-weights';root.mkdir(exist_ok=True)
    lock={str(p.relative_to(r.ROOT)):r.digest(p) for p in
          (Path(__file__),HERE/'AMENDMENT-02.md',HERE/'label-graph.json',HERE.parent/'inputs.npz')}
    r.save_json(root/'provenance.json',lock)
    reports={};common=None
    for folder in sorted([*HERE.glob('*-seed[01]'),*HERE.glob('*-mean2')]):
        if not (folder/'summary.json').exists():continue
        oof=np.load(folder/'oof.npz');joint=np.zeros(400);adaptive=np.zeros(400);records=[]
        for k in range(5):
            inner=np.load(folder/f'fold{k}-selection.npz');outer=np.load(folder/f'fold{k}-outer.npz')
            anchor=np.load(HERE/f'anchors/fold{k}.npz');B=inner['B'];good=inner['eligible'];ix=outer['dev_index']
            assert np.array_equal(B,anchor['B']) and set(B).isdisjoint(outer['V'])
            t=inner['non021'][good];g=anchor['prediction_B'][good];rr=inner['residual'][good]
            target=d['y'][B][good]-t
            a,_=e.coefficient(t,g-t,d['y'][B][good],np.ones(len(t),bool))
            w=simplex_weight(np.c_[g-t,rr],target)
            tg=outer['non021'];gg=d['C73_cnn'][0,ix];rv=outer['residual'];bad=d['fail'][0,ix]
            adaptive[ix]=tg+a*(gg-tg);joint[ix]=tg+w[0]*(gg-tg)+w[1]*rv
            adaptive[ix[bad]]=oof['base'][ix[bad]];joint[ix[bad]]=oof['base'][ix[bad]]
            # Poison outer labels; the weight fit receives an unchanged B target.
            altered=d['y'].copy();altered[outer['V']]+=10000
            ww=simplex_weight(np.c_[g-t,rr],altered[B][good]-t);assert np.array_equal(ww,w)
            records.append(dict(fold=k,control_021_weight=a,joint_021_weight=float(w[0]),
                                joint_cnn_weight=float(w[1]),joint_T_weight=float(1-w.sum()),
                                weight_validation_ids=d['ids'][B].tolist(),outer_label_poison_exact=True))
        if common is None:common=adaptive.copy()
        else:assert np.array_equal(common,adaptive),'Weight-only control must be independent of CNN arm'
        report=dict(vs073=r.compare(oof['base'][None],joint[None],y,fold),
                    vs_weight_only=r.compare(adaptive[None],joint[None],y,fold),
                    weight_only_vs073=r.compare(oof['base'][None],adaptive[None],y,fold),weights=records)
        reports[folder.name]=report
        np.savez_compressed(root/f'{folder.name}.npz',ids=oof['ids'],fold=fold[0],baseline=oof['base'],
                            weight_only=adaptive,joint=joint)
        r.save_json(root/f'{folder.name}.json',report)
    assert reports,'No completed CNN output'
    r.save_json(root/'summary.json',reports)
    for name,report in reports.items():
        print(name,'joint',report['vs073']['candidate']['pooled'],
              'CNN incremental gain',report['vs_weight_only']['gain'],flush=True)


if __name__=='__main__':run()
