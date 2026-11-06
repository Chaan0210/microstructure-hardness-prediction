"""Descriptive mechanisms and failure cases; never selects a new candidate."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import numpy as np,json,sys,csv
def main():
    result=json.loads((s.HERE/'results.json').read_text());o=np.load(_repo_io(s.HERE/'oof.npz'));d=dict(np.load(_repo_io(s.OLD/'inputs.npz')));graphs=json.loads((s.OLD/'graphs.json').read_text());noise_out={a:np.zeros(400) for a in ['ssl','random']};rows=[];position={int(j):i for i,j in enumerate(d['dev'])}
    for k,g in enumerate(graphs):
        for stage in ['selector','refit']:
            rr={a:json.loads((s.HERE/f'models/fold{k}-{stage}-{a}/training.json').read_text()) for a in s.ARMS};r=rr['ssl'];rows.append(dict(fold=k,stage=stage,SSL_first_loss=r['pretraining']['curve'][0]['loss'],SSL_last_loss=r['pretraining']['curve'][-1]['loss'],SSL_last_terms=r['pretraining']['curve'][-1]['ssl_terms'],frozen_SSL_rank=r['frozen_representation'],frozen_random_rank=rr['none']['frozen_representation'],finetuned_ranks={a:v['representation'] for a,v in rr.items()},training_RMSE={a:v['train_rmse'] for a,v in rr.items()}))
        for name,arm in [('ssl','ssl'),('random','none')]:
            z=np.load(_repo_io(s.HERE/f'models/fold{k}-refit-{arm}/frozen-probe.npz'));assert np.array_equal(z['train'],g['T']) and np.array_equal(z['query'],g['V']);pred,_=s.ridge_predict(z['train_features'],d['noise'][z['train']],z['query_features']);ix=np.array([position[int(j)] for j in z['query']]);noise_out[name][ix]=pred
    noise=d['noise'][d['dev']];noise_probes={a:dict(RMSE=s.rmse(p,noise),correlation=float(np.corrcoef(p,noise)[0,1]),scope='Descriptive fixed-alpha100 training-only Ridge for image-derived noise, not hardness; does not select any model') for a,p in noise_out.items()}
    gain=(o['baseline082']-o['y'])**2-(o['ssl']-o['y'])**2;hi=o['noise']>=12;fail=[]
    for i in np.argsort(gain)[:10]:
        fail.append(dict(ID=str(o['ids'][i]),fold=int(o['fold'][i]),noise=float(o['noise'][i]),hardness=float(o['y'][i]),baseline082=float(o['baseline082'][i]),candidate=float(o['ssl'][i]),image_prediction=float(o['ssl_image'][i]),SSE_gain=float(gain[i])))
    size={a:[r['arms'][a] for r in result['selection']] for a in s.ARMS}
    s.save(_repo_io('diagnostics.json'),dict(representations=rows,noise_probes=noise_probes,worst_SSL_changes=fail,new_branch_size_transfer=size,
        training_size_caution='A213 and T320 compare the SAME V80 for the new branch; fixed-A ensemble still uses the refitted082 baseline, so this does not isolate a full-pipeline training-size effect.',
        irreducibility_caution='Shared residuals and poor learned representations do not establish irreducible ambiguity. These results constrain this objective, architecture, strength and budget only.'))
    s.arrays('noise-diagnostic.npz',ids=o['ids'],noise=noise,**noise_out)
    print('NOISE_PROBES',noise_probes);print('SSL_FROZEN_RANK_MEAN',np.mean([r['frozen_SSL_rank']['effective_rank'] for r in rows]));print('SSL_WORST',fail[:3])
if __name__=='__main__':
    sys.addaudithook(s.guard)
    with s.threadpool_limits(limits=2):main()
