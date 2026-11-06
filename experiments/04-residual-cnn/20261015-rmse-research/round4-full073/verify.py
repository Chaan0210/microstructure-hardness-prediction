"""Independent membership, formula, checkpoint and label-poison verification."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import cnn_experiment as e
HERE=e.HERE;r=e.r


def check_sources():
    old=json.loads((HERE.parent/'provenance.json').read_text())
    protected={k:v for k,v in old.items() if '/submission-candidate-' in k}
    for name,digest in protected.items():assert r.digest(_repo_path(r.ROOT, name))==digest,name
    teacher=json.loads((HERE/'teacher-provenance.json').read_text())
    for name,digest in teacher.items():assert r.digest(_repo_path(r.ROOT, name))==digest,name
    deriv=json.loads((HERE/'fitter-derivation.json').read_text())
    assert r.digest(deriv['source'])==deriv['source_sha256']
    assert r.digest(deriv['derived'])==deriv['derived_sha256']
    environment=json.loads((HERE/'environment.json').read_text())
    for name,digest in environment['files'].items():assert r.digest(_repo_path(r.ROOT, name))==digest,name
    return dict(protected_original_submissions=protected,teacher_source_files=len(teacher),
                original_cnn_source_sha256=deriv['source_sha256'])


def check_baseline():
    d=dict(np.load(_repo_io(HERE.parent/'inputs.npz')))
    ridge=np.load(_repo_io(HERE.parent/'consistency-strict.npz'))['production']
    gate=d['noise'][d['dev']]>=12
    # scope.strict holds the high-noise DTA fit on every query, not the gated partner.
    log=np.where(gate,np.load(_repo_io(HERE.parent/'scope.npz'))['strict'],d['C73_logr'])
    pred=.16*d['C73_cnn']+.384*d['C73_tab']+.128*ridge+.128*log+.05*d['C73_set']+.15*d['C73_gmm']
    pred=np.where(d['fail'],d['C73_cnn'],pred)
    old=np.load(_repo_io(HERE.parent/'scope.npz'))['strict_blend']
    error=float(np.max(np.abs(pred-old)));assert error<1e-10
    return dict(reconstruction_max_error=error,original_rmse=r.rmse(old[0],d['y'][d['dev']]),
                all_original_components_present=True,noise_gates_preserved=True)


def check_teachers(complete_only=False):
    d=dict(np.load(_repo_io(HERE.parent/'inputs.npz')));gg=json.loads((HERE/'label-graph.json').read_text())
    reports=[]
    for g in gg:
        for node in g['nodes']:
            path=HERE/f"teachers/fold{g['fold']}-{node['name']}"
            if complete_only and not (path/'predictions.npz').exists():continue
            z=np.load(_repo_io(path/'predictions.npz'));tr=z['train'];query=z['query']
            assert np.array_equal(tr,node['train']) and np.array_equal(query,node['query'])
            assert set(tr).isdisjoint(query) and not (set(tr)|set(query))&set(g['V'])
            info=json.loads((path/'contract.json').read_text())
            assert info['train_ids']==d['ids'][tr].tolist() and info['query_ids']==d['ids'][query].tolist()
            import hashlib
            assert info['target_sha256']==hashlib.sha256(d['y'][tr].tobytes()).hexdigest()
            tabs=list(path.glob('tab-*.npz'));assert len(tabs)==72
            for noisy in (False,True):
                qi=query[(d['noise'][query]>=12)==noisy]
                pp=[]
                for slot in ('038','039','044'):
                    for variant in ('u1dm1szH2Z','u1dm2szH2Z','u1dmavgH2Z','u1dmavg4H2Z'):
                        for seed in range(3):
                            file=path/f'tab-{int(noisy)}-{slot}-{variant}-seed{seed}.npz'
                            tab=np.load(_repo_io(file));assert np.array_equal(tab['train'],tr) and np.array_equal(tab['query'],qi)
                            pp.append(tab['prediction'])
                assert len(pp)==36
                assert np.array_equal(z['tab'][(d['noise'][query]>=12)==noisy],np.mean(pp,axis=0)) or np.allclose(z['tab'][(d['noise'][query]>=12)==noisy],np.mean(pp,axis=0),atol=1e-12,rtol=0)
            setfiles=list(path.glob('grain-set-*.npz'));assert len(setfiles)==3
            pp=[]
            for file in sorted(setfiles):
                s=np.load(_repo_io(file));assert np.array_equal(s['train'],tr) and np.array_equal(s['query'],query);pp.append(s['prediction'])
                history=json.loads(file.with_suffix('.json').read_text())['stopping']
                ia,ib=map(set,(history['inner_train_ids'],history['inner_validation_ids']))
                assert not ia&ib and ia|ib==set(d['ids'][tr])
            assert np.allclose(np.mean(pp,axis=0),z['set'],rtol=0,atol=1e-12)
            detail=json.loads((path/'details.json').read_text())
            for part in detail['linear']:assert set(part['augmentation_source_ids'])<=set(d['ids'][tr])
            mixture=(.384*z['tab']+.128*z['ridge']+.128*z['logridge']+.05*z['set']+.15*z['gmm'])/.84
            bad=np.nan_to_num(d['X'][query,list(d['keys']).index('n_obj')],nan=0)<20
            mixture[bad]=d['y'][tr].mean();assert np.array_equal(mixture,z['base'])
            reports.append(dict(fold=g['fold'],node=node['name'],n_train=len(tr),n_query=len(query),
                                tab_fits=len(tabs),set_fits=len(setfiles),all_checks_passed=True))
    return reports


def check_cnn(folder,poison=False):
    d=e.c.load_data();labels=d.pop('y');gg=json.loads((HERE/'label-graph.json').read_text());reports=[]
    for name,digest in json.loads((folder/'provenance.json').read_text()).items():assert r.digest(_repo_path(r.ROOT, name))==digest
    for g in gg:
        k=g['fold'];A,B,T,V=[np.array(g[key]) for key in ('A','B','T','V')]
        if folder.name.startswith('stats'):e.c.ComplementCNN=e.StatsOnly
        sel=np.load(_repo_io(folder/f'fold{k}-selection.npz'));sj=json.loads((folder/f'fold{k}-selection.json').read_text())
        outer=np.load(_repo_io(folder/f'fold{k}-outer.npz'))
        for key,base in (('add',sel['base073']),('replace',sel['non021'])):
            w,_=e.coefficient(base,sel['residual'],labels[B],sel['eligible'])
            assert w==sj[key]['selected']
            assert set(sj['weight_validation_ids']).isdisjoint(d['ids'][V])
            poisoned_outer=labels.copy();poisoned_outer[V]+=10000
            poisoned_weight,_=e.coefficient(base,sel['residual'],poisoned_outer[B],sel['eligible'])
            assert poisoned_weight==w
        assert np.array_equal(outer['addition'],outer['base']+sj['add']['selected']*outer['residual'])
        assert np.array_equal(outer['replacement'],outer['non021']+sj['replace']['selected']*outer['residual'])
        for stage,kind,tr,va in [('selector','deep',A,B),('final','outer',T,V)]:
            teacher,sources=e.teacher_oof(d,g,kind,tr)
            raw=None
            if folder.name.startswith(('aux','context')):
                if stage=='selector':
                    z=np.load(_repo_io(HERE/f'teachers/fold{k}-outer0/predictions.npz'));qb=z['base'];qg=z['tab']-z['ridge']
                else:
                    ix=outer['dev_index'];baseline=np.load(_repo_io(HERE.parent/'scope.npz'))['strict_blend'][0]
                    qb=(baseline[ix]-.16*d['C73_cnn'][0,ix])/.84
                    qg=d['C73_tab'][0,ix]-np.load(_repo_io(HERE.parent/'consistency-strict.npz'))['production'][0,ix]
                raw=e.raw_context(d,g,kind,teacher,va,qb,qg)
            data=e.cnn_data(d,tr,va,labels[tr],teacher,sources,k,'weight_selector' if stage=='selector' else 'outer_refit',raw)
            path=folder/f'fold{k}-{stage}';checkpoint=torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True)
            model=e.c.ComplementCNN();model.load_state_dict(checkpoint['state_dict'])
            expected=np.load(_repo_io(path/'predictions.npz'))['validation_prediction']
            pred=e.c.predict(model,data['images'],data['context'],va,checkpoint['arm'])*data['meta']['target_std']
            assert np.array_equal(pred,expected)
            changed=labels.copy();changed[va]+=10000
            poisoned=e.cnn_data(d,tr,va,changed[tr],teacher,sources,k,data['meta']['stage'],raw)
            assert poisoned['meta']==data['meta']
            assert np.array_equal(poisoned['targets']['residual'],data['targets']['residual'])
            if poison and k==0 and stage=='selector':
                new=HERE/f'poison-{folder.name}-selector0'
                refit=e.fitted(poisoned,checkpoint['seed'],new,checkpoint['arm'])
                assert np.array_equal(refit,expected)
                a=torch.load(_repo_io(new/'model.pt'),weights_only=True)['state_dict']
                assert all(torch.equal(a[key],v) for key,v in checkpoint['state_dict'].items())
            reports.append(dict(fold=k,stage=stage,reload_exact=True,query_poison_targets_unchanged=True,
                                full_poison_refit=bool(poison and k==0 and stage=='selector')))
    return reports


def check_set_checkpoints():
    import copy
    import full_branch as b
    d,labels=b.load();sm=d['set_module'];bank=dict(d['object_bank'])
    # Even training labels are absent during independent checkpoint inference.
    bank['y']=np.full(len(bank['ids']),np.nan);reports=[]
    for path in sorted((HERE/'teachers').glob('*/predictions.npz')):
        z=np.load(_repo_io(path));query=d['bank_index'][z['query']]
        for seed in b.SET_SEEDS:
            sm.CONFIG={**copy.deepcopy(d['set_config']),'seed':seed}
            pred,checkpoint,model,data=sm.load_checkpoint_predict(path.parent/f'grain-set-{seed}.pt',bank,query,'cpu')
            expected=np.load(_repo_io(path.parent/f'grain-set-{seed}.npz'))['prediction']
            difference=float(np.max(np.abs(pred-expected)));assert difference==0.
            assert set(checkpoint['context']['training_ids'])==set(z['train_ids'])
            assert checkpoint['context']['y_mean']==float(labels[z['train']].mean())
            assert checkpoint['context']['y_scale']==float(max(labels[z['train']].std(),1e-8))
            reports.append(dict(teacher=path.parent.name,seed=seed,reload_max_error=difference,labels_absent=True))
            del model,data
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path);p.add_argument('--poison',action='store_true')
    p.add_argument('--complete-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        report=dict(sources=check_sources(),baseline=check_baseline(),teachers=check_teachers(a.complete_only),partial=a.complete_only)
        if not a.complete_only:
            assert len(report['teachers'])==30
            # Run this once for the teacher-only audit; later CNN audits reuse it.
            if a.folder is None:report['grain_set_checkpoint_reload']=check_set_checkpoints()
        if a.folder:report['cnn']=check_cnn(a.folder,a.poison)
        destination=(a.folder or HERE)/('verification-partial.json' if a.complete_only else 'verification.json');r.save_json(destination,report)
        print('VERIFIED',len(report['teachers']),'teachers',len(report.get('cnn',[])),'CNN fits',flush=True)
