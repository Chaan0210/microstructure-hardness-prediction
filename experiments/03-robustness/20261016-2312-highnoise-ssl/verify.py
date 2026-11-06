"""Inference, source isolation, matched controls, and selection arithmetic audit."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import study as s
import evaluate as ev
import numpy as np,json,math,sys,gc,os
def main():
    result=json.loads((s.HERE/'results.json').read_text());out=np.load(_repo_io(s.HERE/'oof.npz'));d=s.load_pixels();records=[];comparisons=[]
    lock=json.loads((s.HERE/'analysis-code-lock.json').read_text());assert all(s.sha(s.HERE/k)==v for k,v in lock.items())
    for k,g in enumerate(d['graphs']):
        for stage in ['selector','refit']:
            tr=np.array(g['A' if stage=='selector' else 'T']);q=np.r_[g['B'],g['V']] if stage=='selector' else np.array(g['V']);meta_ref=dict(image_mean=float(d['pixels'][tr].mean(dtype=np.float64)/255),image_std=float(d['pixels'][tr].std(dtype=np.float64)/255),target_mean=float(d['y'][tr].mean()),target_std=float(d['y'][tr].std()),train_ids=d['ids'][tr].tolist());rr={}
            for arm in s.ARMS:
                path=s.HERE/f'models/fold{k}-{stage}-{arm}';ck=s.torch.load(_repo_io(path/'model.pt'),map_location='cpu',weights_only=True);m=s.Model();m.load_state_dict(ck['state_dict']);assert ck['normalization']==meta_ref
                p,z=s.predict(m,d['pixels'][q],ck['normalization']);old=np.load(_repo_io(path/'predictions.npz'));assert np.array_equal(p,old['prediction']) and np.array_equal(z,old['embedding']);assert set(tr).isdisjoint(q)
                rec=json.loads((path/'training.json').read_text());assert rec['training_pixels_sha256']==s.hashlib.sha256(d['pixels'][tr].tobytes()).hexdigest();assert rec['training_labels_sha256']==s.hashlib.sha256(d['y'][tr].tobytes()).hexdigest();rr[arm]=rec
                if arm in ['ssl','none']:
                    if arm=='ssl':
                        frozen=s.torch.load(_repo_io(path/'ssl.pt'),map_location='cpu',weights_only=True);assert set(frozen['normalization'])=={'image_mean','image_std','train_ids'};m.load_state_dict(frozen['state_dict']);assert frozen['record']['labels_used'] is False
                    else:s.torch.manual_seed(s.SEED+100*k);m=s.Model()
                    _,t=s.predict(m,d['pixels'][tr],meta_ref);_,v=s.predict(m,d['pixels'][q],meta_ref);pr=np.load(_repo_io(path/'frozen-probe.npz'));assert np.array_equal(t,pr['train_features']) and np.array_equal(v,pr['query_features']);p2,head=s.ridge_predict(t,d['y'][tr],v);assert np.array_equal(p2,pr['prediction']);assert head==json.loads((path/'frozen-head.json').read_text())
                records.append(dict(fold=k,stage=stage,arm=arm,reload_prediction_max_difference=0.,normalization_training_only=True,source_ids_disjoint=True));del m;gc.collect()
            assert len({v['initial_sha256'] for v in rr.values()})==1
            for key in ['order_sha256','updates','source_visits','view_visits']:
                assert len({v['supervised'][key] for v in rr.values()})==1,key
            assert len({rr[a]['supervised']['views_sha256'] for a in ['augment','ssl','long']})==1
            for key in ['order_sha256','views_sha256','updates','source_visits','view_visits']:
                assert rr['ssl']['pretraining'][key]==rr['long']['first_supervised'][key]
            comparisons.append(dict(fold=k,stage=stage,initialization_exact=True,fine_tune_orders_and_augmentation_exact=True,SSL_and_long_first_phase_views_and_exposures_exact=True))
    arithmetic={}
    for arm in ev.ARMS:
        score=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(out[arm],out['y']))/len(out['y']));assert abs(score-result['results'][arm]['candidate_rmse'])<1e-12;arithmetic[arm]=score
    selected=[]
    for k,g in enumerate(d['graphs']):
        z=np.load(_repo_io(s.OLD/f'anchors/fold{k}.npz'));B=z['B'];V=z['V'];ix=z['ix'];high=z['highB']
        for arm in ev.ARMS:
            pp=ev.predictions(k,'selector',arm)['prediction'][:len(B)];dd=pp-z['p82B'];nn=math.fsum(float(dd[i])*float(d['y'][B[i]]-z['p82B'][i]) for i in np.flatnonzero(high));den=math.fsum(float(dd[i])**2 for i in np.flatnonzero(high));w=max(0.,min(.25,nn/max(den,1e-12)));saved=result['selection'][k]['arms'][arm]['weight'];assert abs(w-saved)<1e-14
            poison=d['y'].copy();poison[V]+=10000;assert ev.weight(pp,z['p82B'],poison[B],high)==saved
            vp=ev.predictions(k,'refit',arm)['prediction'];assembled=z['p82V']+z['highV']*saved*(vp-z['p82V']);assert np.array_equal(assembled,out[arm][ix]);selected.append(dict(fold=k,arm=arm,weight=saved,outer_label_poison_exact=True))
    s.save(_repo_io('verification.json'),dict(all_passed=True,models=records,matched_controls=comparisons,independent_RMSE=arithmetic,blend_checks=selected,code_locks_match=True,limits='This verifies the new pipeline conditional on frozen historical082 representations; it is not a fresh evaluation of the entire historical selection process.'))
    print('VERIFIED',len(records),'models;',len(selected),'nested blend checks',flush=True)
def replay():
    """Full same-seed SSL+supervised replay, outside labels poisoned and reads blocked."""
    assert (s.HERE/'BUDGET-AMENDMENT.md').exists();s.CAP=55*60
    completed=json.loads((s.HERE/'training-complete.json').read_text());d=s.load_pixels();g=d['graphs'][0];tr=np.array(g['A']);q=np.r_[g['B'],g['V']];original=s.HERE/'models/fold0-selector-ssl';r=json.loads((original/'training.json').read_text());needed=r['pretraining']['seconds']+r['supervised']['seconds']
    if completed['neural_seconds']+needed>s.CAP:
        s.save(_repo_io('training-replay.json'),dict(executed=False,reason='Amended55-minute neural budget has insufficient reserve for another complete fit',required_estimate_seconds=needed,already_used_seconds=completed['neural_seconds']));return
    changed=d['y'].copy();outside=np.setdiff1d(np.arange(500),tr);changed[outside]+=10000
    px=d['pixels'][tr].copy();labels=changed[tr].copy();meta=r['normalization'];reg=s.regions(px);s.torch.manual_seed(s.SEED);m=s.Model();folder=s.HERE/'training-replay';folder.mkdir(exist_ok=False);scope={'fit':True}
    def block_reads(event,args):
        if scope['fit'] and event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=s.Path(os.fsdecode(args[0]));mode,flags=args[1:];writing=(mode is not None and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if not writing and p.suffix.lower() in ['.npz','.npy','.png','.jpg','.csv','.pt','.joblib']:raise RuntimeError('Data/model reads prohibited during isolated fit')
    sys.addaudithook(block_reads)
    a=s.phase(m,px,reg,{k:meta[k] for k in ['image_mean','image_std','train_ids']},s.SEED+10000,s.EPOCHS,r['strength']);scope['fit']=False;old=s.torch.load(_repo_io(original/'ssl.pt'),map_location='cpu',weights_only=True);assert all(s.torch.equal(v,old['state_dict'][k]) for k,v in m.state_dict().items());scope['fit']=True
    b=s.phase(m,px,reg,meta,s.SEED+20000,s.EPOCHS,r['strength'],((labels-meta['target_mean'])/meta['target_std']).astype(np.float32));scope['fit']=False;old=s.torch.load(_repo_io(original/'model.pt'),map_location='cpu',weights_only=True);assert all(s.torch.equal(v,old['state_dict'][k]) for k,v in m.state_dict().items());p,_=s.predict(m,d['pixels'][q],meta);assert np.array_equal(p,np.load(_repo_io(original/'predictions.npz'))['prediction'])
    s.torch.save(dict(state_dict=m.state_dict(),normalization=meta),_repo_io(folder/'model.pt'));s.save(_repo_io('training-replay.json'),dict(executed=True,epochs=2*s.EPOCHS,outside_labels_changed=len(outside),SSL_states_exact=True,supervised_states_exact=True,predictions_exact=True,dataset_reads_blocked=True,seconds=a['seconds']+b['seconds'],total_neural_seconds=completed['neural_seconds']+a['seconds']+b['seconds']))
if __name__=='__main__':
    sys.addaudithook(s.guard);s.torch.set_num_threads(2);s.torch.set_num_interop_threads(1);s.torch.use_deterministic_algorithms(True)
    with s.threadpool_limits(limits=2):
        if len(sys.argv)>1 and sys.argv[1]=='replay':replay()
        else:main()
