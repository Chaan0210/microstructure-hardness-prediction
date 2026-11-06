"""Actual readout poison replays and independent nested assembly arithmetic."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
import readout as r
import numpy as np,json,sys,math,gc,time
import torch
def predict(X,y,Q,seed,device):
    m=r.TabPFNRegressor.create_default_for_version(r.ModelVersion('v3.5'),model_path=str(r.WEIGHT),device=device,inference_precision=torch.float32,memory_saving_mode=True,random_state=seed,n_preprocessing_jobs=1,show_progress_bar=False);m.fit(X,y.copy());p=np.asarray(m.predict(Q),float);small=np.asarray(m.predict(Q[:8]),float);del m;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return p,small
def main():
    start=time.monotonic();d=r.physical();result=json.loads((c.HERE/'results.json').read_text());o=np.load(_repo_io(c.HERE/'oof.npz'));checks=[];arithmetic={};replays=[]
    for k,g in enumerate(d['graphs']):
        an=np.load(_repo_io(c.OLD/f'anchors/fold{k}.npz'));B=an['B'];V=an['V'];q=np.flatnonzero(an['highB']);a=np.load(_repo_io(c.HERE/f'readouts/fold{k}-selector/predictions.npz'));v=np.load(_repo_io(c.HERE/f'readouts/fold{k}-refit/predictions.npz'));poison=d['y'].copy();poison[V]+=10000
        for arm in r.ARMS:
            delta=(a[arm]-a['base']).mean(0)[:len(B)];num=math.fsum(float(delta[j])*float(poison[B[j]]-an['p82B'][j]) for j in q);den=math.fsum(float(delta[j])**2 for j in q)+len(q);beta=max(0.,min(.5,num/den));record=result['selection'][k]['arms'][arm]['beta'];assert abs(beta-record)<1e-12;expected=an['p82V']+an['highV']*record*(v[arm]-v['base']).mean(0);assert np.array_equal(expected,o[arm][an['ix']]);checks.append(dict(fold=k,arm=arm,beta=beta,outer_label_poison_unchanged=True,assembly_exact=True))
    for arm in r.ARMS:
        full=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(o[arm],o['y']))/len(o['y']));high=o['high'];noise=math.sqrt(math.fsum((float(p)-float(y))**2 for p,y in zip(o[arm][high],o['y'][high]))/int(high.sum()));assert abs(full-result['results'][arm]['candidate_rmse'])<1e-12 and abs(noise-result['results'][arm]['high_noise']['candidate_rmse'])<1e-12;arithmetic[arm]=dict(overall=full,high=noise)
    tr=np.array(d['graphs'][0]['A']);outside=np.setdiff1d(np.arange(500),tr);labels=d['y'].copy();labels[outside]+=10000
    for arm in ['topology','local']:
        folder=c.HERE/f'readouts/fold0-selector/{arm}-seed0';ctx=np.load(_repo_io(folder/'context.npz'));assert np.array_equal(labels[tr],ctx['y']);old=np.load(_repo_io(folder/'tab0/prediction.npz'))['prediction'];p,small=predict(ctx['X'],labels[tr],ctx['Q'],0,'mps');err=float(np.max(abs(p-old)));subset=float(np.max(abs(small-p[:8])));assert err<1e-5 and subset<1e-4;cpu,_=predict(ctx['X'],labels[tr],ctx['Q'][:8],0,'cpu');cpuerr=float(np.max(abs(cpu-p[:8])));assert cpuerr<1e-3;replays.append(dict(arm=arm,outside_labels_changed=len(outside),MPS_replay_error=err,query_subset_error=subset,CPU_MPS_HV_error=cpuerr))
    gate=result['results']['topology'];comp=result['topology_vs_controls'];expected=dict(overall_gain05=gate['gain']>=.05,high_gain=gate['high_noise']['gain']>0,three_folds=gate['improved_folds']>=3,CI_vs082=gate['conditional_CI95'][0]>0,CI_vs_each_control=all(z['conditional_CI95'][0]>0 for z in comp.values()),top10=gate['removed_largest_gains']['10']['gain']>0,both_seeds=all(z['gain']>=0 for z in gate['seeds']));decision=json.loads((c.HERE/'decision.json').read_text());assert expected==decision['checks'] and all(expected.values())==decision['advance']
    c.save(_repo_io('final-verification.json'),dict(all_passed=True,calibration=checks,independent_RMSE=arithmetic,selected_actual_replays=replays,decision_math_exact=True,seconds=time.monotonic()-start,source_sha256=c.sha(__file__)));print('FINAL NUMERICAL VERIFICATION PASSED',len(checks),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.mps.set_per_process_memory_fraction(.27);sys.addaudithook(c.guard)
    with c.threadpool_limits(limits=2):main()
