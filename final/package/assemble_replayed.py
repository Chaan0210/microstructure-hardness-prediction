"""Combine independently replayed heads, preserving frozen coefficients.

This path measures backend/numerical differences. It never replaces the selected
byte-exact CSV. Raw physical features and CNN context remain frozen caches;
source-flow features can be supplied from the new raw-image inference.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import joblib
from reproduce import PROJECT,H,F,SOURCE,TAGS,sha,dump,csv_read,csv_write

def main(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    standard=Path(a.standard);tab=Path(a.tab);source=Path(a.source_tab);cnn=Path(a.cnn)
    tmeta=json.loads((tab/'verification.json').read_text());smeta=json.loads((source/'verification.json').read_text())
    assert tmeta['complete'] and tmeta['members']==72 and smeta['complete'] and smeta['members']==9
    C=np.load(PROJECT/'experiments/082-fusion-20261018-0155/reproduction/incumbent-components.npz')
    N=np.load(standard/'incumbent-neural-replay.npz');L=np.load(standard/'incumbent-linear-replay.npz');G=np.load(standard/'incumbent-gmm-replay.npz')
    T=np.load(tab/'predictions.npz');S=np.load(source/'predictions.npz');R=np.load(cnn/'cnn-predictions.npz')
    ids=C['ids'];failed=C['failed'];high=C['high'];low=C['low'];eligible=~failed
    for z in [N,L,G,T,S,R]:assert np.array_equal(z['ids'],ids)
    tree=PROJECT/'experiments/rmse-research-20261015/round4-full073/final500'
    coefficient=json.loads((tree/'recipe.json').read_text())['production_weights']['addition'][1]
    def tab_mean(z):return np.where(low,np.mean([z['nb_'+s] for s in ['038','039','044']],0),np.mean([z['full_'+s] for s in ['038','039','044']],0))
    cached=np.load(PROJECT/'experiments/push-1300/fin-h9s2-tab-cpu.npz');cr=np.load(tree/'test-predictions.npz')
    original_set=np.load(PROJECT/'experiments/probabilistic-grains-900/set_model/full500/predictions.npz')
    # make_053.py uses float32 set_mean_raw, not the float64 mean obtained
    # by applying fallback to the individual seed predictions first.
    direct=(.16*C['G']+.384*tab_mean(cached)+.128*C['Ridge']+.128*C['logRidge']+.05*original_set['set_mean_raw']+.15*C['GMM']+high*coefficient*cr['seed_corrections'].mean(0))
    direct[failed]=C['G'][failed]
    canonical=csv_read(PROJECT/'artifacts/submission-candidate-083.csv')[1]
    construction_error=float(abs(direct-canonical).max());assert construction_error<1e-10
    base=(.16*N['G']+.384*tab_mean(T)+.128*np.where(high,L['B_ridge'],L['nb_ridge'])
          +.128*np.where(high,L['B_logridge'],L['nb_logridge'])+.05*N['Set'].astype(original_set['set_mean_raw'].dtype)
          +.15*np.where(high,G['gmm_full'],G['gmm_nb'])+high*coefficient*R['seed_corrections'].mean(0))
    base[failed]=N['G'][failed]
    physical=np.load(PROJECT/'experiments/push-1300/abl/feat-h9s2-full-044-u1dmavg-test.npz')
    assert np.array_equal(physical['ids'],ids)
    P=physical['X'].astype(float);pred={'control':joblib.load(F/'models/boostedreadout-control.joblib')['model'].predict(P)}
    flowdir=Path(a.flow_dir) if a.flow_dir else SOURCE/'features';feature_checks={}
    for tag in TAGS:
        path=flowdir/f'{tag}-test.npz';z=np.load(path);old=np.load(SOURCE/f'features/{tag}-test.npz')
        assert np.array_equal(z['ids'],ids) and np.array_equal(z['flow'],old['flow']), 'Fresh flow must match the frozen feature arrays'
        feature_checks[tag]={'sha256':sha(path),'exact_frozen_array':True}
        pred[tag]=joblib.load(F/f'models/boostedreadout-{tag}.joblib')['model'].predict(np.c_[P,z['flow']])
    final=base+eligible*(.384*(np.mean([S[tag] for tag in TAGS],0)-S['control'])+.256*(np.mean([pred[tag] for tag in TAGS],0)-pred['control']))
    assert np.isfinite(final).all() and np.array_equal(final[failed],N['G'][failed])
    reference=csv_read(F/'candidate-source-flow38-boostedreadout.csv')[1]
    target=out/'replayed-all-heads.csv';csv_write(target,ids,final)
    err=float(abs(final-reference).max());rms=float(np.sqrt(np.mean((final-reference)**2)))
    report=dict(complete=True,rows=1000,all_ordered_unique_ids=True,max_abs_prediction_difference=err,rms_prediction_difference=rms,
        cached_component_formula_max_difference=construction_error,csv_byte_identical=sha(target)==sha(F/'candidate-source-flow38-boostedreadout.csv'),
        set_mean_precision='Original053 float32 mean of rawseed outputs;0.05 multiplication stays float32 before float64 ensemble addition.',
        csv_sha256=sha(target),absolute_tolerance_HV=.002,
        tolerance_source='Preexisting full076 saved-model numerical replay tolerance; fixed before this combined062 result.',
        source_flow_regenerated=bool(a.flow_dir),feature_checks=feature_checks,
        scope='All72 selected inherited TabPFN contexts and9source/control contexts refit; linear/GMM heads refit; G018/G021/Set saved states inferred; residualCNN raw318images with cached context; HGB saved states inferred. Physical upstream features remain frozen caches. No full auxiliary retraining.',
        primary_submission_replaced=False,public_private_scores_belong_to_original_csv_only=True)
    dump(out/'verification.json',report);assert err<.002,report
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['standard','tab','source-tab','cnn','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--flow-dir');main(p.parse_args())
