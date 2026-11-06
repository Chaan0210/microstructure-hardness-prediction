"""Independent recomputation of pilot gate; never loads hardness labels."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,hashlib,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import formation_pilot_049 as p
import evaluate_formation_049 as ev

def main():
    start=time.perf_counter();out=p.OUT
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text());manifest=json.loads((_repo_path(ROOT, 'artifacts/run-001/split_manifest.json')).read_text())
    pool=sorted(k for k,v in manifest['development_folds'].items() if v!=0)
    expected=sorted(np.random.default_rng(p.SEED).choice(pool,40,replace=False).tolist());assert expected==plan['ids'];assert len(set(expected))==40
    assert all(manifest['development_folds'][i]!=0 for i in expected)
    rows=[json.loads((out/f'{i}.json').read_text()) for i in expected]
    B=[];C=[];finite_count=0
    for i,row in zip(expected,rows):
        assert row['ID']==i
        assert row['image_sha256']==p.sha(_repo_path(ROOT, 'data/train', f'{i}.png'))
        assert [v['variant'] for v in row['variants']]==plan['variants']
        B.append([v['baseline'] for v in row['variants']]);C.append([v['candidate'] for v in row['variants']])
        for v in row['variants']:
            for key,val in v['diagnostics'].items():assert np.isfinite(np.asarray(val)).all(),(i,key)
            finite_count+=1
        with np.load(_repo_io(out/f'{i}.npz')) as a:
            for key in a.files:assert np.isfinite(a[key]).all(),(i,key)
            assert np.allclose(np.median(a['start_features'],axis=0),row['variants'][0]['candidate'],rtol=0,atol=0)
    B=np.asarray(B);C=np.asarray(C);assert B.shape==C.shape==(40,4,4);assert np.isfinite(B).all() and np.isfinite(C).all()
    scale=np.maximum(np.quantile(B[:,0],.75,axis=0)-np.quantile(B[:,0],.25,axis=0),plan['stability_gate']['floor_scale'])
    sb=np.quantile(np.sqrt(((B[:,1:]-B[:,0,None])**2).mean(1)),.5,axis=0)/scale
    sc=np.quantile(np.sqrt(((C[:,1:]-C[:,0,None])**2).mean(1)),.5,axis=0)/scale
    assert np.allclose(sb,report['stability']['baseline'],atol=1e-14,rtol=0)
    assert np.allclose(sc,report['stability']['candidate'],atol=1e-14,rtol=0)
    passed=bool(np.all(sb>1e-8) and np.all(sc/sb<=.7));assert passed==report['stability']['passed']
    fixture_rows=json.loads((out/'fixtures.json').read_text());fixture_metric_error=0
    for f in fixture_rows:
        with np.load(_repo_io(out/f"fixture-{f['layout']}-{f['severity']}.npz")) as a:
            assert np.isfinite(a['u']).all() and np.isfinite(a['d']).all()
            for name,tol in [('boundary',2),('pore',0)]:
                actual=ev.pr(a[name],a[name+'_truth'],tol)
                for metric in ['precision','recall']:
                    delta=abs(actual[metric]-f[name][metric]);fixture_metric_error=max(fixture_metric_error,delta);assert delta==0
    with np.load(_repo_io(out/'benchmark.npz')) as a:
        repeat_error=float(np.max(np.abs(a['features']-C[0,0])))
    assert repeat_error<1e-9
    for file,hash_ in plan['sources'].items():assert p.sha(_repo_path(ROOT, file))==hash_,file
    for file,hash_ in plan['protected'].items():assert p.sha(_repo_path(ROOT, file))==hash_,file
    for file,hash_ in report['sources'].items():assert p.sha(_repo_path(ROOT, file))==hash_,file
    assert not report['hypothesis_promoted'];assert not report['regression_or_cv_executed']
    diagnostics=[]
    for k,name in enumerate(report['names']):
        diagnostics.append({'descriptor':name,'baseline_sensitivity':float(sb[k]),'new_sensitivity':float(sc[k]),'relative_reduction':float(1-sc[k]/sb[k]),'gate_pass':bool(sc[k]/sb[k]<=.7),'baseline_identity_iqr':float(np.quantile(B[:,0,k],.75)-np.quantile(B[:,0,k],.25)),'candidate_identity_iqr':float(np.quantile(C[:,0,k],.75)-np.quantile(C[:,0,k],.25)),'spread_gate_informative':bool(np.quantile(B[:,0,k],.75)-np.quantile(B[:,0,k],.25)>1e-8),'baseline_identity_range':[float(B[:,0,k].min()),float(B[:,0,k].max())],'candidate_identity_range':[float(C[:,0,k].min()),float(C[:,0,k].max())]})
    v={'valid':True,'label_blind_ids_match':True,'n_images':40,'n_conditions':finite_count,'all_finite':True,'sensitivity_recalculation_max_error':float(max(np.max(np.abs(sb-report['stability']['baseline'])),np.max(np.abs(sc-report['stability']['candidate'])))),'fixture_mask_recalculation_max_error':fixture_metric_error,'benchmark_repeat_feature_max_error':repeat_error,'protected_best006_unchanged':True,'protected_sha256':plan['protected'],'report_sha256':p.sha(out/'report.json'),'plan_sha256':p.sha(out/'plan.json'),'descriptors':diagnostics,'audit_caveats':['Two-start output masks are unions, not consensus. Precision and per-start feature disagreement checks prevent recall-only promotion; this run fails strongly.','Noise parameter is conditional AR innovation scale, not observed marginal residual standard deviation. HH calibration is a proxy.','Affine stability partly comes from image-local percentile normalization and cannot be attributed solely to joint inverse rendering.','IQR ratio gate is uninformative when baseline IQR is near zero; raw IQR and informativeness are reported per descriptor.','Initialization disagreement and degeneracy gates concern identity images only, not all perturbed versions.','This evaluator checks the recorded implementation, not correctness of synthetic generator assumptions.'],'seconds':time.perf_counter()-start}
    identity=[row['variants'][0]['diagnostics'] for row in rows]
    sigmas=np.array([d['blur_sigma'] for d in identity]).ravel()
    ratios=np.array([np.array(d['noise_normalized'])/d['noise_initial_normalized'] for d in identity]).ravel()
    v['posthoc_nuisance_diagnostics']={'scope':'Identity only, explanatory; not a revised gate','blur_sigma_quantiles':np.quantile(sigmas,[0,.5,1]).tolist(),'innovation_to_initial_HH_ratio_quantiles':np.quantile(ratios,[0,.5,1]).tolist(),'innovation_near_lower_bound_fraction':float(np.mean(ratios<.55)),'warning':'A small fitted innovation scale and blur can indicate fitting observed noise into latent fields; not proof of a unique cause.'}
    v['fixture_summary']={'boundary_precision_range':[min(f['boundary']['precision'] for f in fixture_rows),max(f['boundary']['precision'] for f in fixture_rows)],'grain_relative_error_range':[min(f['grain_relative_error'] for f in fixture_rows),max(f['grain_relative_error'] for f in fixture_rows)],'pore_precision_by_fixture':[f['pore']['precision'] for f in fixture_rows]}
    p.save(_repo_path(ROOT, 'reports/formation-049-validation.json'),v)
    text='Joint image-formation pilot049: rejected before regression.\n'
    text+='40 original fold0 training images,4 conditions each; no labels/holdout/test.\n'
    for d in diagnostics:text+=f"{d['descriptor']}: sensitivity reduction {100*d['relative_reduction']:.2f}% (required >=30%).\n"
    text+=f"Fixtures passed {sum(report['preservation']['fixture_pass'])}/6; degeneracy {report['preservation']['degenerate_images']}/40.\n"
    text+=f"Pilot elapsed {report['seconds']:.2f}s, peak RSS {report['rss_bytes']/1e6:.1f}MB.\n"
    text+='Model assumptions: bounded isotropic blur, continuous region prior and sparse dark layer, conditional separable AR residuals. Not an identified official generator or paper reproduction.\n'
    text+='Failures do not establish competition noise floor or impossibility of other inverse models. No follow-up residual CV, candidate or submission.\n'
    text+='Best006 retained, Public12.2224405924 from parent confirmation; current rank/quota not checked.\n'
    text+='Reproduce in a fresh copy retaining inputs, because outputs are exclusive-create:\n.venv/bin/python -I -B formation_pilot_049.py setup\n.venv/bin/python -I -B formation_pilot_049.py benchmark\n.venv/bin/python -I -B evaluate_formation_049.py\n.venv/bin/python -I -B validate_formation_049.py\n'
    (_repo_path(ROOT, 'reports/formation-049-results.txt')).write_text(text)
    status_path=_repo_path(ROOT, 'reports/status.json');s=json.loads(status_path.read_text())
    from datetime import datetime,timezone
    s.update(status='formation049_completed_rejected_no_candidate',latest_research='reports/formation-049-summary.json',latest_completed='reports/formation-049-summary.json',latest_verification='reports/formation-049-validation.json',latest_research_promoted=False,running_work=[],new_submission_candidate=None,current_study_submission_performed=False,current_study_cost=0,updated_at_utc=datetime.now(timezone.utc).isoformat())
    status_path.write_text(json.dumps(s,indent=2))
    print(json.dumps(v),flush=True)
if __name__=='__main__':main()
