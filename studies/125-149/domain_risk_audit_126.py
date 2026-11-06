"""Train-only discovery of image conditions associated with014-minus013 error changes."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys, datetime, time, argparse
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import constrained_weights_125 as protected
import numpy as np
from scipy.stats import spearmanr
p=protected.protected;a=p.a;e=p.e
OUT=_repo_path(ROOT, 'artifacts/domain-risk-audit-126')
IMAGE_NAMES=('image_std','noise_MAD_proxy','noise_contrast_ratio','anisotropy1','autocorr16','tile_mean_std','dark_largest_fraction','spectral_low_fraction')
MORPH_NAMES=('denoised_grain_median_interior_area_weighted_diam','denoised_grain_range_interior_area_weighted_diam','denoised_grain_median_boundary_fraction','denoised_grain_median_interior_area_fraction')
NAMES=IMAGE_NAMES+MORPH_NAMES

def fit_rule(X,loss_change):
    assert X.shape==(len(loss_change),12) and np.isfinite(X).all() and np.isfinite(loss_change).all()
    loss=np.asarray(loss_change,float); records=[]
    for j,name in enumerate(NAMES):
        for side,quant in [('low',.25),('high',.75)]:
            cut=float(np.quantile(X[:,j],quant,method='linear'))
            mask=X[:,j]<=cut if side=='low' else X[:,j]>=cut
            size=int(mask.sum())
            if .15*len(mask)<=size<=.4*len(mask):
                hi=float(loss[mask].mean());lo=float(loss[~mask].mean())
                records.append({'feature':name,'column':j,'side':side,'quantile':quant,'cut':cut,'n_condition':size,'n_other':int((~mask).sum()),'condition_mean_MSE_change':hi,'other_mean_MSE_change':lo,'contrast_MSE':hi-lo})
    assert records
    chosen=max(range(len(records)),key=lambda j:records[j]['contrast_MSE'])
    return {'n_fit':len(loss_change),'chosen':records[chosen],'all_train_rules':records,'tie_rule':'First feature in fixed order, low before high; no fitted threshold beyond training quartile.'}

def apply_rule(m,X):
    r=m['chosen'];x=X[:,r['column']]
    return x<=r['cut'] if r['side']=='low' else x>=r['cut']

def features():
    lock=e.read(_repo_path(ROOT, 'artifacts/image-diagnostics-094/matching-locked.json'))
    f=_repo_path(ROOT, 'artifacts/image-diagnostics-094/image_metrics.npz');assert e.sha(f)==lock['metrics_sha256']
    with np.load(_repo_io(f),allow_pickle=False) as z:
        ids=z['ids'].tolist();X=z['X'][:,[z['names'].tolist().index(n) for n in IMAGE_NAMES]]
    meta=e.read(_repo_path(ROOT, 'artifacts/run-022/report.json'))
    with np.load(_repo_path(ROOT, 'artifacts/run-022/development_features.npz'),allow_pickle=False) as z:
        assert z['ids'].tolist()==ids
        X=np.column_stack([X,z['X'][:,[meta['features'].index(n) for n in MORPH_NAMES]]])
    manifest=e.read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'))
    assert ids==sorted(manifest['development_folds']) and not set(ids)&set(manifest['holdout'])
    assert X.shape==(400,12) and np.isfinite(X).all()
    return ids,X

def check():
    protected.check();plan=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in {**plan['sources_sha256'],**plan['inputs_sha256'],**plan['protected_sha256']}.items():assert e.sha(_repo_path(ROOT, f))==h,f
    return plan

def prepare():
    protected.check();OUT.mkdir(exist_ok=False)
    files=['artifacts/image-diagnostics-094/image_metrics.npz','artifacts/image-diagnostics-094/matching-locked.json','artifacts/run-022/report.json','artifacts/run-022/development_features.npz','artifacts/run-001/split_manifest.json','reports/baseline124-weights125-summary.json']
    for split in e.SPLITS:
        for f in sorted((p.study.OUT/split).rglob('*.json')):files.append(str(f.relative_to(ROOT)))
        for f in sorted((p.study.OUT/split).rglob('*.npz')):files.append(str(f.relative_to(ROOT)))
    review=['reports/diagnostic-research-068-072-summary.json','reports/image-diagnostics-094-assessment.json','reports/grain-visual-audit-097-nlm098-summary.json','reports/frame-features-101-summary.json','reports/blindspot-preflight-102-summary.json','reports/research-batch-105-108-summary.json','reports/input-noise-110-summary.json','reports/sliced-distribution-099-summary.json','reports/frame-censoring-100-summary.json','reports/pipeline-audit-103-patch-phase-104-summary.json','reports/persistence-repair-109-summary.json','reports/problem-definition-111-summary.json','reports/residual-family-117-summary.json','reports/pooled-residual-119-summary.json']
    files+=review
    plan={'study':'126 train-only image-domain014-minus013 residual-change audit','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'decision_before_new_outcomes':'No distinct supported restoration/representation correction follows the failed068-110 families. Use requested fallback: audit whether train-discovered conditions of worse014-minus013 error change transfer to outer validation, rather than retry noise/denoising/threshold/lambda grids.','question':'Does baseline removal and downstream refitting from013 to014 have less favorable error changes under an image condition selected only from outer-training paired anchor errors? This tests transferable heterogeneity in the removal effect, not generic high-error imagery.','duplicate_audit':{'068':'Descriptive correlations and fold-local feature quantile bins for fixed008, but feature/error associations inspected globally.','094':'Image-only controls of five outcome-selected anchors and descriptive fixed011/attention comparisons; not prospective train-only condition discovery.','071_072_097_098_101_102_110':'TV/NLM, stronger watershed, frame features, blindspot/noise assumptions and noise augmentation already failed. No repetition.','099_100_103_104_105_106_107_108_109_111':'Sliced distributions, frame censoring, patch phase, amplitude, persistence, three-point structure and representation audits do not test paired013-to014 error-change condition transfer. No parameter retests.', '113_115_117_119':'Signed-error correction families, not this risk-transfer audit. No hardness prediction changed.','125':'Base blend weights; unrelated to condition-discovery audit.'},'feature_names':list(NAMES),'feature_meaning':'Fixed8 cached image-local068/094 proxies plus4 preexisting022 morphology/support proxies. Use raw per-image X only; never global400 ranks, pair distances or any y-derived feature. Noise and grain proxies are not observed true noise/grain geometry.','selection':'For each outer fold, assemble320 paired honest pre-physics base013/base014 anchor residuals from three B106/107 blocks. Each B anchor was fitted/calibrated exclusively on A213/214 with deeper142/143 fits. Fit one risk rule on320 training rows: each of12 columns has low25% and high25% tail using train-only linear quantile; eligible tail count15%-40%; compute delta=(y-anchor014)^2-(y-anchor013)^2; select maximum train tail-vs-rest mean delta contrast, deterministic fixed-order ties. Apply unchanged to outer80. No inner tuning of risk learner or new residual correction.','scope_warning':'These are paired honest errors of smaller pre-physics013/014 anchors, not cross-fitted complete013/014 residual errors. Final014-minus013 outer loss changes are the primary evaluation; changes in the pre-physics bases are a stage diagnostic. Training-size and model-stage mismatch are explicit. Pooling320 anchor errors is safe for this fixed outer risk procedure, not permission to cross-validate models on pooled residual labels.','primary':'Mean(delta014-minus013 | selected condition) minus mean(delta | other), pooled over3x400 pairs with only400 unique IDs; report actual013/014 RMSE and signed residual changes in each group. A positive contrast need not mean harm: both groups may improve. It is a prospective association audit, not improvement in prediction RMSE, causal evidence, or independent research validation.','stability':'Report selected conditions15folds, risk-label agreement across3partitions, 014-minus013 squared-error-change rank correlations and descriptive top40 most adverse-change overlap. No subgroup is removed, reweighted or corrected.','uncertainty':'10000 paired ID cluster bootstrap draws seed20271105, all three predictions and risk flags travel together. Conditional on fitted rules/models; excludes adaptive research and refitting. No iid1200 inference or permutation p-value.','validation':'Independent sort/interpolate quantiles and loop rule scoring; all15 rules refit after poisoning outer labels and outer feature values; assert decisions unchanged. Reload both anchors and assert yB-anchor014 equals saved residual; independently reconstruct base013 affine too; check group disjointness/A-B-V scopes and reconstruct014 outer sum. Independent raw8 image metrics on400 development images, morphology4 on6 label-blind hash-selected images. Synthetic fixed rule fixtures.','resources':'Existing uv packages, CPU2, no neural fitting/MPS needed. 15 tiny fixed risk-rule fits plus independent checks;400 raw image metric passes and6 morphology re-extractions. Target under5min and2GB; stop and report if exceeded, do not install.','protected_sha256':{str(f.relative_to(ROOT)):e.sha(f) for f in [p.CSV,_repo_path(ROOT, 'reports/candidate014-summary.json'),_repo_path(ROOT, 'reports/candidate014-verification.json'),_repo_path(ROOT, 'reports/candidate014-preservation.json')]},'sources_sha256':{s:e.sha(_repo_path(ROOT, s)) for s in ['domain_risk_audit_126.py','validate_domain_risk_audit_126.py','diagnose_errors_068.py','diagnose_images_094.py','denoised_grain_features.py']},'inputs_sha256':{f:e.sha(_repo_path(ROOT, f)) for f in files},'scope':'Development400 only; holdout/test remain unaccessed; no candidate, full500 fit, submission, install, cost, or upstream edits.'}
    e.write(_repo_path(OUT, 'plan.json'),plan);e.emit({'prepared':True,'features':list(NAMES),'new_hardness_model':False,'plan_sha256':e.sha(_repo_path(OUT, 'plan.json'))})

def load(split):
    ids,X=features();d=protected.load(split);assert d['ids']==ids
    path=p.study.OUT/split;rec=e.read(path/'report.json')
    assert e.sha(path/'predictions.npz')==rec['predictions_sha256']
    with np.load(_repo_io(path/'predictions.npz'),allow_pickle=False) as z:
        base=z['base_no_baseline795'].copy();final=z['no_baseline795'].copy()
    nested=[]
    for k in range(5):
        T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);folder=path/f'no_baseline795-fold-{k}';fr=e.read(folder/'report.json')
        assert e.sha(folder/'report.json')==rec['fold_reports_sha256'][folder.name]
        ff=a.readnp(folder/'predictions.npz',fr['predictions_sha256']);np.testing.assert_array_equal(ff['base'],base[V]);np.testing.assert_array_equal(ff['primary'],final[V]);np.testing.assert_array_equal(ff['primary'],ff['base']+ff['corrections'].mean(0))
        train_anchor=np.full(400,np.nan);old_anchor=np.full(400,np.nan);trace=[]
        for j in range(3):
            A,B,P,short,old=d['cells'][k,j];bp=folder/f'block-{j}';br=e.read(bp/'report.json')
            assert e.sha(bp/'report.json')==fr['block_reports_sha256'][f'block-{j}/report.json']
            zz=a.readnp(bp/'predictions.npz',br['predictions_sha256'])
            assert not set(d['g'][A])&set(d['g'][B]);assert not set(d['g'][T])&set(d['g'][V])
            assert set(A)|set(B)==set(T) and not set(A)&set(B)
            cal,_=a.affine(a.mixture(d['deepP'][k,j],'no_baseline795'),d['y'][A],a.mixture(P,'no_baseline795'))
            np.testing.assert_allclose(.9*cal+.1*short,zz['anchor_predictions'],atol=1e-10,rtol=0)
            np.testing.assert_array_equal(d['y'][B]-zz['anchor_predictions'],zz['residual'])
            assert np.isnan(train_anchor[B]).all();train_anchor[B]=zz['anchor_predictions'];old_anchor[B]=old
            oc,_=a.affine(a.mixture(d['deepP'][k,j],'full'),d['y'][A],a.mixture(P,'full'))
            np.testing.assert_allclose(.9*oc+.1*short,old,atol=1e-10,rtol=0)
            trace.append({'anchor_train_indices':A.tolist(),'residual_block_indices':B.tolist(),'source':str(bp.relative_to(ROOT)),'source_predictions_sha256':br['predictions_sha256']})
        assert np.isfinite(train_anchor[T]).all() and np.isnan(train_anchor[V]).all()
        nested.append({'T':T,'V':V,'anchor014':train_anchor[T],'anchor013':old_anchor[T],'trace':trace})
    return d,X,base,final,nested

def comparison(y,old,new,mask):
    assert mask.any() and (~mask).any()
    delta=(y-new)**2-(y-old)**2
    groups={}
    for name,ix in [('condition',mask),('other',~mask)]:
        groups[name]={'n':int(ix.sum()),'RMSE013':e.rmse(y[ix],old[ix]),'RMSE014':e.rmse(y[ix],new[ix]),'mean_MSE_change014_minus013':float(delta[ix].mean()),'mean_residual013':float((y-old)[ix].mean()),'mean_residual014':float((y-new)[ix].mean())}
    groups['condition_minus_other_MSE_change']=float(delta[mask].mean()-delta[~mask].mean())
    return groups

def run():
    check();start=time.perf_counter();outputs={};parts=[];models=[]
    for split in e.SPLITS:
        d,X,base,final,nested=load(split);risk=np.zeros(400,bool);records=[]
        for k,cell in enumerate(nested):
            T,V=cell['T'],cell['V'];delta=(d['y'][T]-cell['anchor014'])**2-(d['y'][T]-cell['anchor013'])**2
            m=fit_rule(X[T],delta);risk[V]=apply_rule(m,X[V]);m.update(split=split,fold=k,train_indices=T.tolist(),validation_indices=V.tolist(),anchor_sources=cell['trace']);records.append(m);models.append(m)
        e.write(_repo_path(OUT, f'rules-{split}.json'),records)
        outputs.update({split+'_y':d['y'],split+'_fold':d['fid'],split+'_base014':base,split+'_base013':d['reference011'],split+'_final014':final,split+'_final013':d['reference013'],split+'_risk':risk})
        row={'split':split,'final':comparison(d['y'],d['reference013'],final,risk),'base_before_residual':comparison(d['y'],d['reference011'],base,risk),'selected_conditions':[m['chosen']['feature']+':'+m['chosen']['side'] for m in records]};parts.append(row);e.emit(row)
    ids=d['ids'];Y=np.stack([outputs[s+'_y'] for s in e.SPLITS]);F=np.stack([outputs[s+'_final014'] for s in e.SPLITS]);O=np.stack([outputs[s+'_final013'] for s in e.SPLITS]);B=np.stack([outputs[s+'_base014'] for s in e.SPLITS]);OB=np.stack([outputs[s+'_base013'] for s in e.SPLITS]);H=np.stack([outputs[s+'_risk'] for s in e.SPLITS]);change=(Y-F)**2-(Y-O)**2
    draw=np.random.default_rng(20271105).integers(0,400,(10000,400));n=H.sum(0);ls=(change*H).sum(0);nl=(change*~H).sum(0);hn=n[draw].sum(1);ln=3*400-hn;hd=ls[draw].sum(1)/hn;ld=nl[draw].sum(1)/ln
    pair=[]
    for i in range(3):
        for j in range(i+1,3):
            topi=set(np.argsort(-change[i],kind='stable')[:40]);topj=set(np.argsort(-change[j],kind='stable')[:40]);pair.append({'partitions':[e.SPLITS[i],e.SPLITS[j]],'squared_error_change_Spearman':float(spearmanr(change[i],change[j]).statistic),'top40_adverse_change_overlap':len(topi&topj),'risk_flag_agreement':float(np.mean(H[i]==H[j])),'risk_flag_Jaccard':float(np.sum(H[i]&H[j])/np.sum(H[i]|H[j]))})
    chosen=[m['chosen']['feature']+':'+m['chosen']['side'] for m in models]
    np.savez_compressed(_repo_path(OUT, 'predictions.npz'),ids=np.array(ids),feature_names=np.array(NAMES),X=X,**outputs)
    report={'study':'126 train-only domain013-to014 residual-change audit','status':'audit_executed','per_partition':parts,'pooled_final':comparison(Y,O,F,H),'pooled_base_before_residual':comparison(Y,OB,B,H),'conditional_ID_cluster95_change_contrast_CI':np.quantile(hd-ld,[.025,.975]).tolist(),'conditional_ID_cluster95_condition_mean_MSE_change_CI':np.quantile(hd,[.025,.975]).tolist(),'conditional_ID_cluster95_other_mean_MSE_change_CI':np.quantile(ld,[.025,.975]).tolist(),'selected_condition_counts':{key:chosen.count(key) for key in sorted(set(chosen))},'train_selected_mean_MSE_change_contrast':float(np.mean([m['chosen']['contrast_MSE'] for m in models])),'split_stability':pair,'rule_fit_count':15,'predictions_sha256':e.sha(_repo_path(OUT, 'predictions.npz')),'rules_sha256':{s:e.sha(_repo_path(OUT, f'rules-{s}.json')) for s in e.SPLITS},'plan_sha256':e.sha(_repo_path(OUT, 'plan.json')),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start,'limits':check()['scope_warning']+' Same400 historically reused; conditional bootstrap does not validate causal image domains. No fitted prediction correction or claim of lower RMSE.'}
    e.write(_repo_path(OUT, 'report.json'),report);check();e.emit({k:v for k,v in report.items() if k not in ['per_partition','limits','rules_sha256']})

if __name__=='__main__':
    p.torch.set_num_threads(2)
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    prepare() if args.mode=='prepare' else run()
