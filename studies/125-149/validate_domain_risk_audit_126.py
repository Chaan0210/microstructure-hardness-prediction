"""Independent rule arithmetic, outer exclusion, cache reconstruction and raw replay."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import domain_risk_audit_126 as q
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import diagnose_errors_068 as raw068
import diagnose_images_094 as raw094
import denoised_grain_features as morph
e=q.e

def manual_rule(X,delta):
    candidates=[]
    for j,name in enumerate(q.NAMES):
        values=sorted(float(v) for v in X[:,j]);n=len(values)
        for side,fraction in [('low',.25),('high',.75)]:
            pos=(n-1)*fraction;i=int(pos);t=pos-i
            cut=values[i]+t*(values[min(i+1,n-1)]-values[i])
            ix=[i for i in range(n) if (X[i,j]<=cut if side=='low' else X[i,j]>=cut)]
            rest=[i for i in range(n) if i not in ix]
            if .15*n<=len(ix)<=.4*n:
                hi=sum(float(delta[i]) for i in ix)/len(ix);lo=sum(float(delta[i]) for i in rest)/len(rest)
                candidates.append((hi-lo,name,j,side,cut,len(ix),hi,lo))
    return max(candidates,key=lambda x:x[0]),candidates

def main():
    q.check();start=time.perf_counter();report=e.read(q.OUT/'report.json');assert e.sha(q.OUT/'predictions.npz')==report['predictions_sha256']
    # Fixed synthetic example: known high first-column loss change, deterministic ties.
    X=np.zeros((100,12));X[:,0]=np.arange(100);X[:,1]=X[:,0];delta=np.where(X[:,0]>=75,100.,-10.)
    fixture=q.fit_rule(X,delta);assert fixture['chosen']['column']==0 and fixture['chosen']['side']=='high'
    np.testing.assert_array_equal(q.apply_rule(fixture,X),X[:,0]>=75)
    ids,features=q.features();manualerr=0.;decisionerr=0.;nfit=0;replay=[];fulloutputs=[]
    with np.load(_repo_io(q.OUT/'predictions.npz'),allow_pickle=False) as z:
        assert z['ids'].tolist()==ids;np.testing.assert_array_equal(z['X'],features)
        for split in e.SPLITS:
            d,X,base,final,nested=q.load(split);saved=e.read(q.OUT/f'rules-{split}.json')
            assert e.sha(q.OUT/f'rules-{split}.json')==report['rules_sha256'][split]
            H=np.zeros(400,bool)
            for k,cell in enumerate(nested):
                T,V=cell['T'],cell['V'];old=cell['anchor013'];new=cell['anchor014'];delta=(d['y'][T]-new)**2-(d['y'][T]-old)**2
                analytic=(old-new)*(2*d['y'][T]-old-new);err=float(abs(analytic-delta).max());assert err<1e-10;manualerr=max(manualerr,err)
                chosen,allrules=manual_rule(X[T],analytic);m=saved[k];rr=m['chosen'];assert rr['feature']==chosen[1] and rr['side']==chosen[3];assert rr['column']==chosen[2] and rr['n_condition']==chosen[5];assert abs(rr['cut']-chosen[4])<1e-12
                assert len(allrules)==len(m['all_train_rules'])
                for row,manual in zip(m['all_train_rules'],allrules):
                    err=max(abs(row['contrast_MSE']-manual[0]),abs(row['condition_mean_MSE_change']-manual[6]),abs(row['other_mean_MSE_change']-manual[7]));assert err<1e-10;decisionerr=max(decisionerr,err)
                assert m['train_indices']==T.tolist() and m['validation_indices']==V.tolist()
                hv=np.array([(x[chosen[2]]<=chosen[4] if chosen[3]=='low' else x[chosen[2]]>=chosen[4]) for x in X[V]])
                H[V]=hv
                # Upstream anchors have independently verified A/B/V/deep lineage; do not retrain them.
                alter=d['y'].copy();alter[V]+=10000
                poisonedX=X.copy();poisonedX[V]=np.linspace(-1e9,1e9,len(V)*12).reshape(len(V),12)
                original=q.fit_rule(X[T],delta)
                altered=q.fit_rule(poisonedX[T],(alter[T]-new)**2-(alter[T]-old)**2)
                assert original==altered
                assert original['chosen']==rr and original['all_train_rules']==m['all_train_rules']
                np.testing.assert_array_equal(q.apply_rule(altered,X[V]),hv)
                nfit+=1
            np.testing.assert_array_equal(H,z[split+'_risk']);np.testing.assert_array_equal(final,z[split+'_final014']);np.testing.assert_array_equal(d['reference013'],z[split+'_final013'])
            delta=(d['reference013']-final)*(2*d['y']-d['reference013']-final)
            row=next(r for r in report['per_partition'] if r['split']==split)
            contrast=float(delta[H].mean()-delta[~H].mean());assert abs(contrast-row['final']['condition_minus_other_MSE_change'])<1e-10
            fulloutputs.append((d['y'].copy(),d['reference013'].copy(),final.copy(),H.copy()))
            replay.append({'split':split,'all5_outer_label_and_feature_poisoning_passed':True,'independent_rules':5})
    # Resampling by multinomial ID counts independently reproduces the cluster bootstrap.
    Y,O,F,H=map(np.stack,zip(*fulloutputs));delta=(O-F)*(2*Y-O-F);rng=np.random.default_rng(20271105);bc=[];bh=[];bl=[]
    for _ in range(10000):
        ix=rng.integers(0,400,400);counts=np.bincount(ix,minlength=400);hi=counts@np.sum(delta*H,0)/(counts@np.sum(H,0));lo=counts@np.sum(delta*~H,0)/(counts@np.sum(~H,0));bc.append(hi-lo);bh.append(hi);bl.append(lo)
    for key,arr in [('conditional_ID_cluster95_change_contrast_CI',bc),('conditional_ID_cluster95_condition_mean_MSE_change_CI',bh),('conditional_ID_cluster95_other_mean_MSE_change_CI',bl)]:np.testing.assert_allclose(np.quantile(arr,[.025,.975]),report[key],atol=1e-10,rtol=0)
    for pair in report['split_stability']:
        i,j=map(e.SPLITS.index,pair['partitions']);rho=float(np.corrcoef(rankdata(delta[i]),rankdata(delta[j]))[0,1]);assert abs(rho-pair['squared_error_change_Spearman'])<1e-12
    # Re-extract from allowlisted development raw files only. No label-based image selection.
    lock=e.read(_repo_path(ROOT, 'artifacts/image-diagnostics-094/matching-locked.json'));selected=sorted(ids,key=lambda x:hashlib.sha256(('126-feature-replay:'+x).encode()).hexdigest())[:6];rawerr=0.;morpherr=0.
    for i,ident in enumerate(ids):
        file=_repo_path(ROOT, 'data/train', f'{ident}.png');assert e.sha(file)==lock['image_sha256'][ident]
        with Image.open(_repo_io(file)) as im:
            assert im.mode=='L' and im.size==(256,256);raw=np.array(im)
        row=raw068.metrics(raw);row.update(raw094.additional(raw));vals=np.array([row[n] for n in q.IMAGE_NAMES]);err=float(abs(vals-X[i,:8]).max());assert err<1e-12;rawerr=max(rawerr,err)
        if ident in selected:
            values,names,_=morph.extract_denoised(raw);vals=np.array([values[list(names).index(n)] for n in q.MORPH_NAMES]);err=float(abs(vals-X[i,8:]).max());assert err<1e-12;morpherr=max(morpherr,err)
    out={'passed':True,'rule_fits_independently_recomputed':nfit,'rule_scoring_max_error':decisionerr,'paired_squared_loss_identity_max_error':manualerr,'outer_label_and_feature_intervention':replay,'all014_and013_base_anchor_scopes_and_formulae_rechecked':True,'upstream_training_replay_certificate':e.sha(e.OUT/'validation-replay.json'),'all400_raw_image_metrics_reextracted_max_error':rawerr,'morphology_replay6_labelblind_IDs':selected,'morphology_replay_max_error':morpherr,'ID_cluster_bootstrap_independently_reproduced':True,'synthetic_rule_selection_and_tie_fixture_passed':True,'report_sha256':e.sha(q.OUT/'report.json'),'source_sha256':e.sha(Path(__file__)),'seconds':time.perf_counter()-start,'scope':'No upstream retraining; existing training intervention certificate reused. No holdout/test decoding, model correction, upload or cost.'}
    q.check();e.write(q.OUT/'validation.json',out);e.emit(out)
if __name__=='__main__':
    q.p.torch.set_num_threads(2);main()
