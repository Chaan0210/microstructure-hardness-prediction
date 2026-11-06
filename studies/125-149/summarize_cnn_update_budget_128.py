"""Conclude the prespecified128 budget/data contrast with400-ID uncertainty."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,datetime
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cnn_update_budget_128 as q
import numpy as np
e=q.e

def main():
    plan=q.check();decision=e.read(q.OUT/'screen-decision.json');splits=list(e.SPLITS) if decision['advance'] else ['original'];Y=[];P=[];recs=[];valid={};rows=[];evidence={};exposure=[]
    for split in splits:
        path=q.OUT/split;r=e.read(path/'report.json');v=e.read(q.OUT/f'validation-{split}.json');assert v['passed'] and v['report_sha256']==e.sha(path/'report.json');valid[split]=v
        assert e.sha(path/'predictions.npz')==r['predictions_sha256']
        with np.load(_repo_io(path/'predictions.npz')) as z:
            if Y:np.testing.assert_array_equal(z['y'],Y[0]);assert z['ids'].tolist()==ids
            ids=z['ids'].tolist();Y.append(z['y'].copy());P.append(z['predictions'].copy())
        recs.append(r);rows.append({k:v for k,v in r.items() if k not in ['folds','predictions_sha256','seconds']})
        for fold in r['folds']:
            for arm,info in fold['arms'].items():
                folder=path/f'fold-{fold["fold"]}-{arm}'
                with np.load(_repo_io(folder/'streams.npz')) as z:
                    order=z['order'];counts=np.bincount(order.ravel(),minlength=info['n_train']);duplicate=np.array([32-len(set(x.tolist())) for x in order]);exposure.append({'split':split,'fold':fold['fold'],'arm':arm,'n_train':info['n_train'],'updates':info['updates'],'example_presentations':int(order.size),'minimum_presentations_per_image':int(counts.min()),'maximum_presentations_per_image':int(counts.max()),'boundary_batch_duplicate_presentations':int(duplicate.sum()),'mean_distinct_images_per_batch':float(32-duplicate.mean())})
        for f in [path/'report.json',path/'predictions.npz',q.OUT/f'validation-{split}.json']:evidence[str(f.relative_to(ROOT))]=e.sha(f)
    Y=np.stack(Y);P=np.stack(P);E=(P-Y[:,None,:])**2;perid=E.mean(0);R=np.sqrt(perid.mean(1));draw=np.random.default_rng(20271109).integers(0,400,(10000,400));boot=np.sqrt(perid[:,draw].mean(-1));contrasts=[]
    for label,a,b in [('small_budget',0,1),('matched800_data',1,2)]:
        delta=boot[a]-boot[b];contrasts.append({'contrast':label,'reference':q.ARMS[a],'new':q.ARMS[b],'RMSE_gain':float(R[a]-R[b]),'MSE_gain':float(perid[a].mean()-perid[b].mean()),'conditional_ID_cluster95_RMSE_gain_CI':np.quantile(delta,[.025,.975]).tolist(),'per_partition_RMSE_gain':[float(np.sqrt(E[s,a].mean())-np.sqrt(E[s,b].mean())) for s in range(len(splits))]})
    train={}
    for arm in q.ARMS:
        rs=[f['arms'][arm] for r in recs for f in r['folds']];train[arm]=float(np.sqrt(sum(r['n_train']*r['train_RMSE']**2 for r in rs)/sum(r['n_train'] for r in rs)))
    maxerr=0.
    for b,ix in enumerate(draw[:200]):
        counts=np.bincount(ix,minlength=400);manual=np.sqrt(perid@counts/400);maxerr=max(maxerr,float(abs(manual-boot[:,b]).max()))
    assert maxerr<1e-12
    budget_help=contrasts[0]['RMSE_gain']>0;datagain=contrasts[1]['RMSE_gain'];legacy=e.read(q.OUT/'legacy-control/report.json')
    result={'study':'128 pairedCNN updatebudget/data-size contrast','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed;014 unchanged','screen_decision':decision,'partitions_completed':splits,'pooled_validation_RMSE':dict(zip(q.ARMS,R.tolist())),'pooled_training_RMSE':train,'contrasts':contrasts,'per_partition':rows,'sampling_exposure':exposure,'protocol':{k:plan[k] for k in ['CNN','sampling','pairing','scheduler','weight_average','selection_scope']},'interpretation':{'budget_helped_observed':budget_help,'data_gain_at_matched800':datagain,'no_automatic_deployment':'This is a diagnostic ofsmall142/143 models undernewfullbatchstreamprotocol. Evena small800gain doesnotestablish thatcurrent320-rowCNN shouldrun1600updates, nor thatwhole014 improves. No newcandidate ormixtureweight evaluated.','scheduler_limit':'400and800 complete the same normalizedcosineprogress anduse5 equallynormalizedlate checkpoints. Their absoluteLRsequences differ; this isbudgetwithstretchedschedule,notpureupdatecount atidenticalLRprefix. Small800/large800 haveidenticalabsoluteLRandbatchbudgets.','sampler_limit':'Concatenatedpermutationskeepeverybatch32 anddropnoexamples. Smalln boundariesmayrepeat anID withinabatch; distinctimagecountsarequantified. The samequeuealgorithm isused foralln, but thisisnotexactlegacyvariablelastbatch sampling. Legacyprotocol reproduced separately.','historical_bias':'Same400knownexamples/partitions usedadaptively. Screenisresearchcomputeallocation,not independentmodelselection. BootstrapkeepsallpartitionpredictionsforanID together,conditionsonfits,and omitsrefitting,screeningandhistoricalsearchuncertainty.','seed_limit':'Oneprespecifiedpairedinitializationperouterfold,notindependentmulti-seedvalidation. No chosenbestseed.'},'legacy_reproduction':{k:v for k,v in legacy.items() if k!='log'},'independent_validation':valid,'bootstrap_count_fixture_max_error':maxerr,'execution':{'main_models':15*len(splits),'legacy_reproduction_fits':1,'heldout_label_refits':1,'main_training_seconds':sum(r['seconds'] for r in recs),'new_packages':0,'cost':0,'holdout_test_access':False,'full500_fit':False,'new_candidate_CSV':False,'uploaded':False},'next_action':{'keep014':True,'automatic_additional_training':False,'if_screen_failed':'Closefixedcomparison; donottrymoreepochs/schedules/seeds. Dataatmatchedbudget maybeusefulbutnoindependentlabelsavailableandholdoutremainssealed.','if_screen_passed':'Reportbudgeteffectandresplitstability. Any320-rowlongerCNNor014integrationrequiresaseparatefixedcomparison;128doesnotpromotethesemodels.'},'evidence_sha256':{**evidence,**{str(f.relative_to(ROOT)):e.sha(f) for f in [q.OUT/'plan.json',q.OUT/'screen-decision.json',q.OUT/'legacy-control/report.json']}},'source_sha256':e.sha(Path(__file__)),'execution_blockers':[]}
    e.write(_repo_path(ROOT, 'reports/cnn-update-budget-128-summary.json'),result);q.check();e.emit({'status':result['status'],'screen':decision,'RMSE':result['pooled_validation_RMSE'],'train_RMSE':train,'contrasts':contrasts,'fits':result['execution']})
if __name__=='__main__':main()
