"""Fixed, paired CNN RNC representation screen; project data and torch only."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import sys,time,datetime,argparse,hashlib,gc,json
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import binary_threepoint_study_130 as previous
import cnn_orientation_129 as old
import numpy as np,torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
cal=old.cal;e=old.e;engine=old.engine;OUT=_repo_path(ROOT, 'artifacts/rnc-131');ARMS=('mse','rnc');STEPS=800;SEED=20271115

def write(p,x):
    data=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x') as f:f.write(data)
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def correction():
    previous.check();src=_repo_path(ROOT, 'reports/binary-threepoint-130-summary.json');x=e.read(src)
    statement='The standalone three-point negative finding was relayed by the parent task from its public-paper research. This worker did not directly verify the full text. It was not first-hand user-provided evidence.'
    files=[src,previous.OUT/'feature-plan.json',_repo_path(ROOT, 'summarize_binary_threepoint_130.py'),_repo_path(ROOT, 'README.md')]
    c={'correction':statement,'supersedes':'130 second-paper negative_evidence and summary limitation attribution; no result or feature change','original_sha256':{str(p.relative_to(ROOT)):e.sha(p) for p in files},'preserve_original_reports':True,'reference014_retained':True}
    cp=_repo_path(ROOT, 'reports/binary-threepoint-130-provenance-correction.json');write(cp,c)
    x['paper_evidence'][1]['negative_evidence']=statement
    x['limitations']=[statement if 'user-providedcontext' in s else s for s in x['limitations']]
    x['provenance_correction']={'file':str(cp.relative_to(ROOT)),'sha256':e.sha(cp),'original_summary_sha256':e.sha(src)}
    write(_repo_path(ROOT, 'reports/binary-threepoint-130-summary-corrected.json'),x)
    p=_repo_path(ROOT, 'README.md');before=e.sha(p);assert before==e.read(_repo_path(ROOT, 'reports/readme-update-130.json'))['after_sha256']
    with (_repo_path(ROOT, 'reports/README-before-130-provenance.md')).open('xb') as f:f.write(p.read_bytes())
    with p.open('a') as f:f.write('\n\n130 출처 정정: 두 번째 논문의 단독 3점 부정 결과는 부모 작업의 공개 논문 연구에서 전달받은 요약입니다. 이 작업에서는 해당 원문을 직접 검증하지 못했습니다. 위의 “사용자 제공 근거” 표기는 이 정정으로 대체합니다. 원본 기록과 수치 결과를 보존하며 130은 종료,014를 유지합니다. 정정본: `reports/binary-threepoint-130-summary-corrected.json`.\n')
    write(_repo_path(ROOT, 'reports/readme-update-130-provenance.json'),{'before_sha256':before,'after_sha256':e.sha(p),'correction_sha256':e.sha(cp)})
    e.emit({'130_provenance_corrected':True,'original_summary_preserved':e.sha(src)==c['original_sha256'][str(src.relative_to(ROOT))]})

def embedding(model,x):
    h=model.features(x)
    stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
    return model.head[1](model.head[0](stats))

def rnc(z,y):
    # Stable all-anchor/all-positive formulation of paper Eq.3; ties use >=, self excluded.
    n=len(y);off=~torch.eye(n,dtype=torch.bool,device=z.device)
    logits=-torch.linalg.vector_norm(z[:,None,:]-z[None,:,:],dim=2)/2.
    logits=logits[off].reshape(n,n-1)
    dist=(y[:,None]-y[None,:]).abs()[off].reshape(n,n-1)
    eligible=dist[:,None,:]>=dist[:,:,None]
    denom=torch.logsumexp(logits[:,None,:].expand(-1,n-1,-1).masked_fill(~eligible,-torch.inf),dim=2)
    return (denom-logits).mean()

def fixtures():
    torch.manual_seed(13);z=torch.randn(8,5,dtype=torch.float64,requires_grad=True);y=torch.tensor([0.,0.,1.,2.,2.,3.,5.,5.],dtype=torch.float64)
    a=rnc(z,y);terms=[]
    for i in range(8):
        for j in range(8):
            if i==j:continue
            js=[k for k in range(8) if k!=i and abs(float(y[k]-y[i]))>=abs(float(y[j]-y[i]))]
            scores=torch.stack([-torch.linalg.vector_norm(z[i]-z[k])/2 for k in js]);terms.append(torch.logsumexp(scores,0)+torch.linalg.vector_norm(z[i]-z[j])/2)
    b=torch.stack(terms).mean();ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z)[0]
    loss_error=float(abs(a-b).detach());grad_error=float(abs(ga-gb).max());assert loss_error<1e-12 and grad_error<1e-12
    identical=torch.zeros((8,5),requires_grad=True);v=rnc(identical,torch.zeros(8));v.backward();assert torch.isfinite(identical.grad).all() and abs(float(v.detach())-np.log(7))<1e-6
    p=torch.tensor([5,2,0,7,1,6,4,3]);assert abs(float(rnc(z[p],y[p]).detach()-a.detach()))<1e-12
    # Positive affine target transform preserves order; no fitted global label bins.
    assert abs(float(rnc(z,3*y+10).detach()-a.detach()))<1e-12
    zz=z.detach().float().to('mps').requires_grad_();m=rnc(zz,y.float().to('mps'));m.backward();me=abs(float(m.detach().cpu())-float(a.detach()));ge=float(abs(zz.grad.cpu()-ga.float()).max());assert me<1e-5 and ge<1e-5
    return {'passed':True,'independent_double_loop_loss_error':loss_error,'independent_gradient_error':grad_error,'MPS_loss_error':me,'MPS_gradient_error':ge,'ties_selfmask_permutation_collapsed_finite_passed':True}

def prepare():
    previous.check();assert torch.backends.mps.is_available();OUT.mkdir(exist_ok=False)
    names=['rnc_study_131.py','cnn_controls.py','ranking_objective_120.py','cnn_orientation_129.py','reports/binary-threepoint-130-summary-corrected.json','reports/binary-threepoint-130-provenance-correction.json','reports/ranking-objective-120-summary.json','artifacts/nested-calibration-074/development_features.npz','artifacts/nested-calibration-074/extraction.json','reports/candidate014-summary.json','reports/candidate014-verification.json','artifacts/submission-candidate-014.csv']
    plan={'study':'131 small paired CNN Rank-N-Contrast screen','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'novelty':'120 penalizes scalar ordering of frozen DINO head;043/083/084 supervised adaptation uses MSE/SmoothL1. No project supervised latent distance RNC found. Existing DINO pretraining is not this target-aware objective.','research':{'paper':'https://arxiv.org/html/2210.01189v2','repository':'https://github.com/kaiwenzha/Rank-N-Contrast','loss_read':'https://raw.githubusercontent.com/kaiwenzha/Rank-N-Contrast/main/loss.py','read_only':True,'external_code_downloaded_or_executed':False,'formula':'Eq3 all i!=j, logsumexp over k!=i with |yi-yk|>=|yi-yj|, minus similarity(i,j); similarity=-Euclidean(z_i,z_j)/2. Ties included, no embedding normalization.','paper_difference':'Small TinyCNN128 and AdamW,64dim pre-dropout latent,400 development images,40epochs versus paper larger ResNets/400epochs/SGD/batch256. Fixed Ridge10 regressor instead of optimized iterative regressor. This is a constrained transfer screen, not reproduction.'},'architecture':'Identical TinyCNN91905parameters; latent64afterSiLU beforeDropout. RNC optimizes encoder+hidden64; scalar65parameters dormant. MSE jointly trains scalarhead. Both execute dropout/head forward to keep RNG consumption paired.','encoder_budget':'40epochs,320train,16distinctimages x2 independently sampled batchD4views=32views/update;20updates/epoch=800;25600imageviews each. Same initial state,order,D4,dropout seed; AdamW .002->.0001 epochcosine wd.01 clip5. Fixed final state; no best epoch/weight averaging. Both same update/image budget, not equal FLOPs; RNC extra pair arithmetic measured.','downstream':'Freeze encoder; identity embeddings. StandardScaler fit320train only, Ridge(alpha10,fit_interceptTrue) on standardized trainingtarget. Identical exact solve perarm; no hyperparameter search/earlystop. No extra CNN backprop.','scope':'All label distances,normalization,encoder/Ridge fits outer320only. Existing group-separated original5fold. No holdout/test decode,inference,submission;014 unchanged.','screen':'Original5fold. Advance unchanged recipe to both known resplits only if RNC standalone RMSE<mse and raw4way RNC<both matchedMSE and legacyCNN raw4way. No rescue ratios/seeds/temperatures/epochs. Any positive screen still requires new honest full014 integration in a separate task.','raw4way':'.3KRR+.3newCNN+.2Smean3+.2Bmean3; cached outer honest predictions. NOT full014; no stale affine/residual use.','resources':'ExistingCPU2/MPS,firstpair reused benchmark; projected10fits<600sec,perfit<180sec,RSS<3GB,MPSdriver<3GB. No installs/payment/externaldata.','validation':'Independent equation/gradient fixtures including ties and collapse; everymodel MPS/CPU reload,independent SVD Ridge/scaler,pairedbudget hashes and group exclusion; fullfold0 botharms retrain after poisoning80outerlabels.','uncertainty':'Same400adaptivehistory,onepairedseed per fold,bootstrapconditionalfixedpredictions only;smallbudgetnegative cannotreject RNCgenerally.','hashes':{f:e.sha(_repo_path(ROOT, f)) for f in names},'cost':0}
    write(_repo_path(OUT, 'plan.json'),plan);v=fixtures();write(_repo_path(OUT, 'fixtures.json'),v);e.emit({'prepared':True,'fixtures':v,'training_started':False})

def check():
    p=e.read(_repo_path(OUT, 'plan.json'))
    for f,h in p['hashes'].items():assert e.sha(_repo_path(ROOT, f))==h,f
    return p

def encode(m,images,idx,norm,device):
    m.eval();chunks=[]
    with torch.inference_mode():
        for start in range(0,len(idx),32):
            x=(images[idx[start:start+32]]-norm['x_mean'])/max(norm['x_std'],1e-6);chunks.append(embedding(m,torch.tensor(x,device=device)).cpu().numpy())
    return np.concatenate(chunks)

def fit(d,k,arm,folder):
    folder.mkdir(exist_ok=False);start=time.perf_counter();T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);assert len(T)==320 and len(V)==80 and not set(d['g'][T])&set(d['g'][V])
    seed=SEED+k;torch.manual_seed(seed);torch.mps.manual_seed(seed);rng=np.random.default_rng(seed);m=engine.TinyCNN().to('mps');init=cal.h.state_hash(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.0001)
    norm={'x_mean':float(d['images'][T].mean()),'x_std':float(d['images'][T].std()),'y_mean':float(d['y'][T].mean()),'y_std':float(d['y'][T].std())};X=(d['images'][T]-norm['x_mean'])/max(norm['x_std'],1e-6);y=(d['y'][T]-norm['y_mean'])/norm['y_std'];orders=[];augs=[];lrs=[];losses=[];samplehash={};step=0;firstgrad=None
    e.emit({'training_started':str(folder.relative_to(ROOT)),'arm':arm,'steps':STEPS})
    for ep in range(40):
        m.train();order=rng.permutation(320)
        for offset in range(0,320,16):
            ii=order[offset:offset+16];views=[];aug=[]
            for j in range(2):
                rot=int(rng.integers(4));flip=bool(rng.random()<.5);v=np.rot90(X[ii],rot,axes=(2,3));v=v[:,:,:,::-1] if flip else v;views.append(v);aug.extend([rot,int(flip)])
            xx=np.ascontiguousarray(np.concatenate(views));yy=np.tile(y[ii],2);orders.append(ii);augs.append(aug);lrs.append(opt.param_groups[0]['lr'])
            if step in (0,99,399,799):samplehash[str(step)]=ah(xx)
            opt.zero_grad(set_to_none=True);z=embedding(m,torch.from_numpy(xx).to('mps'));pred=m.head[3](m.head[2](z)).squeeze(1);target=torch.tensor(yy,dtype=torch.float32,device='mps');loss=rnc(z,target) if arm=='rnc' else torch.nn.functional.mse_loss(pred,target)
            assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.);assert torch.isfinite(gn)
            if firstgrad is None:firstgrad=float(m.features[0].weight.grad.norm().cpu());assert firstgrad>0
            opt.step();losses.append(float(loss.detach().cpu()));step+=1
        sch.step()
        if (ep+1)%20==0:e.emit({'arm':arm,'fold':k,'epoch':ep+1,'seconds':time.perf_counter()-start});assert time.perf_counter()-start<180
    state={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};Z=encode(m,d['images'],np.arange(400),norm,'mps');sc=StandardScaler().fit(Z[T].astype(float));reg=Ridge(alpha=10).fit(sc.transform(Z[T]),y);pp=reg.predict(sc.transform(Z))*norm['y_std']+norm['y_mean'];assert np.isfinite(pp).all()
    payload={'state_dict':state,'normalization':norm,'mean':torch.tensor(sc.mean_),'scale':torch.tensor(sc.scale_),'coef':torch.tensor(reg.coef_),'intercept':float(reg.intercept_),'train':T.tolist(),'validation':V.tolist(),'ids':d['ids'],'arm':arm,'seed':seed};torch.save(payload,_repo_io(folder/'model.pt'))
    with (folder/'arrays.npz').open('xb') as f:np.savez_compressed(_repo_io(f),Z=Z,predictions=pp,order=np.stack(orders),augmentation=augs,lr=lrs,loss=losses)
    dist=np.linalg.norm(Z[V,None,:]-Z[None,V,:],axis=2);yd=abs(d['y'][V,None]-d['y'][None,V]);tri=np.triu_indices(len(V),1);rho=float(spearmanr(dist[tri],yd[tri]).statistic)
    rec={'arm':arm,'fold':k,'updates':step,'image_views':step*32,'initial_sha256':init,'state_sha256':cal.h.state_hash(state),'order_sha256':ah(np.stack(orders)),'augmentation_sha256':ah(np.array(augs)),'LR_sha256':ah(np.array(lrs)),'input_sha256':samplehash,'first_encoder_gradient_norm':firstgrad,'train_RMSE':e.rmse(d['y'][T],pp[T]),'validation_RMSE':e.rmse(d['y'][V],pp[V]),'validation_pair_distance_spearman_diagnostic':rho,'model_sha256':e.sha(folder/'model.pt'),'arrays_sha256':e.sha(folder/'arrays.npz'),'seconds':time.perf_counter()-start,'memory':engine.memory('mps')};write(folder/'report.json',rec);e.emit({k:rec[k] for k in ['arm','fold','train_RMSE','validation_RMSE','seconds']});del m,opt,Z;gc.collect();torch.mps.empty_cache();return pp[V],rec

def run(split):
    check();assert e.read(_repo_path(OUT, 'fixtures.json'))['passed']
    if split!='original':assert e.read(_repo_path(OUT, 'screen-decision.json'))['advance']
    d=cal.inputs(split);base=previous.q.load(split);assert base['ids']==d['ids'];folder=_repo_path(OUT, split);folder.mkdir(exist_ok=False);pred={a:np.full(400,np.nan) for a in ARMS};records=[];start=time.perf_counter()
    for k in range(5):
        pair=[]
        for arm in ARMS:
            p,r=fit(d,k,arm,folder/f'fold-{k}-{arm}');pred[arm][d['fid']==k]=p;pair.append(r);records.append(r)
        for key in ['initial_sha256','order_sha256','augmentation_sha256','LR_sha256','input_sha256','updates','image_views']:assert pair[0][key]==pair[1][key],key
        if k==0:
            secs=sum(r['seconds'] for r in pair);mem=pair[-1]['memory'];b={'first_pair_seconds_reused':secs,'projected10fits_seconds':5*secs,'memory':mem};b['passed']=5*secs<600 and mem['process_peak_rss_bytes']<3e9 and mem['driver_allocated_memory']<3e9;write(folder/'benchmark.json',b);assert b['passed']
    raw=.3*base['P'][:,1]+.2*base['P'][:,3]+.2*base['P'][:,4]
    scores={**{a:e.rmse(d['y'],p) for a,p in pred.items()},'legacyCNN':e.rmse(d['y'],base['P'][:,2]),**{'raw_'+a:e.rmse(d['y'],raw+.3*p) for a,p in pred.items()},'raw_legacy':e.rmse(d['y'],raw+.3*base['P'][:,2])}
    with (folder/'predictions.npz').open('xb') as f:np.savez_compressed(_repo_io(f),ids=np.array(d['ids']),y=d['y'],fold=d['fid'],**pred,raw_mse=raw+.3*pred['mse'],raw_rnc=raw+.3*pred['rnc'],raw_legacy=raw+.3*base['P'][:,2])
    report={'split':split,'RMSE':scores,'records':records,'seconds':time.perf_counter()-start,'predictions_sha256':e.sha(folder/'predictions.npz')};write(folder/'report.json',report);check();e.emit({'split':split,'RMSE':scores,'seconds':report['seconds']})

if __name__=='__main__':
    torch.set_num_threads(2);p=argparse.ArgumentParser();p.add_argument('mode',choices=['correction','prepare',*e.SPLITS]);x=p.parse_args();correction() if x.mode=='correction' else prepare() if x.mode=='prepare' else run(x.mode)
