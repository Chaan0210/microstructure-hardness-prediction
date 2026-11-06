"""Label-blind fixed pilot evaluation; no target data is loaded."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,resource
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import formation_model_049 as m
import formation_pilot_049 as p


def pr(mask,truth,tol=0):
    mask=np.asarray(mask,bool);truth=np.asarray(truth,bool)
    near_truth=ndi.binary_dilation(truth,iterations=tol) if tol else truth
    near_mask=ndi.binary_dilation(mask,iterations=tol) if tol else mask
    return {'recall':float(np.mean(near_mask[truth])) if truth.any() else None,'precision':float(np.mean(near_truth[mask])) if mask.any() else 0.0,'truth_pixels':int(truth.sum()),'predicted_pixels':int(mask.sum())}

def fixture(k,severity,quality):
    yy,xx=np.indices((256,256));sx,sy=((32,32),(48,24),(40,32))[k]
    warped=xx+ (4*np.sin(yy/25) if k==2 else 0)
    ix=np.floor((warped+16)/sx).astype(int);iy=np.floor((yy+16)/sy).astype(int)
    lab=1+ix+20*iy;phase=(ix+iy)%2
    boundary=(lab!=np.roll(lab,1,axis=0)) | (lab!=np.roll(lab,1,axis=1));boundary[0]=False;boundary[:,0]=False
    pore=np.zeros_like(boundary)
    # One isolated disk; two boundary-attached disks. Known truth only fixtures.
    loc=np.argwhere(boundary)
    for j,(cy,cx,r) in enumerate(((70,70,3),(128,128,5),(194,182,7))):
        if j:cy,cx=loc[np.argmin(((loc-np.array([cy,cx]))**2).sum(1))]
        else:
            centers=np.argwhere(ndi.distance_transform_edt(~boundary)>=r+3)
            cy,cx=centers[np.argmin(((centers-np.array([cy,cx]))**2).sum(1))]
        pore|=(yy-cy)**2+(xx-cx)**2<=r*r
    latent=np.where(phase,0.35,0.75);latent[boundary]=0.20;latent[pore]=0.04
    blur=(0.5,1.0)[severity]
    noise=quality['noise_quantiles'][severity*2]
    contrast=quality['contrast_quantiles'][severity*2]
    signal=ndi.gaussian_filter(latent,blur)
    signal=.5+(signal-.5)*min(1.2,max(.5,contrast/.55))
    rng=np.random.default_rng(p.SEED+100+k*2+severity)
    n=ndi.gaussian_filter(rng.normal(size=signal.shape),(.0,.5)[severity]);n=n/n.std()*noise
    image=np.rint(np.clip(signal+n,0,1)*255).astype(np.uint8)
    sg=lab.copy();sg[boundary|pore]=0
    st=m.old._summarize(sg,latent)
    truth=np.array([st[2],float(phase[~(boundary|pore)].mean()),pore.mean(),st[7]])
    return image,truth,boundary & ~pore,pore,{'blur_sigma':blur,'noise_std':noise,'contrast':contrast}

def finite_median(a):
    a=[x for x in a if x is not None]
    return float(np.median(a)) if a else None

def main():
    t=time.perf_counter();torch.set_num_threads(2)
    plan=json.loads((p.OUT/'plan.json').read_text());bench=json.loads((p.OUT/'benchmark.json').read_text())
    assert bench['resource_pass'];assert not (p.OUT/'report.json').exists()
    for n,h in plan['sources'].items():assert p.sha(_repo_path(ROOT, n))==h,n
    sources={n:p.sha(_repo_path(ROOT, n)) for n in ['evaluate_formation_049.py','formation_model_049.py','formation_pilot_049.py']}
    p.save(_repo_io(p.OUT/'evaluation-source.json'),sources)
    images=[p.read_image(i) for i in plan['ids']]
    nq=[p.nuisance(x) for x in images]
    quality={name+'_quantiles':np.quantile([a[name] for a in nq],[.2,.5,.8]).tolist() for name in nq[0]}
    p.save(_repo_io(p.OUT/'training-nuisance.json'),quality)
    print(json.dumps({'stage':'start','quality':quality,'images':40,'variants':4,'label_files_opened':0}),flush=True)
    fixtures=[]
    for k in range(3):
        for severity in range(2):
            x,truth,bd,po,conditions=fixture(k,severity,quality);r=m.estimate(x);b=m.baseline(x)
            item={'layout':k,'severity':severity,'conditions':conditions,'truth':truth.tolist(),'candidate':r['features'].tolist(),'baseline':b['features'].tolist(),'boundary':pr(r['boundary'],bd,2),'pore':pr(r['pore'],po),'baseline_boundary':pr(b['boundary'],bd,2),'baseline_pore':pr(b['pore'],po),'grain_relative_error':float(abs(r['features'][0]-truth[0])/max(truth[0],1e-9)),'phase_abs_error':float(abs(r['features'][1]-truth[1])),'alignment_abs_error':float(abs(r['features'][3]-truth[3]))}
            fixtures.append(item)
            np.savez_compressed(_repo_io(p.OUT/f'fixture-{k}-{severity}.npz'),image=x,truth=truth,boundary_truth=bd,pore_truth=po,u=r['u'],d=r['d'],boundary=r['boundary'],pore=r['pore'])
    p.save(_repo_io(p.OUT/'fixtures.json'),fixtures);print(json.dumps({'stage':'fixtures_done','n':6,'seconds':time.perf_counter()-t,'boundary_recall':[v['boundary']['recall'] for v in fixtures],'pore_precision':[v['pore']['precision'] for v in fixtures]}),flush=True)
    rows=[];allbase=[];allnew=[]
    for ix,(ident,x) in enumerate(zip(plan['ids'],images)):
        entry={'ID':ident,'image_sha256':p.sha(_repo_path(ROOT, 'data/train', f'{ident}.png')),'variants':[]}
        bs=[];cs=[]
        for kind in plan['variants']:
            a=p.perturb(x,kind,ix);b=m.baseline(a);r=m.estimate(a)
            bs.append(b['features']);cs.append(r['features'])
            item={'variant':kind,'baseline':b['features'].tolist(),'candidate':r['features'].tolist(),'diagnostics':r['diagnostics']}
            if kind=='identity':
                # Image-derived anchors are only consistency proxies, never truth.
                item['boundary_anchor']=pr(r['boundary'],b['boundary'],2)
                item['pore_anchor']=pr(r['pore'],b['pore'],1)
                np.savez_compressed(_repo_io(p.OUT/f'{ident}.npz'),u=r['u'],d=r['d'],render=r['render'],boundary=r['boundary'],pore=r['pore'],start_features=r['start_features'])
            entry['variants'].append(item)
        allbase.append(bs);allnew.append(cs);rows.append(entry)
        p.save(_repo_io(p.OUT/f'{ident}.json'),entry)
        print(json.dumps({'stage':'image_complete','completed':ix+1,'total':40,'ID':ident,'seconds':time.perf_counter()-t}),flush=True)
    base=np.asarray(allbase);new=np.asarray(allnew)
    scale=np.maximum(np.subtract(*np.percentile(base[:,0],[75,25],axis=0)),plan['stability_gate']['floor_scale'])
    sb=np.median(np.sqrt(np.mean((base[:,1:]-base[:,:1])**2,axis=1)),axis=0)/scale
    sn=np.median(np.sqrt(np.mean((new[:,1:]-new[:,:1])**2,axis=1)),axis=0)/scale
    ratio=sn/np.maximum(sb,1e-12)
    stability=bool(np.all(sb>1e-8) and np.all(ratio<=.7))
    iqb=np.subtract(*np.percentile(base[:,0],[75,25],axis=0));iqn=np.subtract(*np.percentile(new[:,0],[75,25],axis=0));iqratio=iqn/np.maximum(iqb,1e-9)
    anchors_b=[r['variants'][0]['boundary_anchor']['recall'] for r in rows];anchors_p=[r['variants'][0]['pore_anchor']['recall'] for r in rows]
    start=np.array([r['variants'][0]['diagnostics']['start_feature_range'] for r in rows])/scale
    degenerates=[]
    for row in rows:
        d=row['variants'][0]['diagnostics'];deg=any(v<.02 for v in d['latent_std']) or any(v>1.95*d['noise_initial_normalized'] for v in d['noise_normalized']) or any(v>1.72 for v in d['blur_sigma']) or any(v<.002 or v>.35 for v in d['boundary_fraction']);degenerates.append(deg)
    g=plan['preservation_gate'];fpass=[]
    for f in fixtures:
        ok=f['boundary']['recall']>=g['fixture_boundary_recall_min'] and f['boundary']['precision']>=g['fixture_boundary_precision_min'] and f['pore']['recall']>=g['fixture_pore_recall_min'] and f['pore']['precision']>=g['fixture_pore_precision_min'] and f['grain_relative_error']<=g['fixture_max_relative_grain_error'] and f['phase_abs_error']<=g['fixture_max_abs_phase_error'] and f['alignment_abs_error']<=g['fixture_max_abs_alignment_error'];fpass.append(bool(ok))
    anchor_b=finite_median(anchors_b);anchor_p=finite_median(anchors_p)
    checks={'all_fixture_structures_preserved':all(fpass),'real_feature_spread_noncollapsed':bool(np.all(iqratio>=.5)),'real_boundary_anchors':sum(v is not None for v in anchors_b)>=30 and anchor_b is not None and anchor_b>=.8,'real_pore_anchors':sum(v is not None for v in anchors_p)>=10 and anchor_p is not None and anchor_p>=.75,'start_agreement':bool(np.all(np.median(start,axis=0)<=.5)),'nondegenerate_fraction':float(np.mean(degenerates))<=.10}
    for n,h in plan['protected'].items():assert p.sha(_repo_path(ROOT, n))==h,n
    for n,h in sources.items():assert p.sha(_repo_path(ROOT, n))==h,n
    report={'status':'completed','names':m.NAMES,'images':40,'variants':4,'synthetic_fixture_count':6,'label_files_opened':0,'hypothesis_promoted':bool(stability and all(checks.values())),'stability':{'baseline':sb.tolist(),'candidate':sn.tolist(),'candidate_over_baseline':ratio.tolist(),'reduction_fraction':(1-ratio).tolist(),'passed':stability,'scale':scale.tolist()},'preservation':{'checks':checks,'fixture_pass':fpass,'iqr_ratio':iqratio.tolist(),'boundary_anchor_recall_median':anchor_b,'pore_anchor_recall_median':anchor_p,'pore_anchor_images':sum(v is not None for v in anchors_p),'start_disagreement_normalized_median':np.median(start,axis=0).tolist(),'degenerate_images':int(sum(degenerates))},'seconds':time.perf_counter()-t,'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'sources':sources,'protected_unchanged':True,'best':'candidate006 unchanged','regression_or_cv_executed':False,'new_candidate':None,'limitations':['Fixed synthetic perturbations are probes, not the identified competition generator.','Input-derived real-image anchors can miss structure; fixture metrics are conditional on analytic layouts, not ground truth for real images.','Joint inverse image model remains non-identifiable and optimizer may underfit or overfit. Failure rejects this fixed approximation, not all possible inverse problems or competition ceiling.','No new target score exists because feasibility precedes regression.']}
    p.save(_repo_io(p.OUT/'report.json'),report);p.save(_repo_path(ROOT, 'reports/formation-049-summary.json'),report)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
