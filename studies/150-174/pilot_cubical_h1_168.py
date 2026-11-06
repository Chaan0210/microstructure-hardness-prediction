"""Outcome-blind pre-fit pilot:16 hashed development IDs, no label/prediction reads."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,hashlib,time,resource
from collections import Counter
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.measure import euler_number
import cubical_h1_168 as f
import persistence_features_107 as old
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/cubical-h1-168')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,z):
    with p.open('x') as fp:json.dump(z,fp,indent=2,allow_nan=False)
def h0(x):return np.concatenate([f.encode(old.diagram(v),v.size) for v in (x,255-x)])
def main():
    plan=json.loads((_repo_path(O, 'plan.json')).read_text());audit=json.loads((_repo_path(O, 'implementation-validation.json')).read_text());assert audit['passed']
    for file,h in audit['source_sha256'].items():assert sha(_repo_path(R, file))==h
    ids=sorted(json.loads((_repo_path(R, 'artifacts/run-001/split_manifest.json')).read_text())['development_folds']);assert len(ids)==400
    chosen=sorted(ids,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:16]
    lock=json.loads((_repo_path(R, 'artifacts/nested-calibration-074/extraction.json')).read_text())['image_sha256'];rows=[];H=[];C=[];N1=[];N4=[];times=[];start=time.perf_counter()
    for idx,ident in enumerate(chosen):
        p=_repo_path(R, 'data/train', f'{ident}.png');assert sha(p)==lock[ident]
        with Image.open(_repo_io(p)) as im:assert im.mode=='L' and im.size==(256,256);x=np.array(im)
        t=time.perf_counter();q=f.extract(x);c=h0(x);times.append(time.perf_counter()-t);H.append(q);C.append(c)
        rng=np.random.default_rng(int(hashlib.sha256(ident.encode()).hexdigest()[:8],16));rr={'ID':ident,'sha256':lock[ident],'paired_seconds':times[-1],'relative_H1_H0_PI':float(np.linalg.norm(q-c)/max(np.linalg.norm(q),1e-30)),'noise':{}};diags=[];dualold=[]
        for v in (x,255-x):
            pd=f.diagram(v);np.testing.assert_array_equal(pd,f.dual_union_find(v,168));diags.append(pd)
            d=old.diagram(255-v);mapped=np.column_stack([255-d[:,1],255-d[:,0]]);dualold.append(mapped)
        rr['H1_count']=[len(p) for p in diags];rr['H1_lifetime_sum']=[int(np.diff(p,axis=1).sum()) for p in diags]
        rr['dual_unpadded_H0_unmatched_bars']=[sum((Counter(map(tuple,p))-Counter(map(tuple,c))).values()) for p,c in zip(diags,dualold)]
        dualpi=np.concatenate([f.encode(d,x.size) for d in dualold]);rr['relative_H1_dual_unpadded_H0_PI']=float(np.linalg.norm(q-dualpi)/max(np.linalg.norm(q),1e-30))
        for amp,dest in ((1,N1),(4,N4)):
            z=np.clip(x.astype(int)+rng.integers(-amp,amp+1,size=x.shape),0,255).astype(np.uint8);qn=f.extract(z);dest.append(qn);rr['noise'][str(amp)]={'linf':int(np.abs(x.astype(int)-z.astype(int)).max()),'relative_PI_change':float(np.linalg.norm(qn-q)/max(np.linalg.norm(q),1e-30)),'count':[len(f.diagram(z)),len(f.diagram(255-z))]}
        if idx<2:
            for k in range(4):
                for v in (np.rot90(x,k),np.rot90(x,k)[:,::-1]):np.testing.assert_array_equal(f.extract(v),q)
            # Every integer filtration threshold, independent digitalEuler count.
            for v,pd in zip((x,255-x),diags):
                for level in range(256):
                    m=v<=level;b0=ndi.label(m,np.ones((3,3)))[1];b1=b0-euler_number(m,connectivity=2)
                    assert b1==int(np.sum((pd[:,0]<=level)&(level<pd[:,1])))
        if idx<4:
            # Actual8x8 crop, directly reduce all cubical cells, not max-tree.
            patch=x[120:128,120:128];bars,_=f.boundary_matrix(patch,168);np.testing.assert_array_equal(bars[1],f.diagram(patch))
        rows.append(rr);print('PILOT_IMAGE',idx+1,ident,'seconds',round(times[-1],4),flush=True)
    H=np.array(H);C=np.array(C);N1=np.array(N1);N4=np.array(N4)
    frac=[r['relative_H1_H0_PI'] for r in rows];sens={str(k):float(np.median([r['noise'][str(k)]['relative_PI_change'] for r in rows])) for k in (1,4)}
    checks={'joint_descriptor_collision':audit['joint_collision']['old_both_H0_diagrams_equal'] and audit['joint_collision']['all132Betti_equal'] and audit['joint_collision']['all36C2_andS2_equal'],'distinct_PI_atleast12':sum(v>.001 for v in frac)>=12,'bounded1_noise':sens['1']<.20,'bounded4_noise':sens['4']<.50,'time':400*float(np.mean(times))<300,'memory':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2e9}
    variation=float(np.sqrt(np.mean((H-H.mean(0))**2)));changes={str(k):float(np.sqrt(np.mean((arr-H)**2))/max(variation,1e-30)) for k,arr in [(1,N1),(4,N4)]}
    out={'passed':all(checks.values()),'checks':checks,'IDs':chosen,'n_images':16,'targets_or_errors_read':False,'selection':'smallestSHA256(ID)amonglocked400','projected400_paired_seconds':400*float(np.mean(times)),'median_paired_seconds':float(np.median(times)),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'median_noise_relative_PI_change':sens,'noise_RMS_to_between_image_feature_RMS':changes,'median_relative_H1_H0_PI':float(np.median(frac)),'median_relative_H1_dual_unpadded_H0_PI':float(np.median([r['relative_H1_dual_unpadded_H0_PI'] for r in rows])),'boundary_distinction_every_image':all(sum(r['dual_unpadded_H0_unmatched_bars'])>0 for r in rows),'real_validation':{'UF_images_bothpolarities':16,'D4_images':2,'all256Euler_bothpolarities_images':2,'full_GF2_actual8x8crops':4},'seconds':time.perf_counter()-start,'rows':rows,'limits':['H1 overlaps strongly with opposite-polarityH0 byplanarduality; extra information concerns exterior/boundaryattachment.','Boundednoise diagnostic isnot allrealnoise/blur robustness or materialtopology validation.','Engineering difference isnot residualpredictivevalue; no target read.','PI is lossy, fullH1 diagram exactforstatedcellcomplex only; intensity inversion swaps banks,doesnot implyequalH1.'],'source_sha256':{'pilot_cubical_h1_168.py':sha(_repo_path(R, 'pilot_cubical_h1_168.py'))},'plan_sha256':sha(_repo_path(O, 'plan.json'))}
    np.savez_compressed(_repo_path(O, 'pilot-features.npz'),ids=np.array(chosen),h1=H,h0=C,noise1=N1,noise4=N4)
    write(_repo_path(O, 'pilot.json'),out);print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()
