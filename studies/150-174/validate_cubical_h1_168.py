"""Engineering validation only; reads no real images/targets."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
import json,time,hashlib
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
import cubical_h1_168 as f
import persistence_features_107 as old
import betti_features_055 as betti
import cluster_pair_features_159 as pair
R=_RepoPath(__file__).resolve().parents[1];O=_repo_path(R, 'artifacts/cubical-h1-168')
def write(p,z):
    with p.open('x') as fp:json.dump(z,fp,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixtures():
    ring=np.full((9,9),200,np.uint8);ring[2:7,2:7]=40;ring[3:6,3:6]=150
    opened=ring.copy();opened[2:5,4]=200
    delayed=ring.copy();delayed[2,4]=90
    two=np.full((9,13),200,np.uint8)
    for c,d in [(1,130),(7,180)]:two[2:7,c:c+5]=40;two[3:6,c+1:c+4]=d
    touching=np.full((7,7),200,np.uint8);touching[1,1]=40;touching[2,2]=40
    diagonal=np.full((5,5),200,np.uint8);diagonal[1,2]=40;diagonal[2,1]=40;diagonal[2,3]=40;diagonal[3,2]=40
    # Two maxima merge at100; younger180 dies in reverse, elder220 joins exterior at40.
    merging=np.full((7,9),40,np.uint8);merging[2:5,2:7]=100;merging[3,3]=220;merging[3,5]=180
    return {'closed_ring':(ring,[(40,150)]),'open_ring':(opened,[]),'delayed_gate':(delayed,[(90,150)]),'multiple_rings':(two,[(40,130),(40,180)]),'corner_touch':(touching,[]),'corner_closed_hole':(diagonal,[(40,200)]),'merge_pairing':(merging,[(40,220),(100,180)]),'uniform':(np.full((5,6),70,np.uint8),[])}
def main():
    start=time.perf_counter();O.mkdir(exist_ok=False)
    plan={'study':168,'stage':'implementation_then_label_blind_pilot_then_conditional_model','definition':'Exact positive-length ordinary H1 over F2 for 2D closed-square top-cell sublevel cubical complex. Faces enter at minimum incident pixel intensity. Nonperiodic finite rectangle; holes touching exterior omitted. Not vertex lower-star/rips graph PH, not threshold Betti approximation, not TopTemp exact reproduction.','algorithm':'Max-tree of 4-neighbor complement with256 exterior ring; positive dual finite H0 bars mapped(saddle,peak) to primalH1. Padding only represents immortal exterior in dual computation. Primal complex unpadded.','ties':'Full equal-intensity subcomplex at t; zero intervals omitted. Positive multiset must be invariant to equal-value cell ordering; do not perturb intensities.','implementation_gate':'All handcomputed intervals, random GF2 full-cubical boundary reduction, dual UF, Euler snapshots, D4, polarity swapping must agree; mismatch stops before realimages.','pilot_gate':'16 development IDs chosen smallest SHA256(ID), independent of targets/errors. H1vsH0 PI relative difference >.001 on>=12/16; joint existingfeaturecollision fixture; median relativePI sensitivity to bounded1gray noise<.20 and bounded4noise<.50; projected400pairedfeatures<300s; RSS<2GB. All engineering cutoffs fixed before pilot; no target claim.','feature_if_pass':'32 dims,4x4 birth/lifetime PI perlower/upperpolarity,Gaussian sigma.25 innormalizedgray,linear lifetime weight/65536. Same32dim old107H0 control encodedidentically. No PCA, feature/parameter sweep, denoising, resize.','model_if_pass':'Append32block to018585 residualfeatures; unchangedhonestnested3anchor/3innerfold/5candidate512bootstrap. Original5fold first; H1 mustbeat both018 andmatchedH0 plus>=3/5foldgainsvs018 and passaudits before twoexistingresplits. Same400adaptiveexploration only. No full500/test/holdout/newCSV authorized here.','stop':'If any correctness gatefails stopbefore realimage. Ifpilotfails nofit. Ifmodeloriginalfails noresplits/variants/controlpromotion.','sources':['https://gudhi.inria.fr/python/latest/cubical_complex_user.html'],'protected':json.loads((_repo_path(R, 'artifacts/heterogeneity-calibration-167/plan.json')).read_text())['protected'],'new_installs_downloads':False}
    write(_repo_path(O, 'plan.json'),plan);rec=[];calls=0;rng=np.random.default_rng(20261006)
    for name,(x,expected) in fixtures().items():
        pd=f.diagram(x);np.testing.assert_array_equal(pd,np.array(expected,dtype=np.int16).reshape(-1,2))
        for k in range(4):
            for v in (np.rot90(x,k),np.rot90(x,k)[:,::-1]):np.testing.assert_array_equal(f.diagram(v),pd)
        for seed in (0,1,19):
            bars,essential=f.boundary_matrix(x,seed);calls+=1;np.testing.assert_array_equal(bars[1],pd);assert essential==[0]
            np.testing.assert_array_equal(f.dual_union_find(x,seed),pd)
        for polarity in (x,255-x):
            p=f.diagram(polarity);bars,_=f.boundary_matrix(polarity,2);calls+=1;np.testing.assert_array_equal(bars[1],p)
        np.testing.assert_array_equal(f.extract(255-x),f.extract(x).reshape(2,16)[::-1].ravel())
        rec.append({'name':name,'image':x.tolist(),'expected_H1':expected,'actual_H1':pd.tolist()})
    for trial in range(300):
        shape=tuple(rng.integers(2,9,size=2));x=(rng.integers(0,4,size=shape)*70).astype(np.uint8) if trial<200 else rng.integers(0,256,size=shape,dtype=np.uint8)
        pd=f.diagram(x)
        for seed in (0,1):
            bars,essential=f.boundary_matrix(x,seed);calls+=1;np.testing.assert_array_equal(bars[1],pd);assert essential==[0]
        np.testing.assert_array_equal(f.dual_union_find(x,trial),pd)
        for level in np.unique(x):
            m=x<=level;b0=ndi.label(m,structure=np.ones((3,3)))[1];b1=b0-euler_number(m,connectivity=2);assert b1==np.sum((pd[:,0]<=level)&(level<pd[:,1]))
        for k in range(4):
            for v in (np.rot90(x,k),np.rot90(x,k)[:,::-1]):np.testing.assert_array_equal(f.diagram(v),pd)
    # Same geometry/masks, swap two bright peaks between interior and boundary;
    # third240 peak preserves finite upper-H0 multiset including both swapped peaks.
    A=np.full((256,256),100,np.uint8);A[50:53,50:53]=180;A[0:3,100:103]=220;A[150:153,150:153]=240
    B=A.copy();B[50:53,50:53]=220;B[0:3,100:103]=180
    for u,v in ((A,B),(255-A,255-B)):np.testing.assert_array_equal(old.diagram(u),old.diagram(v))
    ba,_=betti.extract(A);bb,_=betti.extract(B);np.testing.assert_array_equal(ba,bb)
    sa,ca,_=pair.extract(A);sb,cb,_=pair.extract(B);np.testing.assert_array_equal(sa,sb);np.testing.assert_array_equal(ca,cb)
    assert not np.array_equal(f.diagram(A),f.diagram(B));assert np.linalg.norm(f.extract(A)-f.extract(B))>0
    np.savez_compressed(_repo_path(O, 'novelty-fixture.npz'),A=A,B=B)
    report={'passed':True,'hand_fixtures':rec,'random_images':300,'independent_boundary_reductions':calls,'UF_D4_tie_Euler_polarity_passed':True,'joint_collision':{'old_both_H0_diagrams_equal':True,'all132Betti_equal':True,'all36C2_andS2_equal':True,'H1_A':f.diagram(A).tolist(),'H1_B':f.diagram(B).tolist(),'PI_L2_difference':float(np.linalg.norm(f.extract(A)-f.extract(B))),'interpretation':'Strict nonidentity with concatenation of these particular descriptors. Boundary/contrast association, not proof of mechanical relevance or independence from full018deepfeatures.'},'seconds':time.perf_counter()-start,'source_sha256':{v:sha(_repo_path(R, v)) for v in ['cubical_h1_168.py','validate_cubical_h1_168.py','persistence_features_107.py','betti_features_055.py','cluster_pair_features_159.py']},'real_images_read':0,'targets_read':0}
    write(_repo_path(O, 'implementation-validation.json'),report);print(json.dumps({k:v for k,v in report.items() if k not in ['hand_fixtures','source_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
