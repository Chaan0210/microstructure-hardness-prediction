"""Freeze binary-bank source, fixtures and20 original-train image benchmark."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,json,time,datetime,resource,argparse,hashlib
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import binary_threepoint_features_130 as f
import numpy as np
from PIL import Image
OUT=_repo_path(ROOT, 'artifacts/binary-threepoint-130/serialization-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)
    with p.open('x') as h:h.write(text+'\n')
def emit(x):print(json.dumps(x,ensure_ascii=False),flush=True)
def check():
    p=read(_repo_path(OUT, 'feature-plan.json'))
    for n,h in {**p['sources'],**p['protected']}.items():assert sha(_repo_path(ROOT, n))==h,n
    return p

def prepare():
    OUT.mkdir(exist_ok=False);manifest=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(manifest['development_folds']);pilot=[i for i in ids if manifest['development_folds'][i]!=0][:20]
    p={'study':'130 fixed binary2/3point correlation features','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'definition':'Image-local raw grayscale quantiles .2/.5/.8, B=(image<=q), actualp=B.mean includingties; x=B-p. Native256 uint8, no blur/denoise/resize/physicalphase claim. No externaltrainingdata.','triangles':'r1/2/4/8/16/32; vertices0,(r,0),(0,r) or0,(r,0),(0,2r). Mean x0*xu*xv ontheir common valid rectangular support, no wrap/padding. Eachmotif D4eightorientations thenmean/populationstd;3*6*2*2=72.','pairs':'Allthree triangle-edge displacements, canonical absolute sortedcoordinates underD4 and +/-; deduplicate acrossmotifs/radii before extraction:19orbits,114cols. Each oriented pair uses its full valid pair-domain (not a triangle-specific restricted domain); thus same displacement has one statistic. Eightorientationmean/std.','pair_orbits':[[int(x) for x in v] for v in f.ORBITS],'names':{'pair':list(f.PAIR_NAMES),'triple':list(f.TRIPLE_NAMES)},'floor':'For each bank independently, fit mean/std on currenttrainingrows only. Scale_j=max(std_j,RMS_j(std),1e-12); multiply1/sqrt(bankwidth). No imagevariance division, PCA or hard clipping. Allconstanttrainbank yields zeros in training; protectedabsolute floor preventsdivisionbyzero. Scale of2pointbank identicalwhen3pointappended.','existing108':'108 rawintensity-centered triples, line+equal-legcorner,r1..64,compact26+intensity+pairedmoments,KRRdirecttarget,.1blend011. Here binary quantiles,unequal-legtriangle,no collinear motif,r<=32,deduplicatededge controls,and nested014 residual extension. Sharedthirdorder principle,not whollynew family.','existing054_055':'Linealwhole-segment probabilities and signed-distanceBetti are different events; no samebinary triangle product bank found in inspected sources.','paper_basis':[{'url':'https://arxiv.org/html/2109.02255','access':'HTML accessed2026-10-04','finding':'Reduced learned3point representations plusstrongcontrast expansion approximateeffectivebulk/shearmoduli. Their comparator values fromcomplete3pointSCE,not competitionhardness.','difference':'We do not reproduce Bayesianoptimization/reconstruction/effectivemediumtheory oruncenteredprobabilities; fixed centered lowcostbank only.'},{'url':'https://doi.org/10.1016/j.actamat.2025.121132','institutional_abstract':'https://tore.tuhh.de/entities/publication/2c765e52-4364-4fbf-9004-555a1f88075b','access':'Publisher403; institutionalabstract accessed','finding':'Combined3point/Gram/texturedescriptors forMg alloystrainhardening/yieldstress,not hardness.','negative_evidence':'User-provided: standalone3point weakestforstrainhardening; not independentlyverified fromfulltext inthissession.'}],'pilot_ids':pilot,'pilot_selection':'First20lexicographicsortedoriginalfold!=0developmentIDs; no labelread/selection.','development_ids':ids,'resource':'CPU2existinguv. Original20 extractiontime projected400<600sec,peakRSS<2GB. Fixturesmustpass before benchmark. Reusepilotfeaturesin400cache. No model fit/holdout/test/paidservice/package install.','scope':'Bankcannotchangebasedonlabels. Ifengineeringfailure preservefailureandrepairwithseparateversion.','sources':{n:sha(_repo_path(ROOT, n)) for n in ['binary_threepoint_features_130.py','binary_threepoint_preflight_130.py','binary_threepoint_preflight_v2_130.py','threepoint_features_108.py','threepoint_study_108.py']},'protected':{n:sha(_repo_path(ROOT, n)) for n in ['artifacts/submission-candidate-014.csv','reports/candidate014-summary.json','reports/cnn-orientation-129-preservation.json','reports/readme-update-129.json','artifacts/run-001/split_manifest.json']},'cost':0};write(_repo_path(OUT, 'feature-plan.json'),p);emit({'feature_plan_fixed':True,'pair114':len(f.PAIR_NAMES),'triple72':len(f.TRIPLE_NAMES),'sha256':sha(_repo_path(OUT, 'feature-plan.json'))})

def fixtures():
    rng=np.random.default_rng(130);maxerr=0.;maxd4=0.;masks=[np.zeros((16,16),bool),np.ones((16,16),bool),rng.random((16,16))<.3];border=np.zeros((16,16),bool);border[:,[0,-1]]=True;border[4:9,2:6]=True;masks.append(border)
    for mask in masks:
        p,t=f.centered(mask,(1,2,4));ip,it=f.independent(mask,(1,2,4));maxerr=max(maxerr,float(abs(p-ip).max()),float(abs(t-it).max()));np.testing.assert_allclose(p,ip,atol=1e-15,rtol=0);np.testing.assert_allclose(t,it,atol=1e-15,rtol=0)
        cp,ct=f.centered(~mask,(1,2,4));np.testing.assert_allclose(cp,p,atol=1e-15,rtol=0);np.testing.assert_allclose(ct[::2],-t[::2],atol=1e-15,rtol=0);np.testing.assert_allclose(ct[1::2],t[1::2],atol=1e-15,rtol=0)
        if mask.all() or not mask.any():assert np.count_nonzero(p)==np.count_nonzero(t)==0
        for k in range(4):
            for flip in (False,True):
                a=np.rot90(mask,k);a=np.fliplr(a) if flip else a;pp,tt=f.centered(a,(1,2,4));maxd4=max(maxd4,float(abs(pp-p).max()),float(abs(tt-t).max()));np.testing.assert_allclose(pp,p,atol=1e-15,rtol=0);np.testing.assert_allclose(tt,t,atol=1e-15,rtol=0)
    x=border.astype(float)-border.mean();z=np.array([[0,0],[4,0],[0,8]]);A,B,C=f.samples(x,z);assert A.shape==(12,8);finite=float((A*B*C).mean());wrap=float(np.mean(x*np.roll(x,-4,axis=0)*np.roll(x,-8,axis=1)));assert abs(finite-wrap)>1e-6
    assert len(f.ORBITS)==len(set(f.ORBITS))==19;X=np.column_stack([np.arange(8),np.ones(8),1e-16*np.arange(8),np.arange(8)**2]);sc=f.scaling(X,['pair']*4);Z=f.transform(X,sc);assert np.isfinite(Z).all() and np.max(abs(Z))<2;S=f.scaling(np.ones((8,4)),['pair']*4);assert np.array_equal(f.transform(np.ones((8,4)),S),np.zeros((8,4)));sc2=f.scaling(np.c_[X,X**2],['pair']*4+['triple']*4);np.testing.assert_array_equal(sc2['scale'][:4],sc['scale']);np.testing.assert_array_equal(f.transform(np.c_[X,X**2],sc2)[:,:4],Z)
    return {'passed':True,'constant_zero':True,'complement_pair_even_triple_mean_odd_std_even':True,'independent_coordinate_max_error':maxerr,'D4_max_error':maxd4,'boundary_finite_value':finite,'boundary_wrapped_value':wrap,'no_wrap_fixture_distinguishes':True,'pair_orbits_unique':19,'std_floor_and_block_prefix_passed':True,'label_access':False}

def pilot():
    p=check();eng=fixtures();write(_repo_path(OUT, 'fixtures.json'),eng);rows=[];meta=[];hashes={};times=[]
    for ident in p['pilot_ids']:
        file=_repo_path(ROOT, 'data/train', f'{ident}.png')
        with Image.open(_repo_io(file)) as im:a=np.array(im)
        start=time.perf_counter();P,T,m=f.extract(a);times.append(time.perf_counter()-start);rows.append(np.r_[P,T]);meta.append(m);hashes[ident]=sha(file)
    # Reproducibility and independentreal-image check outsidetiming.
    ident=p['pilot_ids'][0]
    with Image.open(_repo_path(ROOT, 'data/train', f'{ident}.png')) as im:a=np.array(im)
    P,T,m=f.extract(a);np.testing.assert_array_equal(np.r_[P,T],rows[0]);ind=[]
    for th in np.quantile(a,f.QUANTILES):pp,tt=f.independent(a<=th);ind.append((pp,tt))
    ip=np.concatenate([z[0] for z in ind]);it=np.concatenate([z[1] for z in ind]);error=float(np.max(abs(np.r_[ip,it]-rows[0])));assert error<1e-15
    total=sum(times);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;rec={'passed':total*20<600 and rss<2e9,'n':20,'extraction_seconds':total,'projected400_seconds':total*20,'RSS_bytes':rss,'image_sha256':hashes,'metadata':meta,'first_real_repeat_exact':True,'first_real_independent_max_error':error,'no_labels_read':True,'features_source_sha256':sha(_repo_path(ROOT, 'binary_threepoint_features_130.py'))}
    with (_repo_path(OUT, 'pilot-features.npz')).open('xb') as h:np.savez_compressed(_repo_io(h),ids=np.array(p['pilot_ids']),X=np.stack(rows))
    rec['cache_sha256']=sha(_repo_path(OUT, 'pilot-features.npz'));write(_repo_path(OUT, 'benchmark.json'),rec);assert rec['passed'];emit({k:v for k,v in rec.items() if k not in ['metadata','image_sha256']})

def extract():
    p=check();b=read(_repo_path(OUT, 'benchmark.json'));assert b['passed'];assert sha(_repo_path(OUT, 'pilot-features.npz'))==b['cache_sha256'];start=time.perf_counter()
    with np.load(_repo_path(OUT, 'pilot-features.npz')) as z:prior=dict(zip(z['ids'].tolist(),z['X'].copy()))
    rows=[];hashes=dict(b['image_sha256']);meta=dict(zip(p['pilot_ids'],b['metadata']))
    for i,ident in enumerate(p['development_ids']):
        if ident in prior:row=prior[ident]
        else:
            file=_repo_path(ROOT, 'data/train', f'{ident}.png')
            with Image.open(_repo_io(file)) as im:a=np.array(im)
            P,T,m=f.extract(a);row=np.r_[P,T];meta[ident]=m;hashes[ident]=sha(file)
        rows.append(row)
        if (i+1)%50==0:emit({'features_complete':i+1,'seconds':time.perf_counter()-start})
    X=np.stack(rows);assert X.shape==(400,186) and np.isfinite(X).all()
    with (_repo_path(OUT, 'development-features.npz')).open('xb') as h:np.savez_compressed(_repo_io(h),ids=np.array(p['development_ids']),X=X)
    rec={'shape':list(X.shape),'pilot20_reused':True,'new_images_extracted':380,'image_sha256':hashes,'metadata':meta,'cache_sha256':sha(_repo_path(OUT, 'development-features.npz')),'seconds':time.perf_counter()-start,'constant_columns':int(np.sum(np.std(X,axis=0)<1e-15)),'features_modified_from_labels':False};write(_repo_path(OUT, 'extraction.json'),rec);check();emit({k:v for k,v in rec.items() if k not in ['metadata','image_sha256']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','pilot','extract']);globals()[p.parse_args().mode]()
