from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
r=_RepoPath(__file__).resolve().parents[1]
s=(_repo_path(r, 'dinob_prenorm_069.py')).read_text().replace('dinob_prenorm_069','dinos_coarse_070').replace('dinob-prenorm-069','dinos-coarse-070').replace('development_Bprenorm','development_Scoarse').replace('Bprenorm','Scoarse')
s=s.replace("X=z['X'].astype(float)","X=np.concatenate([S,z['X'].astype(float)],axis=1)").replace('X.shape==(400,6912)','X.shape==(400,4224)').replace("'input_dim':6912","'input_dim':4224")
s=s.replace('new=.75*r003+.125*Smean+.125*mean','new=.75*r003+.125*mean+.125*Bmean').replace('np.corrcoef(y-mean,y-Bmean)','np.corrcoef(y-mean,y-Smean)').replace('standalone_Scoarse_B504','standalone_Scoarse_S504')
a=s.index('def feature(');b=s.index('def inputs(')
s=s[:a]+'''def coarse(p):
 p=np.asarray(p,dtype=np.float64);assert p.shape==(36,36,384);var=p.var((0,1));parts=[]
 for grid in (2,4):
  size=36//grid;means=p.reshape(grid,size,grid,size,384).mean((1,3));parts.append(means.var((0,1))/(var+1e-4))
 return np.concatenate(parts)

def token_inputs():
 from dinob_062 import streamsha
 ex=json.loads((ROOT/'artifacts/fisher-067/extraction.json').read_text());v=json.loads((ROOT/'reports/fisher-067-validation.json').read_text());assert v['passed'] and v['tokens_reextracted']==400 and v['token_max_error']==0
 path=ROOT/'artifacts/fisher-067/development_tokens.npy';assert streamsha(path)==ex['tokens_sha256'];ids,_=b.ds.base.scope();assert ex['ids']==ids
 return ids,np.load(path,mmap_mode='r'),ex

def prepare():
 prior.check();b.check();pub=json.loads((ROOT/'reports/candidate008-public-submission.json').read_text());old=prior.check();assert not json.loads((ROOT/'artifacts/dinob-prenorm-069/selection.json').read_text())['advance'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'068absoluteerrorsweaklyassociatewithcoarseautocorrelation andtilebrightnessvariation. Globaltokenmean/std andorderlessFisher discardspatialsegregation. Fixedcoarseregionalvariancefraction mayaddinformationaboutlongrangepatchheterogeneity.',
 'limits_of_evidence':'Correlations~0.13 aredescriptive,multipleinspected,notcausal. Negative069meansunnormalizedamplitudesdidnothelpthisrecipe,notallamplitudeinformationuseless.',
 'single_primary':'Append2x384 descriptors toexistingS3456: variance ofregionalmeans on2x2 and4x4 nonoverlappinggrids,dividedbyglobalpatchvariance+1e-4. FinalnormalizedS504tokens;4224dim input,same3seedMLP. Formula .75r003+.125Scoarsemean+.125Bmean.',
 'duplicate_check':'042lags1/4 pairwisesquaredifferences testlocalcoherence(7/28nativepx); this integrates2Dregions~128/64nativepx. 056covariance and067Fisher orderless. 066changesresolution ratherthan retainingregionalspatialvariation. No learnedcoordinates orabsolutequadrantidentity.',
 'risk':'Adds768inputs, changesinitializationandparametercount; representationexperimentnotcapacitymatched. D4 invariance ofcachedtokenmap descriptors doesnot implyDINOinputrotationinvariance.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Descriptorhasno learnedfit;StandardScalerandtargetscale outertrainonly.',
 'screen':'OriginalprimarypooledOOF strictlyimproves008 -> bothknownresplits; otherwise reject. No weight/grid/epsilonsearch.',
 'confirmation':'All3pooledOOF improve008+strictreload/refitvalidation. Foldcount/bootstrapdiagnosticonly.',
 'resource':'Reuseverified796MBmemmaptokens;CPU2threads,15fitsprojected<300s,RSS<2GB,noinstall/download/cost.',
 'sources':{n:fv.sha(ROOT/n) for n in ['dinos_coarse_070.py','validate_dinos_coarse_070.py','dinob_mlp_head_063.py','dinob_mlp_063.py','dinob_complement_064.py']},
 'inputs':{**old['inputs'],**{n:fv.sha(ROOT/n) for n in ['reports/dinob-complement-064-validation.json','artifacts/error-diagnostics-068/report.json','artifacts/fisher-067/extraction.json','reports/fisher-067-validation.json','artifacts/dinob-prenorm-069/selection.json']}},
 'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(ROOT/'reports/candidate008-public-submission.json')},
 'scope':'Onlydevelopment400;holdout/testnotreadforsel;protectedfinalcacheshashedonly.','bias':fv.CAVEAT}
 fv.write(OUT/'plan.json',p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis']}),flush=True)

def extract():
 check();ids,tokens,source=token_inputs();start=time.perf_counter();X=np.stack([coarse(t.reshape(36,36,384)) for t in tokens]);assert X.shape==(400,768) and np.isfinite(X).all() and X.min()>=0 and X.max()<=1+1e-8
 rng=np.random.default_rng(20270108);p=rng.normal(size=(36,36,384));v=coarse(p)
 for k in range(4):
  np.testing.assert_allclose(v,coarse(np.rot90(p,k)),atol=1e-14,rtol=1e-12);np.testing.assert_allclose(v,coarse(np.rot90(p,k)[:,::-1]),atol=1e-14,rtol=1e-12)
 np.testing.assert_array_equal(coarse(np.zeros_like(p)),np.zeros(768));step=np.zeros_like(p);step[:18]=1;checker=np.broadcast_to((np.indices((36,36)).sum(0)%2)[:,:,None],p.shape);assert np.max(abs(coarse(step)-coarse(checker)))>.9;np.testing.assert_array_equal(step.mean((0,1)),checker.mean((0,1)));np.testing.assert_array_equal(step.std((0,1)),checker.std((0,1)))
 with (OUT/'development_Scoarse.npz').open('xb') as f:np.savez_compressed(f,ids=np.array(ids),X=X)
 report={'shape':list(X.shape),'seconds':time.perf_counter()-start,'feature_sha256':fv.sha(OUT/'development_Scoarse.npz'),'token_sha256':source['tokens_sha256'],'fixtures':{'tokenmapD4':True,'zero':True,'sameglobalstatsdifferentcoarsestructure':True},'rss_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};fv.write(OUT/'extraction.json',report);print(json.dumps(report),flush=True)

'''+s[b:]
s=s.replace("['prepare','benchmark_features','extract','benchmark_head','original'","['prepare','extract','benchmark_head','original'").replace("'benchmark_features':benchmark_features,",'')
s=s.replace('Pre-LayerNorm ViT-B descriptor, unchanged008 dimensions, recipe and MLP.','Fixed coarse spatial heterogeneity with S descriptor, unchanged008 recipe.')
(_repo_path(r, 'dinos_coarse_070.py')).write_text(s)
v=(_repo_path(r, 'validate_dinob_prenorm_069.py')).read_text().replace('dinob_prenorm_069','dinos_coarse_070').replace('dinob-prenorm-069','dinos-coarse-070').replace('Bprenorm','Scoarse').replace('6912','4224').replace('standalone_Scoarse_B504','standalone_Scoarse_S504').replace('np.corrcoef(y-d[\'Scoarse\'],y-Bmean)','np.corrcoef(y-d[\'Scoarse\'],y-Smean)').replace(".75*r003+.125*Smean+.125*d['Scoarse']",".75*r003+.125*d['Scoarse']+.125*Bmean")
a=v.index('def verify_features():');b=v.index('def main():')
v=v[:a]+'''def verify_features():
 ids,tokens,ex=e.token_inputs();r=json.loads((e.OUT/'extraction.json').read_text());assert fv.sha(e.OUT/'development_Scoarse.npz')==r['feature_sha256']
 with np.load(e.OUT/'development_Scoarse.npz',allow_pickle=False) as z:assert z['ids'].tolist()==ids;X=z['X'].copy()
 maxerr=0.
 for i,t in enumerate(tokens):
  p=t.astype(float).reshape(36,36,384);parts=[]
  for grid in (2,4):
   n=36//grid;regions=np.stack([p[a*n:(a+1)*n,b*n:(b+1)*n].mean((0,1)) for a in range(grid) for b in range(grid)]);parts.append(regions.var(0)/(p.var((0,1))+1e-4))
  val=np.concatenate(parts);np.testing.assert_allclose(val,X[i],atol=1e-12,rtol=1e-10);maxerr=max(maxerr,float(abs(val-X[i]).max()));np.testing.assert_array_equal(e.coarse(p),X[i])
 return {'rows_independently_recomputed':400,'loop_vs_reshape_max_error':maxerr,'token_cache_verified_sha256':ex['tokens_sha256'],'upstream_tokens_reextracted400_exact':'reports/fisher-067-validation.json'}

'''+v[b:]
(_repo_path(r, 'validate_dinos_coarse_070.py')).write_text(v);compile(s,'070','exec');compile(v,'verify070','exec')
