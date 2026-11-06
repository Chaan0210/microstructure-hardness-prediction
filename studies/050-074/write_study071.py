from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
r=_RepoPath(__file__).resolve().parents[1]
s=(_repo_path(r, 'dinob_prenorm_069.py')).read_text().replace('dinob_prenorm_069','dinob_denoised_071').replace('dinob-prenorm-069','dinob-denoised-071').replace('Bprenorm','Bdenoised').replace('norm=False','norm=True')
s=s.replace("b.ds.inputs(ids[0],'full504')","image_input(ids[0])").replace("b.ds.inputs(ident,'full504')","image_input(ident)")
a=s.index('def feature(')
s=s[:a]+'''def image_input(ident):
 from PIL import Image
 from denoised_grain_features import _noise_sigma
 from skimage.restoration import denoise_tv_chambolle
 with Image.open(ROOT/'data/train'/f'{ident}.png') as im:
  assert im.mode=='L' and im.size==(256,256);raw=np.asarray(im,dtype=np.float64)/255
 sigma=_noise_sigma(raw)
 den=denoise_tv_chambolle(raw,weight=.5*sigma,eps=1e-4,max_num_iter=50,channel_axis=None) if sigma>1e-12 else raw.copy()
 im=Image.fromarray(np.rint(np.clip(den,0,1)*255).astype(np.uint8)).resize((504,504),Image.Resampling.BICUBIC)
 return np.ascontiguousarray((np.repeat((np.asarray(im,dtype=np.float32)/255)[None],3,axis=0)-b.ds.base.MEAN)/b.ds.base.STD)[None]

'''+s[a:]
a=s.index('def prepare():');z=s.index('def benchmark_features():')
s=s[:a]+'''def prepare():
 prior.check();b.check();pub=json.loads((ROOT/'reports/candidate008-public-submission.json').read_text());old=prior.check();assert not json.loads((ROOT/'artifacts/dinos-coarse-070/selection.json').read_text())['advance'];OUT.mkdir(exist_ok=False)
 p={'hypothesis':'068noiseproxyweaklycorrelateswithabsoluteOOFerror(~0.10),while069rawtokenamplitudesdidnothelp. Noise-awareedge-preservingTV preprocessing mayreduce acquisitiontexturewhichDINOencodeswhilepreservingcoarsemorphology.',
 'evidence_limits':'Noiseproxycancontainrealboundarytexture;weakdescriptiveassociationisnotcausal. Earlier022/023TVwatershedfeatures improveddevelopment but004Publicregressed;useonlyasfeasibilityevidence,notsuccessclaim. NoDINOdenoisingexperimentfound.',
 'single_primary':'SameB504normTrue multilayer6912 descriptor andsameMLP as008. Onlychange grayscale input: native256 image-localHH-MAD sigma;TVweight0.5*sigma,eps1e-4,max50iters;rounduint8 thenoriginalPILbicubic504pipeline. NoD4, no extra normalization, no featureconcat.',
 'primary_formula':'.75*r003+.125*Smean3+.125*Bdenoisedmean3',
 'fixed_strength_reason':'0.5 is existing022conservativeTVfactor,chosenbeforethisexperiment,notselectedfromdenoisedDINOresults. No strengthgridor fallback.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Image-localdeterministicdenoising,no learneddatafit. StandardScaler+targetscale outertrainonly.',
 'screen':'OriginalpooledOOF strictlyimproves008 -> bothknownresplits; otherwise reject.',
 'confirmation':'All3pooledOOF improve008 plusstrictverification. Fold/bootstrapdiagnosticsonly.',
 'resource':'ExistingCPU2/MPS;3featurebenchmark<300sprojected400,<3GB;head<300sprojected15fits,<2GB. No newinstall/download/cost.',
 'sources':{n:fv.sha(ROOT/n) for n in ['dinob_denoised_071.py','validate_dinob_denoised_071.py','denoised_grain_features.py','dinob_mlp_head_063.py','dinob_mlp_063.py','dinob_complement_064.py','dinob_062.py']},
 'inputs':{**old['inputs'],**{n:fv.sha(ROOT/n) for n in ['reports/dinob-complement-064-validation.json','artifacts/error-diagnostics-068/report.json','artifacts/dinos-coarse-070/selection.json']}},
 'protected':{**old['protected'],**pub['protected_sha256'],'reports/candidate008-public-submission.json':fv.sha(ROOT/'reports/candidate008-public-submission.json')},
 'scope':'Development400only,noholdout/testreadforsel;protectedfinalcacheshashedonly.','bias':fv.CAVEAT}
 fv.write(OUT/'plan.json',p);print(json.dumps({'prepared':True,'hypothesis':p['hypothesis']}),flush=True)

'''+s[z:]
s=s.replace('Pre-LayerNorm ViT-B descriptor, unchanged008 dimensions, recipe and MLP.','Image-local conservative TV denoising before unchanged frozen ViT-B.')
(_repo_path(r, 'dinob_denoised_071.py')).write_text(s)
v=(_repo_path(r, 'validate_dinob_prenorm_069.py')).read_text().replace('dinob_prenorm_069','dinob_denoised_071').replace('dinob-prenorm-069','dinob-denoised-071').replace('Bprenorm','Bdenoised').replace('norm=False','norm=True').replace("e.b.ds.inputs(ident,'full504')","independent_input(ident)")
a=v.index('def verify_features():')
v=v[:a]+'''def independent_input(ident):
 from PIL import Image
 from skimage.restoration import denoise_tv_chambolle
 with Image.open(ROOT/'data/train'/f'{ident}.png') as im:raw=np.array(im).astype(float)/255
 blocks=raw.reshape(128,2,128,2);hh=(blocks[:,0,:,0]+blocks[:,1,:,1]-blocks[:,0,:,1]-blocks[:,1,:,0])/2;sigma=float(np.median(abs(hh))/.6744897501960817)
 den=denoise_tv_chambolle(raw,weight=.5*sigma,eps=1e-4,max_num_iter=50,channel_axis=None) if sigma>1e-12 else raw.copy()
 im=Image.fromarray(np.rint(np.clip(den,0,1)*255).astype(np.uint8)).resize((504,504),Image.Resampling.BICUBIC);rgb=np.tile(np.asarray(im,dtype=np.float32)[None]/255,(3,1,1));a=((rgb-e.b.ds.base.MEAN)/e.b.ds.base.STD)[None];np.testing.assert_array_equal(a,e.image_input(ident));return a

'''+v[a:]
(_repo_path(r, 'validate_dinob_denoised_071.py')).write_text(v);compile(s,'071','exec');compile(v,'verify071','exec')
