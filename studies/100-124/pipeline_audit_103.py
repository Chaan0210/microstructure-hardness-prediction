"""Close selected best011 path-audit gaps; no model selection or test access."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import sys,json,hashlib,time
from pathlib import Path
ROOT=_RepoPath(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));assert Path(sys.prefix).resolve()==_repo_path(ROOT, '.venv')
import numpy as np,torch
from PIL import Image
from torch.nn import functional as F
import dinob_062 as b
import dino_multilayer_044 as sm
import candidate006_pipeline as p6
import dinob_joint_084 as head
OUT=_repo_path(ROOT, 'artifacts/pipeline-audit-103')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 OUT.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);assert torch.backends.mps.is_available()
 pub=read(_repo_path(ROOT, 'reports/candidate011-public-submission.json'))
 for n,h in pub['protected_sha256'].items():assert sha(_repo_path(ROOT, n))==h,n
 previous=['reports/candidate008-verification.json','reports/candidate009-verification.json','reports/candidate011-verification.json','reports/frozen-complement-085-validation.json','reports/dinob-062-validation.json','reports/dino-046-summary.json'];evidence={n:sha(_repo_path(ROOT, n)) for n in previous}
 manifest=read(_repo_path(ROOT, 'artifacts/run-001/split_manifest.json'));ids=sorted(i for i,k in manifest['development_folds'].items() if k!=0);ids=[ids[j] for j in np.linspace(0,len(ids)-1,12,dtype=int)];assert not set(ids)&set(manifest['holdout'])
 plan={'scope':'Only omitted source/preprocessing/patch geometry audits. Reuse existing alignment/full500/target/reload evidence. No train.csv labels,test/holdout images or predictions decoded; source/metadata only for production path.','selected_train_ids':ids,'evidence':evidence,'sources':{n:sha(_repo_path(ROOT, n)) for n in ['pipeline_audit_103.py','candidate_pipeline.py','candidate006_pipeline.py','candidate008_pipeline.py','candidate009_pipeline.py','candidate011_pipeline.py','dino_spatial_039.py','dino_multilayer_044.py','dinob_062.py','dinob_joint_084.py']},'protected011':pub['protected_sha256']};write(_repo_path(OUT, 'plan.json'),plan)
 maxerr=0.;inverseerr=0.
 for ident in ids:
  path=_repo_path(ROOT, 'data/train', f'{ident}.png');a=p6.input_image(path);dev=b.ds.inputs(ident,'full504')[0];np.testing.assert_array_equal(a,dev)
  with Image.open(_repo_io(path)) as im:assert im.mode=='L' and im.size==(256,256);z=np.array(im.resize((504,504),Image.Resampling.BICUBIC),dtype=np.float32)/np.float32(255)
  independent=np.stack([(z-np.float32(mu))/np.float32(sd) for mu,sd in zip([.485,.456,.406],[.229,.224,.225])]);np.testing.assert_array_equal(a,independent);maxerr=max(maxerr,float(abs(a-independent).max()));inv=a*b.ds.base.STD+b.ds.base.MEAN;inverseerr=max(inverseerr,float(abs(inv-z[None]).max()))
 routes={'dev_vs_production_input_exact_images':12,'independent_input_max_error':maxerr,'inverse_affine_pixel_error':inverseerr,'native_shape':[256,256],'DINO_shape':[3,504,504],'resize':'PIL BICUBIC full frame, no crop','brightness_contrast':'Fixed ImageNet affine normalization preserves per-image intensity offsets/scales; no per-image standardization','production_train_test':'Both candidate006 and candidate008 call same path-only input_image helper; split only changes path. Previously raw10 test checks already passed; no new test decode needed.'}
 topology={}
 for arm,model,descriptor in [('S',b.ds.base.model(),lambda m,x:sm.feature(m,x).reshape(1,3456)),('B',b.model(),b.feature)]:
  model=model.to('mps');x=torch.from_numpy(p6.input_image(_repo_path(ROOT, 'data/train', f'{ids[0]}.png'))[None]).to('mps')
  with torch.inference_mode():
   official=model.patch_embed(x);conv=F.conv2d(x,model.patch_embed.proj.weight,model.patch_embed.proj.bias,stride=(14,14));manual=conv.permute(0,2,3,1).reshape(1,1296,model.embed_dim);manual=model.patch_embed.norm(manual);np.testing.assert_array_equal(official.cpu().numpy(),manual.cpu().numpy())
   pos=model.interpolate_pos_encoding(torch.cat([model.cls_token,official],1),504,504);N=model.pos_embed.shape[1]-1;M=int(N**.5);assert M*M==N;grid=model.pos_embed[:,1:].float().reshape(1,M,M,model.embed_dim).permute(0,3,1,2);kw={'scale_factor':((36+model.interpolate_offset)/M,)*2} if model.interpolate_offset else {'size':(36,36)};ip=F.interpolate(grid,mode='bicubic',antialias=model.interpolate_antialias,**kw).permute(0,2,3,1).reshape(1,1296,model.embed_dim);manualpos=torch.cat([model.pos_embed[:,:1].float(),ip],1);np.testing.assert_array_equal(pos.cpu().numpy(),manualpos.cpu().numpy())
   tokens=torch.cat([model.cls_token,manual],1)+manualpos;parts=[]
   for k,block in enumerate(model.blocks):
    tokens=block(tokens)
    if k in (5,8,11):
     z=model.norm(tokens);patch=z[:,1:];parts.append(torch.cat([z[:,0],patch.mean(1),patch.std(1,correction=0)],1))
   direct=torch.cat(parts,1);api=descriptor(model,x);error=float(abs(direct-api).max().cpu());assert error<1e-5
  topology[arm]={'patch_kernel_stride':[14,14],'grid':[36,36],'all504_pixels_covered':True,'flatten_order':'row-major y,x,channels','patch_embedding_manual_error':0.,'position_interpolation_manual_error':0.,'pretrained_position_grid':[M,M],'interpolation_offset':model.interpolate_offset,'interpolation_antialias':model.interpolate_antialias,'layers_zero_based':[5,8,11],'population_std_correction':0,'manual_full_descriptor_error':error,'pooling':'Position-aware tokens then global patch mean/std; pooling does not make orientation or translation invariant.'};del model;torch.mps.empty_cache()
 final=read(_repo_path(ROOT, 'artifacts/final-model-086/decision.json'));dev=read(_repo_path(ROOT, 'artifacts/frozen-complement-085/plan.json'));assert final['config']==dev['config']==head.CFG
 old=read(_repo_path(ROOT, 'artifacts/final-model-017/decision.json'));assert old['panel']=='run-012' and old['ensemble_weights']=={'baseline':.5,'cnn':.25,'kernel':.25}
 record={'passed':True,'new_numerical_bug_found':False,'reused_prior_verified_items':['ID/target/group alignment and fold-train statistics','full500 S/B six-head refit state reproduction','nested009 affine calibration and heldout-label perturbation','full500 new011 head exact refit','full1000 cached component/head reload plus CSV byte reproduction','raw10 test S/B feature and ensemble spot checks'],
 'newly_closed_gaps':{'input_route_equivalence':routes,'patch_and_position_semantics':topology,'full500_vs_OOF_config':{'equal':True,'epochs':12,'OOF_train_rows':320,'full_train_rows':500,'effective_batch':16,'OOF_updates':240,'full_updates':384,'interpretation':'Same epoch-based schedule intentionally gives more full-data updates; not an accidental settings mismatch. Full500 seed20270215 and outerfold seed20270215+k are intentional.'}},
 'actual_recipe':{'candidate003':'.5 baseline795 + .25 compact26 kernel + .25 single CNN128 identity prediction','candidate008':'.75 candidate003 + .125 S_MLP_mean3 + .125 B_MLP_mean3','candidate009':'Saved train-only affine transform of008','candidate011':'.9 candidate009 + .1 short frozenB head','TV31_in_best011':False,'CNN_D4':'Training randomD4; inherited003 inference identity, not3seedD4 inference.','DINO_D4':'Identity images inbest011;046 D4 mean had alreadyfailed and willnotberepeated.','rotation_assumption':'Augmentation is an existing empirical assumption, not a verified label symmetry. Official reference distinguishes alignment fromabsoluteangle but gives no complete label equation. No wholeensembleD4invariance claim.'},
 'documentation_issue':'README describes historical004 withTV31/multiseedD4; actualbest011 lineage is above. This is stale documentation, not a numeric inferencebug. Preserve historicalREADME and records; use this scopedcurrentlineage audit.',
 'next_nonduplicate_hypothesis':'FrozenViTB patch-grid phase averaging: four fixed full504 views with offsets(0,0),(0,7),(7,0),(7,7), reflect padding only bottom/right, same6912 descriptor mean. Fit exactly same011 shorthead/foldtrainnormalizer; replaceonly011 freshhead10% branch. Unlike039 crop/scale or046 rotation, tests patch-grid origin/context sensitivity. It also modifies edgecontext/absoluteposition, so cannot isolatealiasing orassume label-preserving crops. Empirical developmentCV willtest utility.',
 'unrepeated_gap':'Full1000 rawCPU DINO reextraction hasnotbeenperformed; priorraw10 plusallhashes andsamepreprocess path provide bounded coverage. No concretecache corruption evidence; do notspendtestaccess orrepeat all1000 merely tosearchforgain.',
 'source_sha256':sha(Path(__file__)),'plan_sha256':sha(_repo_path(OUT, 'plan.json')),'seconds':time.perf_counter()-start,'protected011_files_verified':20,'holdout_test_used':False,'newtraining':False,'cost':0}
 for n,h in pub['protected_sha256'].items():assert sha(_repo_path(ROOT, n))==h
 write(_repo_path(OUT, 'report.json'),record);print(json.dumps({k:v for k,v in record.items() if k not in ('newly_closed_gaps','actual_recipe','reused_prior_verified_items')}),flush=True)
if __name__=='__main__':main()
