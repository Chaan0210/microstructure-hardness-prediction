from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c,engine as e
from common import R,P,np,torch
c.check();records=[]
with c.Monitor('validation',300):
 for split in c.SPLITS:
  d=c.data(split);oof=c.load(R/split/'OOF.npz');assert c.sha(R/split/'OOF.npz')==c.read(R/'training-complete.json')['OOF_SHA256'][split]
  for k in range(5):
   folder=R/split/f'fold-{k}';fit=c.read(folder/'fit.json');T,V=c.indices(d,k);norm=c.norm(d,T);poison=dict(d);poison['y']=d['y'].copy();poison['y'][V]=np.nan;assert c.norm(poison,T)==norm;streams=c.load(folder/'streams.npz');oldfolder=P/f'experiments/auxiliary-repartition-201/{split}/fold-{k}-detach';old=c.load(oldfolder/'streams.npz')
   for name in ['order','rot','flip','lr']:np.testing.assert_array_equal(streams[name],old[name])
   np.testing.assert_array_equal(streams['values'],old['values'][:,[0,2]])
   assert streams['rng_CPU'].shape==streams['rng_MPS'].shape==(800,)
   np.testing.assert_array_equal(streams['EMA_update_indices'],np.arange(1,801));assert fit['optimizer_reference_exact_checks']==fit['EMA_no_side_effect_checks']==800;assert fit['BatchNorm_modules']==0 and fit['GroupNorm_modules']==4 and fit['buffers']==[]
   opt=torch.load(_repo_io(folder/'optimizer-verification.pt'),map_location='cpu',weights_only=True);a,b=opt['optimizer'],opt['reference_optimizer'];assert a['param_groups']==b['param_groups'] and opt['scheduler']==opt['reference_scheduler']
   historic=torch.load(_repo_io(oldfolder/'optimizer-final.pt'),map_location='cpu',weights_only=True);e.assert_tree(a,historic['main']);assert opt['scheduler']==historic['schedulers'][0]
   for key,state in a['state'].items():
    for n,x in state.items():
     if torch.is_tensor(x):assert torch.equal(x,b['state'][key][n])
     else:assert x==b['state'][key][n]
    assert int(state['step'])==800
   for proof in opt['proof_samples']:assert proof['live_optimizer_parameters_moments_SHA256']==proof['reference_SHA256'] and proof['optimizer_reference_exact'] and proof['EMA_side_effects_absent']
   torch.manual_seed(20261003+k);init=c.legacy.TinyCNN();initstate=torch.load(_repo_io(folder/'ema-update-0.pt'),map_location='cpu',weights_only=True);assert all(torch.equal(v,init.state_dict()[n]) for n,v in initstate.items());del init
   sd800=torch.load(_repo_io(folder/'ema-update-800.pt'),map_location='cpu',weights_only=True);direct=torch.load(_repo_io(folder/'EMA-direct-float64-rounded.pt'),map_location='cpu',weights_only=True);u=None
   for ep in range(76,81):
    sd=torch.load(_repo_io(folder/f'epoch-{ep}.pt'),map_location='cpu',weights_only=True)
    if u is None:u={n:v/5 for n,v in sd.items()}
    else:
     for n,v in sd.items():u[n]+=v/5
   checks={}
   for arm in ['last5_uniform','update_ema']:
    obj=torch.load(_repo_io(folder/f'{arm}.pt'),map_location='cpu',weights_only=True);assert obj['normalization']==norm and obj['train_ids']==d['ids'][T].tolist() and obj['validation_ids']==d['ids'][V].tolist() and obj['seed']==20261003+k;expected=u if arm=='last5_uniform' else sd800;assert all(torch.equal(v,obj['state_dict'][n]) for n,v in expected.items());m=c.legacy.TinyCNN().to(c.device);m.load_state_dict(obj['state_dict']);m.requires_grad_(False);before=c.statehash(m.state_dict());mp=e.predict(m,d['images'],V,norm,c.device);np.testing.assert_array_equal(mp,oof[arm][V]);m=m.to('cpu');cp=e.predict(m,d['images'],V,norm,'cpu');err=float(abs(cp-mp).max());assert err<=.001 and c.statehash(m.state_dict())==before;rec={'MPS_reload_error':0.,'CPU_MPS_error':err,'state_unchanged':True}
    if arm=='update_ema':
     m.load_state_dict(direct);dp=e.predict(m,d['images'],V,norm,'cpu');de=float(abs(dp-cp).max());assert de<=.001;rec['independent_float64_state_prediction_error']=de
    checks[arm]=rec;del m
   records.append({'split':split,'historical_final_main_optimizer_scheduler_exact':True,'fold':k,'checks':checks,'EMA_initialization_exact_theta0':True,'final_E800_exact':True,'uniform_last5_reconstructed_exact':True,'live_reference_optimizer_final_exact':True,'sample_RNG_optimizer_proof_valid':True,'normalization_poison_exact':True});c.emit({'split':split,'validated_fold':k,'checks':checks})
   if c.device=='mps':torch.mps.empty_cache()
 weight=.99**800+sum(.01*.99**(800-t) for t in range(1,801));assert abs(weight-1)<1e-12
 c.write(R/'validation.json',{'passed':True,'folds':records,'EMA_effective_weight_sum':weight,'gradient_fits':10,'additional_gradient_validation_fits':0,'gradient_updates':8000,'optimizer_reference_verification_steps':8000,'performance_scores_computed':False,'OOF_SHA256':c.read(R/'training-complete.json')['OOF_SHA256'],'plan_SHA256':c.sha(R/'plan.json'),'torch':torch.__version__,'device':c.device})
c.check()
