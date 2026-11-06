from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,np,torch
from study import e,RECIPE,inputs,check
import math,time,gc
name='swin_v2_t';root=R/name;original=root/'fold-0/last_stage112';poison=root/'audit-label-poison-last_stage112'

def compare(a,b):
 aa=e.ck(a/'model.pt');bb=e.ck(b/'model.pt');params={};changed=[]
 for key in ['head','tail']:
  entries=[]
  for n in aa[key]:
   x=aa[key][n];y=bb[key][n];assert x.shape==y.shape and x.dtype==y.dtype;exact=torch.equal(x,y)
   if x.is_floating_point():diff=float((x-y).abs().max());l2=float(torch.linalg.vector_norm(x-y)/max(float(torch.linalg.vector_norm(x)),1e-12))
   else:diff=float((x!=y).sum());l2=None
   entries.append({'name':n,'exact':exact,'maxabs':diff,'relative_L2':l2})
   if not exact:changed.append(key+'.'+n)
  params[key]={'all_exact':all(q['exact'] for q in entries),'maxabs':max(q['maxabs'] for q in entries),'entries':entries}
 ap=c.load(a/'predictions.npz')['prediction'];bp=c.load(b/'predictions.npz')['prediction'];u=c.load(a/'updates.npz')['values'];v=c.load(b/'updates.npz')['values'];np.testing.assert_array_equal(u[:,:5],v[:,:5]);bad=np.argwhere(u[:,5:]!=v[:,5:]);grad={'first_different_update':int(bad[0,0]) if len(bad) else None,'first_different_norm_column':int(bad[0,1]+5) if len(bad) else None,'maxabs':float(abs(u[:,5:]-v[:,5:]).max())};assert aa['order_adaptation_sha256']==bb['order_adaptation_sha256'] and aa['adaptation_seed']==bb['adaptation_seed'] and aa['recipe']==bb['recipe'];return {'a':str(a.relative_to(R)),'b':str(b.relative_to(R)),'all_weights_exact':not changed,'changed_parameter_tensors':len(changed),'changed_names':changed,'parameters':params,'prediction_maxabs':float(abs(ap-bp).max()),'prediction_RMS_difference':float(np.sqrt(np.mean((ap-bp)**2))),'predictions_finite':bool(np.isfinite(ap).all() and np.isfinite(bp).all()),'prediction_within_existing_002':bool(np.max(abs(ap-bp))<.002),'gradient_trace':grad,'sample_orders_seed_recipe_actual_LR_updates_exact':True}

def main():
 check();e.CURRENT_MODEL=name;X,d=inputs(name);cache=np.load(_repo_io(root/'prefix.npy'),mmap_mode='r');T=np.array(c.plan['folds']['0']['train']);V=np.array(c.plan['folds']['0']['validation']);o=e.ck(root/'fold-0/initial-head.pt');assert torch.equal(o['feature_scale'],e.ck(poison/'model.pt')['feature_scale']);before=compare(original,poison);c.write(R/'swin-repro-failure-initial.json',before);c.emit({k:v for k,v in before.items() if k not in ['parameters','changed_names']})
 with c.Monitor('swin-repro-diagnostic',300):
  out=root/'audit-identical-labels-last_stage112';out.mkdir();e.train(o,cache,X,d['y'][T].copy(),RECIPE,202710186+99,out,V,train_tail=True);again=compare(original,out);pair=compare(poison,out);strict=all(z['all_weights_exact'] for z in [before,again,pair]);report={'purpose':'Oneexact-same-labelrepetitionafterstrictSwinpoisonreplayfailure;noLR/seed/architecture/thresholdchange;notanewselectiontrial.','strict_bitwise_reproducibility_passed':strict,'original_vs_poison':before,'original_vs_identical_labels':again,'poison_vs_identical_labels':pair,'interpretation':'Ifsame-labelrepetitionalsodiffers, mismatchdoesnotrequireheldoutlabelchange. Device/runtimefloatingexecutionnon-determinismisaninference,notaprovenkernelrootcause. Neverrelaxstrictcriterionafterobservingfailure.','same_label_difference_observed':not again['all_weights_exact'],'no_threshold_relaxation':True,'cost':0};c.write(R/'swin-repro-diagnostic.json',report);c.emit({'strict_bitwise_reproducibility_passed':strict,'original_vs_poison_prediction_maxabs':before['prediction_maxabs'],'original_vs_same_label_prediction_maxabs':again['prediction_maxabs'],'same_label_weights_exact':again['all_weights_exact']})
 check()
if __name__=='__main__':main()
