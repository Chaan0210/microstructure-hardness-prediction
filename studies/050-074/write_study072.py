from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
r=_RepoPath(__file__).resolve().parents[1]
base=(_repo_path(r, 'dinob_denoised_071.py')).read_text();run=base[base.index('def run(split):'):base.index("if __name__=='__main__':")]
run=run.replace('new=.75*r003+.125*Smean+.125*mean','new=.75*r003+.125*Smean+.0625*Bmean+.0625*mean')
s='''"""Exploratory fixed raw/TV prediction mean after071 standalone replacement failed."""
import os,sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import dinob_denoised_071 as base
import numpy as np,torch
prior=base.prior;fv=base.fv;h=base.h;SEEDS=base.SEEDS;b=base.b
OUT=ROOT/'artifacts/denoise-complement-072'
inputs=base.inputs

def check():
 base.check();p=json.loads((OUT/'plan.json').read_text())
 for path,digest in {**p['sources'],**p['inputs'],**p['protected']}.items():assert fv.sha(ROOT/path)==digest,path
 return p

def prepare():
 base.check();v=json.loads((ROOT/'reports/dinob-denoised-071-validation.json').read_text());assert v['passed'] and not v['eligible_for_candidate'];outcome=json.loads((base.OUT/'original/report.json').read_text());pub=json.loads((ROOT/'reports/candidate008-public-submission.json').read_text());OUT.mkdir(exist_ok=False)
 p={'hypothesis':'RawB andTVB have close standalone RMSE13.6158/13.6676 but residualcorrelation0.93467. Equalpredictionmean maycancel representation-specific errors withoutfittingextraheads orweights. It could fail in otherpartitions.',
 'posthoc_disclosure':'Newexploratoryfollowup chosenafter071 failed its primaryreplacement screen. Doesnotreverse071 gate,claimpreregistrationbefore071 results,orclaimindependentvalidation. Do not search additionalratios ifthisfails.',
 'single_primary':'.75*r003+.125*Smean+.0625*Brawmean+.0625*BTVmean = .5*008+.5*071primary. This retains totalBweight.125 and usesfixed50:50 likepriorS/Bcomplementarity064,notoptimizedweights.',
 'screen':'OriginalpooledOOF strictlyimproves008 -> train071samefixedheads onbothknownresplits andevaluatefixedmixture. Foldcount/bootstrapdiagnosticonly.',
 'confirmation':'All3pooledOOF strictlyimprove008 plusstrictverification -> lockexploratoryrecipe beforefull500/testcandidatecreation.',
 'seeds':list(SEEDS),'config':h.CONFIG,'effective_aux_weight':0.,'preprocessing':'Onlytrainingfold StandardScaler+targetscale. Frozenimage-localTVfeaturesverified071. Originalreuses15existing071models. Eachresplit fits15sameheads.',
 'sources':{n:fv.sha(ROOT/n) for n in ['denoise_complement_072.py','validate_denoise_complement_072.py','dinob_denoised_071.py','dinob_mlp_head_063.py']},
 'inputs':{n:fv.sha(ROOT/n) for n in ['reports/dinob-denoised-071-validation.json','artifacts/dinob-denoised-071/original/report.json','artifacts/dinob-denoised-071/original/development_predictions.npz','reports/dinob-complement-064-validation.json']},
 'protected':{**base.check()['protected'],**pub['protected_sha256']},'resource':'ExistingCPU2threads,verified071features,30additionalheadfitsonlyiforiginalimproves,noinstall/download/cost.',
 'scope':'Development400onlyuntilmodellock;noholdout/testselection.','bias':fv.CAVEAT}
 fv.write(OUT/'plan.json',p);fv.write(OUT/'head-benchmark.json',json.loads((base.OUT/'head-benchmark.json').read_text()));print(json.dumps({'prepared':True,'hypothesis':p['hypothesis'],'posthoc_disclosure':p['posthoc_disclosure']}),flush=True)

def original():
 check();ids,X,y,g,fid,r003,r007,Smean,Bmean,ref=inputs('original');src=base.OUT/'original';out=OUT/'original';out.mkdir(exist_ok=False)
 with np.load(src/'development_predictions.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 assert d['ids'].tolist()==ids;np.testing.assert_array_equal(d['reference008'],ref);np.testing.assert_array_equal(d['fold'],fid);np.testing.assert_array_equal(d['y'],y)
 d['primary']=.75*r003+.125*Smean+.0625*Bmean+.0625*d['Bdenoised'];report=json.loads((src/'report.json').read_text());report['scores']={name:{'rmse':fv.rmse(y,d[name]),'fold_rmse':[fv.rmse(y[fid==k],d[name][fid==k]) for k in range(5)]} for name in report['scores']};report['comparison']=prior.old.comparison(y,ref,d['primary'],fid);report['models_reused_from']=str(src.relative_to(ROOT));report['seconds']=0.
 with (out/'development_predictions.npz').open('xb') as f:np.savez_compressed(f,**d)
 report['predictions_sha256']=fv.sha(out/'development_predictions.npz');fv.write(out/'report.json',report);fv.write(OUT/'selection.json',{'advance':bool(report['comparison']['gain']>0),'primary':'primary','comparison':report['comparison'],'diagnostic_only':'foldcounts/bootstrapnotgates'});print(json.dumps(report),flush=True)

'''+run+'''
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','original','20261031','20261101']);a=ap.parse_args();{'prepare':prepare,'original':original}.get(a.mode,lambda:run(a.mode))()
'''
(_repo_path(r, 'denoise_complement_072.py')).write_text(s)
v=(_repo_path(r, 'validate_dinob_denoised_071.py')).read_text().replace('import dinob_denoised_071 as e','import denoise_complement_072 as e').replace('dinob-denoised-071-validation.json','denoise-complement-072-validation.json').replace('dinob-denoised-071-screen-validation.json','denoise-complement-072-screen-validation.json').replace(".75*r003+.125*Smean+.125*d['Bdenoised']",".75*r003+.125*Smean+.0625*Bmean+.0625*d['Bdenoised']")
a=v.index('def independent_input(');b=v.index('def main():')
v=v[:a]+'''def verify_features():
 p=e.check();v=json.loads((ROOT/'reports/dinob-denoised-071-validation.json').read_text());assert v['passed'] and v['feature_verification']['rows_reextracted']==400 and v['feature_verification']['max_error']==0
 return {'verified400_exact_source':'reports/dinob-denoised-071-validation.json','source_sha256':fv.sha(ROOT/'reports/dinob-denoised-071-validation.json'),'rawimages_read_again':False}

'''+v[b:]
v=v.replace("fd=out/f'fold-{k}'","fd=(e.base.OUT/split if split=='original' else out)/f'fold-{k}'")
(_repo_path(r, 'validate_denoise_complement_072.py')).write_text(v);compile(s,'072','exec');compile(v,'verify072','exec')
