import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
from common import np,R,O,PLAN,SPLITS
from residual_math import residual,independent
import joblib
from sklearn.model_selection import GroupKFold
KINDS=('none','ridge1','ridge01','rbf1','rbf01');W=np.array([0.,.3,.3,.2,.2]);assert tuple(residual.KINDS)==KINDS

def nodes(si,k):return [n for n in PLAN['nodes'] if n['si']==si and n['outer']==k]
def coefficient(p,y):
 center=float(p.mean());target=float(y.mean());z=p-center;slope=float(z@(y-target)/(z@z));return {'prediction_center':center,'target_center':target,'slope':slope,'intercept':target-slope*center}
def apply(m,p):return m['target_center']+m['slope']*(p-m['prediction_center'])
def manual_affine(p,y,v):return np.column_stack([np.ones(len(v)),v])@np.linalg.lstsq(np.column_stack([np.ones(len(p)),p]),y,rcond=None)[0]
def raw(P,old,new,arm):
 assert arm in ['R','N'];Q=P.copy();oldmean=old.mean(0);Q[:,2]=oldmean if arm=='R' else np.concatenate([old,new],axis=0).mean(0)
 result=Q@W;manual=.3*Q[:,1]+.3*Q[:,2]+.2*Q[:,3]+.2*Q[:,4];np.testing.assert_allclose(result,manual,atol=1e-10,rtol=0)
 if arm=='N':
  Qr=P.copy();Qr[:,2]=oldmean;np.testing.assert_allclose(result-Qr@W,.3*(Q[:,2]-oldmean),atol=1e-10,rtol=0)
 return result

def fold_inputs(z,si,k,arm,new_predictions=None):
 T,V=c.indices(z,si,k);members=nodes(si,k);lookup={int(x):j for j,x in enumerate(T)};middle=np.full(len(T),np.nan);cells=[]
 outer=raw(z['outerP'][si,V],z['oldouter'][si][:,V],z['newouter'][si][:,V],arm)
 for j in range(3):
  node=next(n for n in members if n['middle']==j and n['type']=='middle');d=c.node_data(node);A=d['train'];B_hold=d['validation'];extra=None if arm=='R' else new_predictions[node['name']]
  mid=raw(d['P'],d['oldseeds'],extra,arm);middle[[lookup[int(x)] for x in B_hold]]=mid;deep=np.full(len(A),np.nan);lookA={int(x):i for i,x in enumerate(A)};sources=[]
  for ell in range(3):
   dn=next(n for n in members if n['middle']==j and n['deep']==ell);dd=c.node_data(dn);assert set(dd['train'])|set(dd['validation'])==set(A) and not(set(dd['train'])|set(dd['validation']))&(set(B_hold)|set(V));extra=None if arm=='R' else new_predictions[dn['name']];dp=raw(dd['P'],dd['oldseeds'],extra,arm);deep[[lookA[int(x)] for x in dd['validation']]]=dp;sources.append({'name':dn['name'],'cache_sha256':dn['cache_sha256']})
  assert np.isfinite(deep).all();cells.append({'j':j,'A':A,'B_hold':B_hold,'middle':mid,'deep':deep,'short_dino_B':d['short'],'node':node['name'],'sources':sources})
 assert np.isfinite(middle).all();return {'T':T,'V':V,'outer':outer,'middle':middle,'cells':cells,'arm':arm}

def bootstrap_weights(loss,groups,seed):
 unique,inverse=np.unique(groups,return_inverse=True);draw=np.random.default_rng(seed).integers(0,len(unique),(512,len(unique)));counts=np.stack([np.bincount(row,minlength=len(unique)) for row in draw]);wins=np.argmin(counts[:,inverse]@loss.T,axis=1);w=np.bincount(wins,minlength=5)/512;return w,draw,wins

def manual_weights(loss,groups,draw):
 unique=np.unique(groups);grouped=np.column_stack([loss[:,groups==g].sum(1) for g in unique]);win=np.array([np.argmin(grouped[:,row].sum(1)) for row in draw]);return np.array([(win==i).mean() for i in range(5)]),win

def anchor_for(cell,y):
 A=cell['A'];m=coefficient(cell['deep'],y[A]);anchor=.9*apply(m,cell['middle'])+.1*cell['short_dino_B'];return anchor,m

def fit_fold(z,si,k,arm,folder,new_predictions=None,labels=None):
 # No reads of old affine/anchor/residual caches in this fitting function.
 folder.mkdir(parents=True,exist_ok=False);y=z['y'] if labels is None else labels;fd=fold_inputs(z,si,k,arm,new_predictions);T=fd['T'];V=fd['V'];assert np.isfinite(y[T]).all()
 outercal=coefficient(fd['middle'],y[T]);base=.9*apply(outercal,fd['outer'])+.1*z['short'][si,V];corrections=[];reports=[]
 for cell in fd['cells']:
  j=cell['j'];A=cell['A'];B_hold=cell['B_hold'];anchor,ac=anchor_for(cell,y);res=y[B_hold]-anchor;path=folder/f'block-{j}';path.mkdir();ip=np.full((5,len(B_hold)),np.nan);folds=list(GroupKFold(3,shuffle=True,random_state=20271013+100*k+j).split(z['X'][B_hold],res,z['groups'][B_hold]));hashes={};splits=[]
  for t,(it,iv) in enumerate(folds):
   train=B_hold[it];valid=B_hold[iv];assert not set(z['groups'][train])&set(z['groups'][valid]);splits.append({'train':train.tolist(),'validation':valid.tolist()})
   for ci,kind in enumerate(KINDS):
    m=residual.fit(z['X'][train],res[it],PLAN['blocks'],kind);ip[ci,iv]=residual.predict(m,z['X'][valid]);f=path/f'inner-{t}-{kind}.joblib';joblib.dump({'model':m,'train':train,'validation':valid,'arm':arm,'residual_train_sha256':c.ah(res[it])},f);hashes[f.name]=c.H(f)
  assert np.isfinite(ip).all();loss=(ip-res)**2;seed=20271204+10000*si+100*k+j;w,draw,wins=bootstrap_weights(loss,z['groups'][B_hold],seed);full=[]
  for kind in KINDS:
   m=residual.fit(z['X'][B_hold],res,PLAN['blocks'],kind);full.append(residual.predict(m,z['X'][V]));f=path/f'full-{kind}.joblib';joblib.dump({'model':m,'train':B_hold,'validation':V,'arm':arm,'residual_train_sha256':c.ah(res)},f);hashes[f.name]=c.H(f)
  full=np.stack(full);correction=w@full;corrections.append(correction)
  # All B_hold labels can change residual/selection, but cannot change its anchor.
  for poison in [12345.,float('nan')]:
   py=y.copy();py[B_hold]=poison;py[V]=poison;pa,pm=anchor_for(cell,py);assert pm==ac;np.testing.assert_array_equal(pa,anchor)
  c.arrays(path/'predictions.npz',A=A,B_hold=B_hold,V=V,raw_middle=cell['middle'],raw_deep=cell['deep'],anchor=anchor,residual=res,inner_predictions=ip,loss=loss,weights=w,draws=draw,winners=wins,full_predictions=full,correction=correction)
  br={'arm':arm,'split':SPLITS[si],'outer':k,'block':j,'anchor_calibrator':ac,'splits':splits,'bootstrap_seed':seed,'weights':w.tolist(),'anchor_B_and_V_poison_exact':True,'anchor_sha256':c.ah(anchor),'residual_sha256':c.ah(res),'raw_middle_sha256':c.ah(cell['middle']),'raw_deep_sha256':c.ah(cell['deep']),'models_sha256':hashes,'predictions_sha256':c.H(path/'predictions.npz')};c.W(path/'report.json',br);reports.append(br)
 correction=np.mean(corrections,axis=0);primary=.9*(base+correction)+.1*z['L'][si,V];assert np.isfinite(primary).all()
 c.arrays(folder/'predictions.npz',T=T,V=V,raw_outer=fd['outer'],raw_middle=fd['middle'],base=base,correction=correction,primary=primary)
 r={'arm':arm,'split':SPLITS[si],'fold':k,'outer_calibrator':outercal,'block_reports_sha256':{f'block-{j}/report.json':c.H(folder/f'block-{j}/report.json') for j in range(3)},'predictions_sha256':c.H(folder/'predictions.npz'),'fit_only_T_labels':True,'R_or_N_anchors_computed_here':True,'source_function':'arm-explicit204; no oldanchor/residual reads'};c.W(folder/'report.json',r);return primary,r

def compare_poison(folder,other):
 rr=c.J(folder/'report.json');pr=c.J(other/'report.json');assert rr['outer_calibrator']==pr['outer_calibrator']
 for rel in ['predictions.npz',*[f'block-{j}/predictions.npz' for j in range(3)]]:
  a=c.Z(folder/rel);b=c.Z(other/rel)
  for key in a:np.testing.assert_array_equal(a[key],b[key])
 for j in range(3):
  a=c.J(folder/f'block-{j}/report.json');b=c.J(other/f'block-{j}/report.json');assert a['anchor_calibrator']==b['anchor_calibrator'];assert a['models_sha256']==b['models_sha256']
 return True

def reference_check(folder,z,si,k):
 split=SPLITS[si];fr=c.J(folder/'report.json');fz=c.Z(folder/'predictions.npz');cr=c.J(R/f'artifacts/cnn-seed-mean-133/{split}/pipeline-fold-{k}/report.json');refbase=c.Z(R/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/predictions.npz');errs=[]
 def close(a,b):
  err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert err<1e-9;errs.append(err)
 for key in fr['outer_calibrator']:close(fr['outer_calibrator'][key],cr['outer_calibrator'][key])
 close(fz['base'],refbase['base']);close(fz['primary'],z['R'][si,fz['V']])
 for j in range(3):
  br=c.J(folder/f'block-{j}/report.json');b=c.Z(folder/f'block-{j}/predictions.npz');old=c.Z(R/f'artifacts/spatial-joint-140/{split}/adjacent-fold-{k}/block-{j}/predictions.npz');boot=c.Z(R/f'artifacts/bootstrap-selector-144/{split}/fold-{k}-block-{j}/predictions.npz')
  for key in br['anchor_calibrator']:close(br['anchor_calibrator'][key],cr['anchor_calibrators'][j][key])
  close(b['anchor'],old['anchor_predictions']);close(b['residual'],old['residual']);close(b['inner_predictions'],old['inner_predictions']);close(b['loss'],(old['inner_predictions']-old['residual'])**2);close(b['full_predictions'],boot['full_candidate_predictions']);close(b['correction'],boot['primary'])
  for key in ['weights','draws','winners']:np.testing.assert_array_equal(b[key],boot[key])
 return max(errs)

def validate_fold(folder,z,si,k,arm,new_predictions=None):
 fd=fold_inputs(z,si,k,arm,new_predictions);y=z['y'];T=fd['T'];V=fd['V'];fr=c.J(folder/'report.json');fz=c.Z(folder/'predictions.npz');assert c.H(folder/'predictions.npz')==fr['predictions_sha256'];maxerr=0.;models=0;corr=[]
 def close(a,b,tol=2e-9):
  nonlocal maxerr
  err=float(np.max(np.abs(a-b)));assert err<tol;maxerr=max(maxerr,err)
 expected=.9*manual_affine(fd['middle'],y[T],fd['outer'])+.1*z['short'][si,V];close(expected,fz['base'])
 for cell in fd['cells']:
  j=cell['j'];A=cell['A'];B_hold=cell['B_hold'];path=folder/f'block-{j}';br=c.J(path/'report.json');assert c.H(path/'report.json')==fr['block_reports_sha256'][f'block-{j}/report.json'];b=c.Z(path/'predictions.npz');assert c.H(path/'predictions.npz')==br['predictions_sha256'];anchor=.9*manual_affine(cell['deep'],y[A],cell['middle'])+.1*cell['short_dino_B'];close(anchor,b['anchor']);res=y[B_hold]-b['anchor'];np.testing.assert_array_equal(res,b['residual']);ip=np.empty_like(b['inner_predictions']);folds=list(GroupKFold(3,shuffle=True,random_state=20271013+100*k+j).split(z['X'][B_hold],res,z['groups'][B_hold]));full=[]
  for t,(it,iv) in enumerate(folds):
   for ci,kind in enumerate(KINDS):
    f=path/f'inner-{t}-{kind}.joblib';assert c.H(f)==br['models_sha256'][f.name];obj=joblib.load(f);np.testing.assert_array_equal(obj['train'],B_hold[it]);np.testing.assert_array_equal(obj['validation'],B_hold[iv]);assert obj['arm']==arm and obj['residual_train_sha256']==c.ah(res[it]);actual=residual.predict(obj['model'],z['X'][B_hold[iv]]);pp=independent.manual(obj['model'],z['X'][B_hold[it]],res[it],z['X'][B_hold[iv]],PLAN['blocks']);close(actual,pp);np.testing.assert_array_equal(actual,b['inner_predictions'][ci,iv]);ip[ci,iv]=actual;models+=1
  # Independently verify inner data exclusion without refitting additional models.
  for t,(it,iv) in enumerate(folds):
   poison=res.copy();poison[iv]=np.nan;np.testing.assert_array_equal(poison[it],res[it])
  for ci,kind in enumerate(KINDS):
   f=path/f'full-{kind}.joblib';assert c.H(f)==br['models_sha256'][f.name];obj=joblib.load(f);np.testing.assert_array_equal(obj['train'],B_hold);np.testing.assert_array_equal(obj['validation'],V);assert obj['arm']==arm;actual=residual.predict(obj['model'],z['X'][V]);pp=independent.manual(obj['model'],z['X'][B_hold],res,z['X'][V],PLAN['blocks']);close(actual,pp);np.testing.assert_array_equal(actual,b['full_predictions'][ci]);full.append(actual);models+=1
  loss=(ip-res)**2;draw=np.random.default_rng(br['bootstrap_seed']).integers(0,len(np.unique(z['groups'][B_hold])),(512,len(np.unique(z['groups'][B_hold]))));np.testing.assert_array_equal(draw,b['draws']);w,win=manual_weights(loss,z['groups'][B_hold],draw);np.testing.assert_array_equal(w,b['weights']);np.testing.assert_array_equal(win,b['winners']);manual=sum(float(wi)*pi for wi,pi in zip(w,full));close(manual,b['correction']);corr.append(manual)
 close(.9*(fz['base']+np.mean(corr,axis=0))+.1*z['L'][si,V],fz['primary']);return {'passed':True,'models_verified':models,'max_independent_error':maxerr}
