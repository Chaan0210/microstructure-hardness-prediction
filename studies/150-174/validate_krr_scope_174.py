"""Independent all9x3 KRR algebra, downstream reconstruction and label-scope replays."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,json
import numpy as np
from sklearn.model_selection import GroupKFold
import krr_scope_174 as t
import validate_quality_gate_165 as oldv
R=t.R;OUT=t.OUT;e=t.e;a=t.a;s=t.s;x=t.x

def manual(X,y,Q,width,alpha):
 lo=X.min(0);ran=X.max(0)-lo;den=np.where(ran<10*np.finfo(float).eps,1.,ran);Xb=np.clip(X/den-lo/den,0,1);Qb=np.clip(Q/den-lo/den,0,1);mu=Xb.mean(0);var=Xb.var(0);eps=np.finfo(float).eps;sd=np.sqrt(var);sd[var<=len(X)*eps*var+(len(X)*mu*eps)**2]=1.;blocks=np.array(t.load_blocks);w=np.array([1/np.sqrt((blocks==b).sum()) for b in blocks]);Z=(Xb-mu)/sd*w;W=(Qb-mu)/sd*w;D=np.sum((Z[:,None,:]-Z[None,:,:])**2,2);pair=D[np.triu_indices(len(Z),1)];gamma=1/max(float(np.median(pair[pair>1e-12]))*width**2,1e-12);K=np.exp(-gamma*D)+alpha*np.eye(len(X));ym=float(y.mean());ys=max(float(y.std()),1e-8);coef=np.linalg.solve(K,(y-ym)/ys);pred=np.exp(-gamma*np.sum((W[:,None,:]-Z[None,:,:])**2,2))@coef*ys+ym;return pred,{'lo':lo,'range':ran,'mu':mu,'var':var,'sd':sd,'Z':Z,'gamma':gamma,'ym':ym,'ys':ys,'coef':coef}

def verify_node(d,node,arm,path):
 r=e.read(path/'report.json');z=a.readnp(path/'predictions.npz',r['predictions_sha256']);U=node['U'];Q=node['Q'];S=U if arm=='all' else t.subset(d,U);np.testing.assert_array_equal(z['U'],U);np.testing.assert_array_equal(z['Q'],Q);np.testing.assert_array_equal(z['S'],S);assert not set(d['g'][U])&set(d['g'][Q]);assert set(S)<=set(U);folds=list(GroupKFold(3,shuffle=True,random_state=t.fv.INNER_SEED+node['offset']).split(d['K'][S],d['y'][S],d['g'][S]));ip=np.empty_like(z['inner']);count=0;err=0.;scores=[];ci=0
 for width in t.fv.WIDTHS:
  for alpha in t.fv.ALPHAS:
   for ell,(I,H) in enumerate(folds):
    assert r['splits'][ell]=={'train':S[I].tolist(),'validation':S[H].tolist()};assert not set(d['g'][S[I]])&set(d['g'][S[H]]);ip[ci,H],_=manual(d['K'][S[I]],d['y'][S[I]],d['K'][S[H]],width,alpha);count+=1
   scores.append(e.rmse(d['y'][S],ip[ci]));ci+=1
 err=max(err,float(abs(ip-z['inner']).max()));assert err<1e-7,err;np.testing.assert_allclose(scores,[v['inner_rmse'] for v in r['scores']],atol=1e-8,rtol=0);assert int(np.argmin(scores))==r['chosen_index'];choice=r['chosen'];pred,m=manual(d['K'][U],d['y'][U],d['K'][Q],choice['bandwidth'],choice['alpha']);err=max(err,float(abs(pred-z['predictions']).max()));assert err<1e-7
 file=path/'model.joblib';assert e.sha(file)==r['model_sha256'];obj=t.joblib.load(_repo_io(file));np.testing.assert_array_equal(obj['train'],U);np.testing.assert_array_equal(obj['selection'],S);model=obj['model'];np.testing.assert_array_equal(model.predict(d['K'][Q]),z['predictions']);assert len(model.X_)==len(U);np.testing.assert_allclose(model.bound_.data_min_,m['lo'],atol=1e-12,rtol=0);np.testing.assert_allclose(model.bound_.data_range_,m['range'],atol=1e-12,rtol=0);np.testing.assert_allclose(model.scale_.mean_,m['mu'],atol=1e-12,rtol=0);np.testing.assert_allclose(model.scale_.var_,m['var'],atol=1e-12,rtol=0);np.testing.assert_allclose(model.scale_.scale_,m['sd'],atol=1e-12,rtol=0);np.testing.assert_allclose(model.X_,m['Z'],atol=1e-10,rtol=0);np.testing.assert_allclose(model.coef_,m['coef'],atol=1e-7,rtol=0);assert abs(model.ym_-m['ym'])<1e-12 and abs(model.ys_-m['ys'])<1e-12
 return z['predictions'],count+1,err

def reconstruct(d,split,k,arm,path):
 dd=dict(d);dd['P']=d['P'].copy();dd['middleP']=dict(d['middleP']);dd['middleP'][k]=d['middleP'][k].copy();dd['deepP']=dict(d['deepP']);dd['cells']=dict(d['cells']);T=np.flatnonzero(d['fid']!=k);look={int(i):j for j,i in enumerate(T)};count=0;error=0.
 for j in range(3):
  A,B,P,short,ref=d['cells'][k,j];dd['cells'][k,j]=(A,B,P.copy(),short,ref);dd['deepP'][k,j]=d['deepP'][k,j].copy()
 for node in t.nodes(d,split,k):
  p,c,er=verify_node(d,node,arm,path/node['name']);count+=c;error=max(error,er);Q=node['Q']
  if node['name']=='outer':dd['P'][Q,1]=p
  elif node['name'].startswith('middle'):
   j=int(node['name'].split('-')[1]);dd['cells'][k,j][2][:,1]=p;dd['middleP'][k][[look[int(i)] for i in Q],1]=p
  else:
   j=int(node['name'].split('-')[1]);A=dd['cells'][k,j][0];la={int(i):ii for ii,i in enumerate(A)};dd['deepP'][k,j][[la[int(i)] for i in Q],1]=p
 return dd,count,error

def independent_affine(P,y,Q):
 w=np.array([0,.3,.3,.2,.2]);coef=np.linalg.lstsq(np.column_stack([np.ones(len(P)),P@w]),y,rcond=None)[0];return np.column_stack([np.ones(len(Q)),Q@w])@coef

def validate(split):
 t.check();start=time.perf_counter();d=t.load(split);t.load_blocks=d['Kblocks'];rr=e.read(OUT/split/'report.json');saved=a.readnp(OUT/split/'predictions.npz',rr['predictions_sha256']);assert saved['ids'].tolist()==d['ids'];np.testing.assert_array_equal(saved['y'],d['y']);kr=0;res=0;ke=0.;re=0.;metae=0.;poison={}
 for k in range(5):
  dd,_=x.replace(d,split,k,x.OUT/split/f'cnn-fold-{k}',False);T=np.flatnonzero(d['fid']!=k);V=np.flatnonzero(d['fid']==k);path=OUT/split/f'fold-{k}';fr=e.read(path/'report.json');assert e.sha(path/'report.json')==rr['fold_reports'][f'fold-{k}/report.json']
  for arm in t.ARMS:
   D,c,er=reconstruct(dd,split,k,arm,path/arm/'kernels');kr+=c;ke=max(ke,er);root=path/arm/'pipeline';r=e.read(root/'report.json');assert e.sha(root/'report.json')==fr['arms'][arm]['pipeline_report_sha256'];z=a.readnp(root/'predictions.npz',r['predictions_sha256']);base=.9*independent_affine(D['middleP'][k],d['y'][T],D['P'][V])+.1*d['short'][V];metae=max(metae,float(abs(base-z['base']).max()));assert metae<1e-8;cor=[]
   for j in range(3):
    A,B,P,short,_=D['cells'][k,j];assert set(A)|set(B)==set(T) and not set(A)&set(B);anchor=.9*independent_affine(D['deepP'][k,j],d['y'][A],P)+.1*short;bp=root/f'block-{j}';assert e.sha(bp/'report.json')==r['block_reports'][f'block-{j}/report.json'];p,c,er=oldv.verify_block(D,B,V,anchor,bp);res+=c;re=max(re,er);cor.append(p)
   np.testing.assert_allclose(base+np.mean(cor,0),saved[arm][V],rtol=0,atol=1e-8)
  print('INDEPENDENT_FOLD_PASS',split,k,flush=True)
  if k==0:
   # Changed KRR pipeline is completely rerun; unchanged NN caches retain prior scope certificates.
   pert=dict(dd);pert['y']=dd['y'].copy();pert['y'][V]+=10000;pr=OUT/f'poison-{split}';pr.mkdir(exist_ok=False)
   for arm in t.ARMS:
    pp,nr=t.replace_krr(pert,split,k,arm,pr/f'{arm}-kernels');pv,pfr=t.downstream(pp,split,k,pr/f'{arm}-pipeline');np.testing.assert_array_equal(pv,saved[arm][V]);poison[arm]={'outer80_labels_changed':True,'all13_KRR_selections_fullfits_downstream_recomputed':True,'predictions_exact':True}
   # B labels cannot enter its A-only anchor, including all4 KRR source nodes.
   A,B,P,short,_=dd['cells'][0,0];alt=dict(dd);alt['y']=dd['y'].copy();alt['y'][B]+=10000;lookup={node['name']:node for node in t.nodes(dd,split,0)};same={};root=pr/'anchor-B';root.mkdir()
   for name in ['middle-0','deep-0-0','deep-0-1','deep-0-2']:
    p,nr=t.fit_node(alt,lookup[name],'subset80',root/name);original=a.readnp(path/'subset80/kernels'/name/'predictions.npz');np.testing.assert_array_equal(p,original['predictions']);np.testing.assert_array_equal(a.readnp(root/name/'predictions.npz')['inner'],original['inner']);same[name]=True
   node=lookup['outer'];U=node['U'];S=t.subset(dd,U);outside=np.setdiff1d(U,S);alt=dict(dd);alt['y']=dd['y'].copy();alt['y'][outside]+=10000;_,_,ip,sc,idx=t.select(alt,U,node['offset'],'subset80');original=e.read(path/'subset80/kernels/outer/report.json');np.testing.assert_array_equal(ip,a.readnp(path/'subset80/kernels/outer/predictions.npz')['inner']);assert sc==original['scores'] and idx==original['chosen_index'];poison['anchorB_four_nodes_exact']=same;poison['selection_complement_labels_excluded']={'changed':len(outside),'inner_predictions_scores_chosen_exact':True}
   bp=path/'subset80/pipeline/block-0';br=e.read(bp/'report.json');bz=a.readnp(bp/'predictions.npz');I=np.array(br['splits'][0]['train']);H=np.array(br['splits'][0]['validation']);alt=dd['y'].copy();alt[H]+=10000;look={int(v):i for i,v in enumerate(bz['B'])};target=alt[I]-bz['anchor'][[look[int(v)] for v in I]]
   for kind in s.KINDS:
    m=t.n.fit(dd['X'][I],target,dd['blocks'],kind);old=t.joblib.load(_repo_io(bp/f'inner-0-{kind}.joblib'))['model'];np.testing.assert_array_equal(t.n.predict(m,dd['X'][H]),t.n.predict(old,dd['X'][H]))
   poison['residual_inner_H_exact']=True
 result={'passed':True,'split':split,'KRR_fits_independently_recomputed':kr,'residual_models_independently_verified':res,'affine_fits':40,'KRR_max_error':ke,'residual_max_error':re,'affine_max_error':metae,'poison':poison,'subset_no_labels_hash_rule_group_integrity':True,'all_final_KRR_stats_use_fullU':True,'control018_max_error':float(abs(saved['all']-d['reference018']).max()),'report_sha256':e.sha(OUT/split/'report.json'),'source_sha256':e.sha(R/'validate_krr_scope_174.py'),'upstream_NN_scope_certificate':e.sha(R/'reports/nested-selector-146-audit.json'),'fresh_neural_retraining':False,'seconds':time.perf_counter()-start};t.check();write=t.write;write(OUT/f'validation-{split}.json',result);print('VALIDATION_PASS',json.dumps(result),flush=True)
if __name__=='__main__':validate(sys.argv[1])
