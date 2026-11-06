"""Independent classical algebra, source scopes and neural reloads for fresh176 full500."""
from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[4]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import sys,time,json
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.model_selection import GroupKFold
import train as t
from cnn_model import TinyCNN
from feature_helpers import Short
import mlp_head as h
ROOT=t.ROOT;OUT=t.OUT

def normalized(X,Q,blocks):
 lo=X.min(0);ran=X.max(0)-lo;den=np.where(ran<10*np.finfo(float).eps,1.,ran);B=np.clip(X/den-lo/den,0,1);BV=np.clip(Q/den-lo/den,0,1);mu=B.mean(0);var=B.var(0);sd=np.sqrt(var);eps=np.finfo(float).eps;sd[var<=len(X)*eps*var+(len(X)*mu*eps)**2]=1.;w=np.array([1/np.sqrt(blocks.count(b)) for b in blocks]);return (B-mu)/sd*w,(BV-mu)/sd*w,lo,mu,var,sd

def kernel(X,y,Q,blocks,width,alpha):
 Z,V,lo,mu,var,sd=normalized(X,Q,blocks);D=np.sum((Z[:,None]-Z[None,:])**2,2);pair=D[np.triu_indices(len(X),1)];gamma=1/max(float(np.median(pair[pair>1e-12]))*width**2,1e-12);ym=float(y.mean());ys=max(float(y.std()),1e-8);coef=np.linalg.solve(np.exp(-gamma*D)+alpha*np.eye(len(X)),(y-ym)/ys);return np.exp(-gamma*np.sum((V[:,None]-Z[None,:])**2,2))@coef*ys+ym

def residual(X,y,Q,blocks,kind):
 if kind=='none':return np.zeros(len(Q))
 Z,V,*_=normalized(X,Q,blocks);rs=max(float(np.sqrt(np.mean(y*y))),1e-12);rs=1. if rs<=1e-12 else rs;lam=1. if kind.endswith('1') and not kind.endswith('01') else .1;alpha=lam*len(X)
 if kind.startswith('ridge'):return V@Z.T@np.linalg.solve(Z@Z.T+alpha*np.eye(len(X)),y/rs)*rs
 D=np.sum((Z[:,None]-Z[None,:])**2,2);tri=D[np.triu_indices(len(X),1)];positive=tri[tri>1e-12];gamma=1/float(np.median(positive)) if len(positive) else 1.;K=np.exp(-gamma*D);H=np.eye(len(X))-np.ones((len(X),len(X)))/len(X);C=H@K@H;scale=float(np.trace(C)/len(X));scale=1. if scale<=1e-12 else scale;C/=scale;ev,U=np.linalg.eigh(C);coef=U@((U.T@(y/rs))/(ev+alpha));VK=np.exp(-gamma*np.sum((V[:,None]-Z[None,:])**2,2));cross=(VK-VK.mean(1)[:,None]-K.mean(0)[None,:]+K.mean())/scale;return cross@coef*rs

def cnn_reload(obj,X,Q,device):
 m=TinyCNN().to(device);m.load_state_dict(obj['state_dict']);m.eval();nn=obj['normalization'];p=[]
 with torch.no_grad():
  for st in range(0,len(Q),32):p.extend((m(torch.from_numpy((X[Q[st:st+32]]-nn['x_mean'])/max(nn['x_std'],1e-6)).to(device)).cpu().numpy()*nn['y_std']+nn['y_mean']).tolist())
 del m;return np.array(p)

def verify_node(d,path,T,V):
 r=t.read(path/'report.json');U=np.array(r['U']);Q=np.array(r['Q']);assert set(U)<=set(T) and set(Q)<=set(T)|set(V);assert not set(U)&set(V);isroot=(path.name=='outer' and len(U)==500 and np.array_equal(U,np.arange(500)) and np.array_equal(U,Q));assert bool(r.get('root_in_sample_query',False))==isroot;assert isroot or (not set(U)&set(Q) and not set(d['g'][U])&set(d['g'][Q]));assert d['ids'][U].tolist()==r['train_ids'] and d['ids'][Q].tolist()==r['query_ids']
 for f,hh in r['files'].items():assert t.sha(path/f)==hh
 with np.load(_repo_io(path/'predictions.npz')) as z:p={key:z[key].copy() for key in z.files}
 np.testing.assert_array_equal(U,p['U']);np.testing.assert_array_equal(Q,p['Q']);err={'kernel':0.,'CPU_head':0.,'MPS_reload':0.,'CPU_MPS':0.};scores=[]
 S=np.array(r['KRR_selection']);assert set(S)<=set(U);np.testing.assert_array_equal(S,d['selection'] if r.get('root_in_sample_query',False) else U);inner=list(GroupKFold(3,shuffle=True,random_state=20261005+r['offset']).split(d['K'][S],d['y'][S],d['g'][S]))
 for width in [.25,1.,4.]:
  for alpha in [.01,.1,1.]:
   ip=np.empty(len(S))
   for I,H in inner:
    assert not set(d['g'][S[I]])&set(d['g'][S[H]]);ip[H]=kernel(d['K'][S[I]],d['y'][S[I]],d['K'][S[H]],d['Kblocks'],width,alpha)
   scores.append(t.b.rmse(d['y'][S],ip))
 kr=r['logs']['KRR'];np.testing.assert_allclose(scores,[z['inner_rmse'] for z in kr['inner_scores']],atol=1e-8,rtol=0);assert int(np.argmin(scores))==int(np.argmin([v['inner_rmse'] for v in kr['inner_scores']]));choice=kr['selected'];pred=kernel(d['K'][U],d['y'][U],d['K'][Q],d['Kblocks'],choice['bandwidth'],choice['alpha']);err['kernel']=float(abs(pred-p['KRR']).max());assert err['kernel']<1e-8
 obj=joblib.load(_repo_io(path/'kernel.joblib'));np.testing.assert_array_equal(obj['train'],U);np.testing.assert_array_equal(obj['query'],Q);np.testing.assert_array_equal(obj['model'].predict(d['K'][Q]),p['KRR'])
 for seed in t.CNN_SEEDS:
  obj=torch.load(_repo_io(path/f'CNN-{seed}.pt'),map_location='cpu',weights_only=True);assert obj['seed']==seed+r['offset'];assert obj['train_ids']==d['ids'][U].tolist() and obj['validation_ids']==d['ids'][Q].tolist();assert obj['normalization']=={'x_mean':float(d['images'][U].mean()),'x_std':float(d['images'][U].std()),'y_mean':float(d['y'][U].mean()),'y_std':float(d['y'][U].std())}
  pm=cnn_reload(obj,d['images'],Q,'mps');pc=cnn_reload(obj,d['images'],Q,'cpu');err['MPS_reload']=max(err['MPS_reload'],float(abs(pm-p[f'CNN_{seed}']).max()));err['CPU_MPS']=max(err['CPU_MPS'],float(abs(pm-pc).max()))
 for arm in ['S','B']:
  X=d[arm]
  for seed in t.HEAD_SEEDS:
   obj=torch.load(_repo_io(path/f'{arm}-{seed}.pt'),map_location='cpu',weights_only=True);np.testing.assert_array_equal(obj['train'],U);assert obj['seed']==seed+r['offset'];mu=X[U].mean(0);sd=X[U].std(0);np.testing.assert_allclose(obj['scaler_mean'].numpy(),mu,atol=1e-12,rtol=0);np.testing.assert_allclose(obj['scaler_scale'].numpy(),sd,atol=1e-12,rtol=0);assert obj['target_mean']==float(d['y'][U].mean()) and obj['target_std']==float(d['y'][U].std());m=h.Head(X.shape[1]);m.load_state_dict(obj['model']);pp=h.predict(m,(X[Q]-mu)/sd,obj['target_mean'],obj['target_std']);err['CPU_head']=max(err['CPU_head'],float(abs(pp-p[f'{arm}_{seed}']).max()));assert h.state_hash(obj['model'])==r['logs'][f'{arm}_{seed}']['final_state_sha256']
 if r['with_short']:
  obj=torch.load(_repo_io(path/'short.pt'),map_location='cpu',weights_only=True);np.testing.assert_array_equal(obj['train'],U);assert obj['seed']==20270215+r['offset'];mu=d['B'][U].mean(0);sd=d['B'][U].std(0);np.testing.assert_allclose(obj['scaler_mean'].numpy(),mu,atol=1e-12,rtol=0);np.testing.assert_allclose(obj['scaler_scale'].numpy(),sd,atol=1e-12,rtol=0)
  for device in ['mps','cpu']:
   m=Short(mu,sd,obj['seed']).to(device);m.load_state_dict(obj['state']);pp=t.short_predict(m,d['B'],Q,device)*obj['target_std']+obj['target_mean'];key='MPS_reload' if device=='mps' else 'CPU_MPS';err[key]=max(err[key],float(abs(pp-p['short']).max()));del m
  assert h.state_hash(obj['state'])==r['logs']['short']['final_state']
 assert err['CPU_head']<1e-8 and err['MPS_reload']<1e-8 and err['CPU_MPS']<1e-3,err
 for key,seeds in [('CNN',t.CNN_SEEDS),('S',t.HEAD_SEEDS),('B',t.HEAD_SEEDS)]:np.testing.assert_array_equal(np.mean([p[f'{key}_{s}'] for s in seeds],0),p[key])
 np.testing.assert_array_equal(p['P'],np.column_stack([np.zeros(len(Q)),p['KRR'],p['CNN'],p['S'],p['B']]))
 torch.mps.empty_cache();return r,p,err

def ols(P,y,Q):
 w=np.array([0.,.3,.3,.2,.2]);coef=np.linalg.lstsq(np.c_[np.ones(len(P)),P@w],y,rcond=None)[0];return np.c_[np.ones(len(Q)),Q@w]@coef

def validate():
 k=0
 t.check();d=t.input_data();root=OUT/'full';rr=t.read(root/'report.json');T=np.arange(500);V=np.arange(500);start=time.perf_counter();maxerr={};nodes={}
 for name in ['outer']+[name for j in range(3) for name in [f'middle-{j}']+[f'deep-{j}-{ell}' for ell in range(3)]]:
  r,p,err=verify_node(d,root/name,T,np.array([],dtype=int));nodes[name]=(r,p)
  for key,er in err.items():maxerr[key]=max(maxerr.get(key,0),er)
 with np.load(_repo_io(root/'predictions.npz')) as z:z={key:z[key].copy() for key in z.files}
 assert t.sha(root/'predictions.npz')==rr['prediction_sha256'];np.testing.assert_array_equal(z['T'],T);np.testing.assert_array_equal(z['V'],V);np.testing.assert_array_equal(z['outer'],nodes['outer'][1]['P'])
 middle=np.empty((len(T),5));splits=list(GroupKFold(3,shuffle=True,random_state=20270110+k).split(d['S'][T],d['y'][T],d['g'][T]));cor=[];count=0
 for j,(I,H) in enumerate(splits):
  A,B=T[I],T[H];r,mid=nodes[f'middle-{j}'];np.testing.assert_array_equal(r['U'],A);np.testing.assert_array_equal(r['Q'],B);middle[H]=mid['P'];deep=np.empty((len(A),5));inn=list(GroupKFold(3,shuffle=True,random_state=20270110+k+100*(j+1)).split(d['S'][A],d['y'][A],d['g'][A]))
  for ell,(C,D) in enumerate(inn):
   dr,dp=nodes[f'deep-{j}-{ell}'];np.testing.assert_array_equal(dr['U'],A[C]);np.testing.assert_array_equal(dr['Q'],A[D]);deep[D]=dp['P'];assert not (set(A[C])|set(A[D]))&set(B)
  anchor=.9*ols(deep,d['y'][A],mid['P'])+.1*mid['short'];path=root/f'residual-{j}';br=t.read(path/'report.json')
  for f,hsh in br['files'].items():assert t.sha(path/f)==hsh
  with np.load(_repo_io(path/'predictions.npz')) as bz:bz={key:bz[key].copy() for key in bz.files}
  np.testing.assert_array_equal(B,bz['B']);np.testing.assert_array_equal(V,bz['V']);np.testing.assert_allclose(anchor,bz['anchor'],atol=1e-8,rtol=0);res=d['y'][B]-bz['anchor'];np.testing.assert_array_equal(res,bz['residual']);ip=np.empty_like(bz['inner']);inner=list(GroupKFold(3,shuffle=True,random_state=20271013+100*k+j).split(d['X'][B],res,d['g'][B]));maxerr['residual']=maxerr.get('residual',0)
  for ell,(I,H) in enumerate(inner):
   assert br['splits'][ell]=={'train':B[I].tolist(),'validation':B[H].tolist()}
   for ci,kind in enumerate(t.KINDS):
    obj=joblib.load(_repo_io(path/f'inner-{ell}-{kind}.joblib'));np.testing.assert_array_equal(obj['train'],B[I]);np.testing.assert_array_equal(obj['validation'],B[H]);saved=t.n.predict(obj['model'],d['X'][B[H]]);np.testing.assert_array_equal(saved,bz['inner'][ci,H]);pred=residual(d['X'][B[I]],res[I],d['X'][B[H]],d['blocks'],kind);maxerr['residual']=max(maxerr['residual'],float(abs(pred-saved).max()));ip[ci,H]=saved;count+=1
  unique,inverse=np.unique(d['g'][B],return_inverse=True);draw=np.random.default_rng(20271204+100*k+j).integers(0,len(unique),(512,len(unique)));np.testing.assert_array_equal(draw,bz['draws']);loss=(ip-res)**2;tot=[]
  for sample in draw:tot.append(sum(loss[:,inverse==gi].sum(1) for gi in sample))
  wins=np.argmin(np.array(tot),axis=1);w=np.array([(wins==ci).mean() for ci in range(5)]);np.testing.assert_array_equal(w,bz['weights']);np.testing.assert_array_equal(wins,bz['wins']);full=[]
  for ci,kind in enumerate(t.KINDS):
   obj=joblib.load(_repo_io(path/f'full-{kind}.joblib'));np.testing.assert_array_equal(obj['train'],B);saved=t.n.predict(obj['model'],d['X'][V]);np.testing.assert_array_equal(saved,bz['full'][ci]);pred=residual(d['X'][B],res,d['X'][V],d['blocks'],kind);maxerr['residual']=max(maxerr['residual'],float(abs(pred-saved).max()));full.append(saved);count+=1
  pp=sum(wi*pi for wi,pi in zip(w,full));np.testing.assert_allclose(pp,bz['primary'],atol=1e-12,rtol=0);cor.append(pp)
 np.testing.assert_array_equal(middle,z['middle']);base=.9*ols(middle,d['y'][T],z['outer'])+.1*z['short'];maxerr['affine']=float(abs(base-z['base']).max());assert maxerr['affine']<1e-8 and maxerr['residual']<1e-8;np.testing.assert_allclose(base+np.mean(cor,0),z['primary'],atol=1e-8,rtol=0)
 rec={'passed':True,'rows':500,'root_in_sample_queries_not_validation':True,'nodes':13,'KRR_independent_fits':364,'residual_objects':count,'affine_fits':4,'neural_reloads':121,'errors':maxerr,'seconds':time.perf_counter()-start,'training_report_sha256':t.sha(root/'report.json'),'validator_sha256':t.sha(Path(__file__))};t.write(OUT/'training-validation.json',rec);t.emit(rec)
if __name__=='__main__':validate()
