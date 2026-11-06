from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
import common as c
from common import R,P,B,np,torch,nn
import importlib.util,math,hashlib
from sklearn.decomposition import PCA
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('base185',B/'train.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def floor(X,T):
 mu=X[T].mean(0);var=X[T].var(0);r=float(np.sqrt(var.mean()));scale=np.maximum(np.sqrt(var),r) if r>0 else np.ones_like(var)
 return mu,var,scale,r

def raw_features(name,arm):
 root=R/name;assert c.sha(root/'features.npz')==c.read(root/'extraction.json')['features_SHA256'];f=c.load(root/'features.npz');assert f['ids'].tolist()==c.plan['IDs'];return np.concatenate([f['mean'],f['std']],1).astype(float) if arm=='mean_std' else f['mean'].astype(float)

def fit_pre(X,T,bankwidth):
 state={};scaled=[]
 for j,start in enumerate(range(0,X.shape[1],bankwidth)):
  block=X[:,start:start+bankwidth];mu,var,scale,r=floor(block,T)
  for key,value in [('mean',mu),('var',var),('scale',scale),('RMS',np.array(r))]:state[f'bank{j}_{key}']=value
  scaled.append((block-mu)/scale)
 A=np.concatenate(scaled,1);pca=PCA(n_components=128,svd_solver='full',whiten=False);pca.fit(A[T]);state.update(pca_mean=pca.mean_,pca_components=pca.components_,pca_explained_variance=pca.explained_variance_,pca_explained_variance_ratio=pca.explained_variance_ratio_,pca_singular_values=pca.singular_values_,bankwidth=np.array(bankwidth),bankcount=np.array(len(scaled)),train=np.asarray(T));Z=(A-pca.mean_)@pca.components_.T
 assert Z.shape==(400,128) and np.isfinite(Z).all();return Z,state

def transform(X,s):
 width=int(s['bankwidth']);A=np.concatenate([(X[:,j*width:(j+1)*width]-s[f'bank{j}_mean'])/s[f'bank{j}_scale'] for j in range(int(s['bankcount']))],1);return (A-s['pca_mean'])@s['pca_components'].T

def stats(X,y,T,V,seed):
 mu,var,scale,r=floor(X,T)
 return {'feature_mean':torch.tensor(mu),'feature_var':torch.tensor(var),'feature_scale':torch.tensor(scale),'feature_RMS_floor':r,'scaler_policy':'train_RMS_floor','target_mean':float(y[T].mean()),'target_std':float(y[T].std()),'train':T.tolist(),'validation':V.tolist(),'seed':seed,'dim':X.shape[1]}
old.stats=stats

def fit_head(X,y,T,V,seed):
 o=stats(X,y,T,V,seed);torch.manual_seed(seed);h=old.head(128);o['initial_head_sha256']=c.statehash(h);mu,scale=old.tensors(o,'cpu');xx=(torch.tensor(X[T],dtype=torch.float32)-mu)/scale;yy=torch.tensor((y[T]-o['target_mean'])/o['target_std'],dtype=torch.float32);assert torch.isfinite(xx).all() and torch.isfinite(yy).all()
 opt=torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.01);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100,eta_min=.00001);rng=np.random.default_rng(seed+1);orders=[];curve=[];updates=[];torch.manual_seed(seed+1000)
 for epoch in range(100):
  h.train();order=rng.permutation(len(T));orders.append(order);losses=[]
  for j in range(0,len(T),64):
   I=order[j:j+64];opt.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(h(xx[I]).flatten(),yy[I]);assert torch.isfinite(loss);loss.backward();gn=float(nn.utils.clip_grad_norm_(h.parameters(),1.));assert math.isfinite(gn) and gn>0 and gn<1e6;updates.append([epoch,j//64,len(I),opt.param_groups[0]['lr'],float(loss.detach()),gn]);opt.step();losses.append(float(loss.detach()))
  schedule.step()
  if (epoch+1)%20==0:curve.append({'epoch':epoch+1,'train_loss':float(np.mean(losses))})
 o['head']={k:v.detach().cpu().clone() for k,v in h.state_dict().items()};o['curve']=curve;o['order_sha256']=hashlib.sha256(np.stack(orders).tobytes()).hexdigest();o['final_head_sha256']=c.statehash(h);o['parameter_count']=sum(p.numel() for p in h.parameters());assert o['parameter_count']==8321 and len(updates)==500
 return h.eval(),o,np.array(updates)

def predict(h,Z,V,o,device='cpu'):return old.predict(h,None,None,Z,V,o,device)
def ck(p):return torch.load(_repo_io(p),map_location='cpu',weights_only=True)
def diagnose(X,Z,s,o,T,V):
 mu=o['feature_mean'].numpy();scale=o['feature_scale'].numpy();W=(Z-mu)/scale;trainrms=float(np.sqrt(np.mean(W[T]**2)));ratio=float(np.sqrt(np.mean(W[V]**2))/max(1,trainrms));maxratio=float(abs(W[V]).max()/max(1,float(abs(W[T]).max())));assert np.isfinite(W).all() and ratio<=100 and maxratio<=100
 return {'PCA_explained_variance_ratio_sum':float(s['pca_explained_variance_ratio'].sum()),'PCA_last_component_variance':float(s['pca_explained_variance'][-1]),'PCA128':True,'head_input_maxabs_train':float(abs(W[T]).max()),'head_input_maxabs_validation':float(abs(W[V]).max()),'validation_RMS_ratio':ratio,'validation_maxabs_ratio':maxratio,'post_PCA_RMS_floor':o['feature_RMS_floor'],'raw_bank_RMS_floors':[float(s[f'bank{j}_RMS']) for j in range(int(s['bankcount']))],'raw_bank_minstd':[float(np.sqrt(s[f'bank{j}_var']).min()) for j in range(int(s['bankcount']))]}
