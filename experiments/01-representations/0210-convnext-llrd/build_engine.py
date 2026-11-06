from pathlib import Path as _RepoPath
import sys as _repo_sys
_repo_sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[3]))
from research_paths import resolve as _repo_path, io_path as _repo_io, study_sources as _study_sources, STUDIES as _STUDY_ROOT
from pathlib import Path
R=Path(__file__).resolve().parent;C=_repo_path(R.parent, 'modern-backbones-190');s=(_repo_path(C, 'engine.py')).read_text()
s=s.replace('RECIPES=[]','RECIPES=[]\nCURRENT_MODEL="convnext_tiny"')
start=s.index('def decay_unit():');end=s.index('def train(',start)
s=s[:start]+'''def rng_reference(seed):
 # No model fitting: reproduce only the known head-dropout RNG consumption.
 cpu=torch.get_rng_state().clone();mps=torch.mps.get_rng_state().clone();torch.manual_seed(seed+1000);rows=[];drop=nn.Dropout(.25).to('mps');drop.train();x=torch.ones((8,64),device='mps',requires_grad=True)
 for step in range(240):
  rows.append(c.rnghash());a=drop(x);b=drop(x);del a,b
 end=c.rnghash();torch.set_rng_state(cpu);torch.mps.set_rng_state(mps);return rows,end

def parameter_groups(h,tail,ratio):
 hp=list(h.parameters());tp=list(tail.parameters());parts=[hp,*[list(tail[0][i].parameters()) for i in range(3)],list(tail[2].parameters())]
 assert [id(p) for part in parts for p in part]==[id(p) for p in hp+tp];assert len(set(id(p) for p in hp+tp))==len(hp+tp)
 counts=[sum(p.numel() for p in part) for part in parts];assert counts[1:]==c.plan['tail_group_counts'];assert all(p.requires_grad for p in hp+tp)
 raw=[ratio**2,ratio,1.];den=sum(raw)/3;mult=[r/den for r in raw];lrs=[1e-4,*[1e-5*m for m in mult],1e-5];assert abs(sum(n*l for n,l in zip(counts[1:],lrs[1:]))/sum(counts[1:])-1e-5)<1e-19
 groups=[{'params':part,'lr':lr} for part,lr in zip(parts,lrs)];names=['head.'+n for n,_ in h.named_parameters()]+['tail.'+n for n,_ in tail.named_parameters()];mapping=[];offset=0
 for label,part,lr,reference in zip(c.plan['group_names'],parts,lrs,c.plan['decay_reference_LRs']):
  mapping.append({'name':label,'parameter_count':sum(p.numel() for p in part),'tensor_count':len(part),'parameters':names[offset:offset+len(part)],'initial_lr':lr,'decay_reference_initial_lr':reference});offset+=len(part)
 return groups,mapping

def decay_unit():
 # Equal effective shrinkage is independent of the gradient learning-rate groups.
 values=torch.tensor([1.,-2.,3.],device='mps');a=values.clone();b=values.clone();trace=[]
 for epoch in range(12):
  factor=.1+.9*(1+math.cos(math.pi*epoch/12))/2;decay=1-1e-5*factor*.01
  for step in range(20):a.mul_(decay);b.mul_(decay)
  assert torch.equal(a,b);trace.append({'epoch':epoch,'decay_factor':decay})
 return {'equal_shrinkage_240_steps':True,'trace':trace,'actual_loop_uses_reference_LR_not_group_update_LR':True}

'''+s[end:]
s=s.replace('def train(o,cache,X,y_train,recipe,seed,out,predict_indices,train_tail=True):','def train(o,cache,X,y_train,recipe,seed,out,predict_indices,ratio,fold):\n train_tail=True;assert ratio in [1.,c.plan["ratio"]]')
a="groups=[{'params':hp,'lr':recipe['head_lr']}]+([{'params':tp,'lr':recipe['tail_lr']}] if train_tail else [])"
s=s.replace(a,"groups,mapping=parameter_groups(h,tail,ratio);c.write(out/'parameter-groups.json',mapping);assert len(groups)==5;assert sum(p.numel() for p in tp)==14290944;assert len(list(tail.named_buffers()))==0 and not any(isinstance(mod,nn.modules.batchnorm._BatchNorm) for mod in tail.modules())")
s=s.replace("rng=np.random.default_rng(seed+1);orders=[];updates=[];curve=[];", "reference_rng,reference_end=rng_reference(seed);rng=np.random.default_rng(seed+1);orders=[];updates=[];curve=[];rnglog=[];lrlog=[];decaylog=[];inputlog=[];counter=0;oldfolder=c.C/'convnext_tiny'/f'fold-{fold}'/'last_stage112';oldrec=c.read(oldfolder/'report.json');oldtrace=c.load(oldfolder/'updates.npz')['values'];")
s=s.replace("metric(0);torch.manual_seed(seed+1000)","metric(0);assert curve[0]['train_RMSE']==oldrec['initial_train_RMSE'];torch.manual_seed(seed+1000);c.emit({'started_fold':fold,'ratio':ratio,'group_counts':[z['parameter_count'] for z in mapping]})")
s=s.replace("inds=order[j:j+16];opt.zero_grad", "inds=order[j:j+16];state=c.rnghash();assert state==reference_rng[counter];rnglog.append(state);inputlog.append(inds.copy());opt.zero_grad")
s=s.replace("lrs=[g['lr'] for g in opt.param_groups]+([] if train_tail else [0.])", "lrs=[g['lr'] for g in opt.param_groups];factor=.1+.9*(1+math.cos(math.pi*epoch/12))/2;expected=[m['initial_lr']*factor for m in mapping];np.testing.assert_array_equal(lrs,expected);assert all(g['weight_decay']==0 for g in opt.param_groups);assert abs(sum(m['parameter_count']*l for m,l in zip(mapping[1:],lrs[1:]))/14290944-1e-5*factor)<1e-19;decays=[1-ref*factor*.01 for ref in c.plan['decay_reference_LRs']];assert decays[1:]==[1-1e-5*factor*.01]*4;lrlog.append(lrs);decaylog.append(decays);before_optimizer_rng=c.rnghash()")
s=s.replace("for g in opt.param_groups:\n     for param in g['params']:param.mul_(1-g['lr']*.01)", "for g,decay in zip(opt.param_groups,decays):\n     for param in g['params']:param.mul_(decay)")
s=s.replace("opt.step();updates.append([epoch,j//16,len(inds),*lrs,grad,*norms])", "opt.step();assert c.rnghash()==before_optimizer_rng;row=[epoch,j//16,len(inds),lrs[0],1e-5*factor,grad,*norms];updates.append(row)\n   np.testing.assert_array_equal(row[:5],oldtrace[counter,:5])\n   if ratio==1.:np.testing.assert_array_equal(row,oldtrace[counter])\n   counter+=1")
s=s.replace("if (epoch+1)%3==0:metric(epoch+1);curve[-1]['train_augmented_mean_loss']=float(np.mean(losses))", "if (epoch+1)%3==0:\n   metric(epoch+1);curve[-1]['train_augmented_mean_loss']=float(np.mean(losses));c.emit({'fold':fold,'ratio':ratio,'epoch':epoch+1,'seconds':time.monotonic()-started})\n  assert time.monotonic()-started<120")
s=s.replace("assert len(updates)==12*math.ceil(len(T)/16);", "assert counter==240 and c.rnghash()==reference_end;assert len(updates)==12*math.ceil(len(T)/16);")
s=s.replace("guard.phase='validation';pred=", "assert hashlib.sha256(np.stack(orders).tobytes()).hexdigest()==oldrec['order_sha256']\n guard.phase='validation';pred=")
s=s.replace("'optimizer_reset_from_warmhead':True}", "'optimizer_reset_from_warmhead':True,'ratio':ratio,'parameter_groups':mapping,'RNG_reference_exact_all240':True}")
s=s.replace("c.arrays(out/'predictions.npz',indices=predict_indices,prediction=pred);", "c.arrays(out/'rng-and-groups.npz',order=np.stack(inputlog),group_LR=np.array(lrlog),decay_factor=np.array(decaylog),rng_CPU=np.array([q['CPU'] for q in rnglog]),rng_MPS=np.array([q['MPS'] for q in rnglog]));c.arrays(out/'predictions.npz',indices=predict_indices,prediction=pred);")
s=s.replace("rec={'recipe':recipe", "rec={'ratio':ratio,'fold':fold,'RNG_reference_exact_all240':True,'group_LR_mean_and_schedule_exact_all240':True,'effective_decay_fixed_all240':True,'optimizer_has_no_RNG_side_effect_all240':True,'recipe':recipe")
s=s.replace("c.write(out/'report.json',rec);c.emit", "\n if ratio==1.:\n  reference=ck(oldfolder/'model.pt')\n  for name in ['head','tail']:\n   for n,value in obj[name].items():assert torch.equal(value,reference[name][n]),(name,n)\n  np.testing.assert_array_equal(pred,c.load(oldfolder/'predictions.npz')['prediction']);assert curve==oldrec['curve'];assert c.read(out/'stability.json')==c.read(oldfolder/'stability.json');rec['existing190_control_exact']=True\n c.write(out/'report.json',rec);c.emit")
(_repo_path(R, 'engine.py')).write_text(s)
for p in R.glob('*.py'):compile(p.read_text(),str(p),'exec')
print('Engine syntax valid; historical forward/loss/prediction unchanged')
