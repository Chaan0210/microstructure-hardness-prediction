import math
import common as c
from common import torch,np
F=torch.nn.functional

def mixing(block,q,source):
 B,L,_=q.shape;N=source.shape[1];qi=F.layer_norm(q,(32,),weight=None,bias=None,eps=1e-5) if getattr(block,'query_normalize',False) else q;Q=block.shared(qi).reshape(B,L,4,8).transpose(1,2);K=block.shared(source).reshape(B,N,4,8).transpose(1,2);V=K.clone();logits=Q@K.transpose(-2,-1)/math.sqrt(8);w=logits.softmax(-1);context=(w@V).transpose(1,2).reshape(B,L,32);return context,{'logits':logits,'values':V,'Q':Q,'K':K,'weights':w,'q_input':qi}

def forward(m,x):
 B=len(x);H=m.token_norm(F.gelu(m.project(torch.cat(((x-m.mean)/m.scale,m.coords.expand(B,-1,-1)),2))));raw=m.queries.expand(B,-1,-1);h,cross=mixing(m.cross,raw,H);common=m.cross.out(h);q1=raw+common;q2=q1+m.cross.ff(q1);z=m.self_norm(q2);h2,selfinfo=mixing(m.self_mixer,z,z);q3=q2+m.self_mixer.out(h2);q4=q3+m.self_mixer.ff(q3);slots=m.final_norm(q4);p=m.head(slots.mean(1)).squeeze(-1);return p,{'cross':cross,'self':selfinfo,'raw_query':raw,'cross_residual':q1,'cross_output':common,'slots':slots,'cross_context':h}

def cosine(a):
 b=F.normalize(a,dim=-1);cc=b@b.transpose(-2,-1);mask=~torch.eye(4,dtype=torch.bool);return cc[:,mask].detach().cpu().numpy().ravel()

def probe(m,reader,indices,norm,y):
 assert len(indices)==16;m.eval();before=c.statehash(m.state_dict());acc={};bag={};err=0.;sdp=0.;decomp=0.;loss_sum=0.
 for off in range(0,16,2):
  ii=indices[off:off+2];x=torch.from_numpy(reader.read(ii));p,info=forward(m,x);err=max(err,float(abs(p.detach()-m(x,'mha').detach()).max()));target=torch.tensor((y[ii]-norm['target_mean'])/norm['target_std'],dtype=torch.float32);loss=F.smooth_l1_loss(p,target,beta=1)/8;loss_sum+=float(loss.detach())
  def add(key,value):acc[key]=acc.get(key,torch.zeros_like(value))+value.detach()
  def values(key,value):bag.setdefault(key,[]).extend(np.asarray(value).ravel().tolist())
  add('query_total',torch.autograd.grad(loss,m.queries,retain_graph=True)[0])
  for name,block in [('cross',m.cross),('self',m.self_mixer)]:
   z=info[name];gs,gv=torch.autograd.grad(loss,[z['logits'],z['values']],retain_graph=True);score=torch.autograd.grad(z['logits'],block.shared.weight,grad_outputs=gs.detach(),retain_graph=True)[0];value=torch.autograd.grad(z['values'],block.shared.weight,grad_outputs=gv.detach(),retain_graph=True)[0];total=torch.autograd.grad(loss,block.shared.weight,retain_graph=True)[0];decomp=max(decomp,float(abs(score+value-total).max()));add(name+'_score',score);add(name+'_value',value);add(name+'_total',total)
   if name=='cross':add('query_cross_score',torch.autograd.grad(z['logits'],m.queries,grad_outputs=gs.detach(),retain_graph=True)[0])
   with torch.no_grad():
    logit=z['logits'];w=z['weights'];values(name+'_logit_std',(logit-logit.mean(-1,keepdim=True)).square().mean(-1).sqrt().numpy());values(name+'_Q_RMS',z['Q'].square().mean(-1).sqrt().numpy());values(name+'_KV_RMS',z['K'].square().mean(-1).sqrt().numpy());values(name+'_ESS',(1/w.square().sum(-1)).numpy());values(name+'_maxweight',w.max(-1).values.numpy())
    if off==0:
     Q=z['Q'].double();K=z['K'].double();V=z['values'].double();a=(Q@K.transpose(-2,-1)/math.sqrt(8)).softmax(-1)@V;b=F.scaled_dot_product_attention(Q,K,V,dropout_p=0,is_causal=False);sdp=max(sdp,float(abs(a-b).max()))
  with torch.no_grad():
   for name in ['raw_query','cross_residual','slots']:values(name+'_cosine',cosine(info[name]))
   values('common_cross_over_raw_query_norm',(info['cross_output'].norm(dim=-1)/info['raw_query'].norm(dim=-1)).numpy());K=info['cross']['K'];uni=K.mean(2).unsqueeze(2).expand(len(x),4,4,8).transpose(1,2).reshape(len(x),4,32);ctx=info['cross_context'];values('cross_context_nonuniform_fraction',((ctx-uni).norm(dim=-1)/ctx.norm(dim=-1).clamp_min(1e-12)).numpy())
 assert before==c.statehash(m.state_dict()) and all(p.grad is None for p in m.parameters());assert err==0 and sdp<1e-10
 r={'probe_indices':indices.tolist(),'eval_train16_standardized_SmoothL1':loss_sum,'manual_forward_maxabs':err,'double_SDPA_reference_maxabs':sdp,'gradient_decomposition_maxabs':decomp,'median':{n:float(np.median(v)) for n,v in bag.items()},'query_cross_score_over_total_gradient_norm':float(acc['query_cross_score'].norm()/acc['query_total'].norm()),'query_total_gradient_norm':float(acc['query_total'].norm()),'query_cross_score_gradient_norm':float(acc['query_cross_score'].norm()),'no_optimizer_updates':True,'model_state_unchanged':True}
 for name in ['cross','self']:r[name+'_gradient']={'score_norm':float(acc[name+'_score'].norm()),'value_norm':float(acc[name+'_value'].norm()),'total_norm':float(acc[name+'_total'].norm()),'score_over_value_norm':float(acc[name+'_score'].norm()/acc[name+'_value'].norm())}
 return r
