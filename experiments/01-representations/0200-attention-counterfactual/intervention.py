import copy,math
import common as c
from common import torch,np
F=torch.nn.functional
@torch.no_grad()
def call(m,x,uniform=False):
 original=m.cross.mix
 if uniform:m.cross.mix=lambda q,source,arm:original(q,source,'mean')
 try:return m(x,'mha',True)
 finally:m.cross.mix=original

def attention(z):
 raw=z['cross_weights'].detach().cpu().numpy().astype(float);assert raw.shape[1:]==(4,4,1296);err=float(abs(raw.sum(-1)-1).max());assert err<1e-5;w=raw/raw.sum(-1,keepdims=True);H=-(w*np.log(np.maximum(w,1e-300))).sum(-1);s=z['self_weights'].detach().cpu().numpy().astype(float);ss=s/s.sum(-1,keepdims=True);slots=z['slots'].detach().cpu().numpy().astype(float);slots/=np.maximum(np.linalg.norm(slots,axis=-1,keepdims=True),1e-30);cos=slots@slots.transpose(0,2,1);mask=~np.eye(4,dtype=bool)
 return {'cross_ESS':1/(w*w).sum(-1),'cross_entropy':H,'cross_KL_uniform':np.log(1296)-H,'cross_TV_uniform':.5*abs(w-1/1296).sum(-1),'cross_maxweight':w.max(-1),'self_weights':ss,'self_ESS':1/(ss*ss).sum(-1),'self_offdiagonal_mass':1-np.diagonal(ss,axis1=-2,axis2=-1),'slot_cosine':cos[:,mask]},err
@torch.no_grad()
def independent_double(m,x,uniform):
 m=copy.deepcopy(m).cpu().double();x=x.cpu().double();H=F.layer_norm(F.gelu(F.linear(torch.cat(((x-m.mean)/m.scale,m.coords.expand(len(x),-1,-1)),2),m.project.weight,m.project.bias)),(32,),m.token_norm.weight,m.token_norm.bias,m.token_norm.eps);q=m.queries.expand(len(x),-1,-1)
 qi=F.layer_norm(q,(32,),weight=None,bias=None,eps=1e-5) if getattr(m.cross,'query_normalize',False) else q;Q=F.linear(qi,m.cross.shared.weight).reshape(len(x),4,4,8).transpose(1,2);KV=F.linear(H,m.cross.shared.weight).reshape(len(x),1296,4,8).transpose(1,2)
 if uniform:
  weights=torch.full((len(x),4,4,1296),1/1296,dtype=torch.float64);ctx=weights@KV
 else:ctx=F.scaled_dot_product_attention(Q,KV,KV,dropout_p=0,is_causal=False)
 q=q+F.linear(ctx.transpose(1,2).reshape(len(x),4,32),m.cross.out.weight,m.cross.out.bias);q=q+m.cross.ff(q);z=m.self_norm(q);KV=F.linear(z,m.self_mixer.shared.weight).reshape(len(x),4,4,8).transpose(1,2);ctx=F.scaled_dot_product_attention(KV,KV,KV,dropout_p=0,is_causal=False);q=q+F.linear(ctx.transpose(1,2).reshape(len(x),4,32),m.self_mixer.out.weight,m.self_mixer.out.bias);q=q+m.self_mixer.ff(q);p=m.head(m.final_norm(q).mean(1)).squeeze(-1);direct,_=call(m,x,uniform);err=float(abs(p-direct).max());assert err<1e-10;return p.cpu().numpy(),err
