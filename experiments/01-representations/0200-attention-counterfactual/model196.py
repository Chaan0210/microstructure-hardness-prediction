import math
import common as c
from common import np,torch,R
nn=torch.nn;F=nn.functional;cfg=c.plan["recipe"]
class Mixer(nn.Module):
 def __init__(self):
  super().__init__();self.shared=nn.Linear(32,32,bias=False);self.out=nn.Linear(32,32);self.drop=nn.Dropout(.25);self.ff=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,64),nn.GELU(),nn.Dropout(.25),nn.Linear(64,32),nn.Dropout(.25))
 def mix(self,q,source,arm):
  B,L,_=q.shape;N=source.shape[1];Q=self.shared(q).reshape(B,L,4,8).transpose(1,2);KV=self.shared(source).reshape(B,N,4,8).transpose(1,2)
  if arm=='mha':weights=torch.softmax(Q@KV.transpose(-2,-1)/math.sqrt(8),dim=-1);context=weights@KV
  else:assert arm=='mean';weights=torch.full((B,4,L,N),1/N,device=q.device,dtype=q.dtype);context=KV.mean(2,keepdim=True).expand(B,4,L,8)
  return context.transpose(1,2).reshape(B,L,32),weights,Q,KV
 def forward(self,q,source,arm):
  h,w,_,_=self.mix(q,source,arm);q=q+self.drop(self.out(h));return q+self.ff(q),w

class Model(nn.Module):
 def __init__(self,norm):
  super().__init__();self.project=nn.Linear(770,32);self.token_norm=nn.LayerNorm(32);self.queries=nn.Parameter(torch.empty(4,32));nn.init.normal_(self.queries,std=.02);self.cross=Mixer();self.self_norm=nn.LayerNorm(32);self.self_mixer=Mixer();self.final_norm=nn.LayerNorm(32);self.head=nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(.25),nn.Linear(16,1));self.register_buffer('mean',torch.tensor(norm['mean'],dtype=torch.float32));self.register_buffer('scale',torch.tensor(norm['scale'],dtype=torch.float32));yy,xx=np.meshgrid(np.linspace(-1,1,36,dtype=np.float32),np.linspace(-1,1,36,dtype=np.float32),indexing='ij');self.register_buffer('coords',torch.from_numpy(np.stack((xx.ravel(),yy.ravel()),1)))
 def forward(self,x,arm,diagnostics=False,coords=None):
  pos=self.coords if coords is None else coords;z=torch.cat(((x-self.mean)/self.scale,pos.expand(len(x),-1,-1)),2);H=self.token_norm(F.gelu(self.project(z)));q=self.queries.expand(len(x),-1,-1);q,cross=self.cross(q,H,arm);z=self.self_norm(q);h,w,_,_=self.self_mixer.mix(z,z,arm);q=q+self.self_mixer.drop(self.self_mixer.out(h));q=q+self.self_mixer.ff(q);slots=self.final_norm(q);p=self.head(slots.mean(1)).squeeze(-1)
  if not diagnostics:return p
  return p,{'cross_weights':cross,'self_weights':w,'slots':slots}

def initialize(norm,k,device):
 torch.manual_seed(cfg['seed']+k);m=Model(norm).to(device);assert sum(p.numel() for p in m.parameters())==38209;return m

