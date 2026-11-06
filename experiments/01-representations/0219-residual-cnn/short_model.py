import torch
from torch import nn

class ShortHead(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(6912,32),nn.GELU(),nn.Dropout(.5),nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)

class Short(nn.Module):
 def __init__(self,mean,scale,seed=20270215):
  super().__init__();torch.manual_seed(seed);self.head=ShortHead();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
 def forward(self,x):return self.head((x-self.mean)/self.scale)
