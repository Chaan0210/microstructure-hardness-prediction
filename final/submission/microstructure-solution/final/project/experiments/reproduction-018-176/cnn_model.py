import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__();layers=[];inc=1
        for i,out in enumerate((16,32,64,96)):
            layers.extend([nn.Conv2d(inc,out,5 if i==0 else 3,stride=2,padding=2 if i==0 else 1),nn.GroupNorm(4,out),nn.SiLU()]);inc=out
        self.features=nn.Sequential(*layers);self.head=nn.Sequential(nn.Linear(194,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1))
    def forward(self,x):
        h=self.features(x)
        stats=torch.cat([h.mean((2,3)),(h.var((2,3),correction=0)+1e-6).sqrt(),x.mean((2,3)),(x.var((2,3),correction=0)+1e-6).sqrt()],1)
        return self.head(stats).squeeze(1)
