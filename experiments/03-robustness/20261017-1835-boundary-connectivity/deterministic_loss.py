"""Verification prototype: aggregate duplicate maximin edges before GPU backward.

Mathematically the same per-image class-balanced pair BCE. This implementation
was NOT used for the completed nested experiment; its scores must not be reused
as validation of this prototype.
"""
import numpy as np
import torch
from torch.nn import functional as F
import graph
def weights(critical,targets,confidence,n_edges):
    positive=np.zeros((len(critical),n_edges),np.float64);negative=np.zeros_like(positive)
    for i,(index,y,w) in enumerate(zip(critical,targets,confidence)):
        totals=[float(w[y==label].astype(float).sum()) for label in [0,1]];classes=sum(v>0 for v in totals)
        if not classes:continue
        for label,total in enumerate(totals):
            if total<=0:continue
            select=y==label;dest=positive if label else negative;np.add.at(dest[i],index[select],w[select].astype(float)/total/classes)
    return positive,negative
def from_critical(logits,critical,targets,confidence):
    p,n=weights(critical,targets,confidence,logits.shape[1]);p=torch.as_tensor(p,dtype=logits.dtype,device=logits.device);n=torch.as_tensor(n,dtype=logits.dtype,device=logits.device)
    return (F.softplus(-logits)*p+F.softplus(logits)*n).sum(1)
def connection_loss(logits,pairs,targets,confidence):
    values=logits.detach().cpu().numpy();critical=np.stack([graph.critical_edges(x,pair) for x,pair in zip(values,pairs)]);return from_critical(logits,critical,targets,confidence),critical
