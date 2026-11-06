"""Pure-stdlib evaluation contract primitives; no data IO or ML training."""
from dataclasses import dataclass
from math import isfinite, log1p, sqrt
from statistics import median
from hashlib import sha256
import json

@dataclass(frozen=True)
class Fold:
    train: tuple
    valid: tuple

def validate_partition(fold, allowed, groups):
    t, h = set(fold.train), set(fold.valid)
    assert len(t) == len(fold.train) and len(h) == len(fold.valid)
    assert t and h and not t & h and t | h == set(allowed)
    assert not {groups[i] for i in t} & {groups[i] for i in h}

def twofold_plan(scope, groups):
    """Synthetic deterministic group split, not a proposed real-data split."""
    gs = sorted({groups[i] for i in scope})
    assert len(gs) >= 2
    result = []
    for k in (0, 1):
        h = tuple(i for i in scope if gs.index(groups[i]) % 2 == k)
        t = tuple(i for i in scope if i not in h)
        f = Fold(t, h); validate_partition(f, scope, groups); result.append(f)
    return result

def sentinel_response(train, query, labels):
    """Arithmetic dependency sentinel, deliberately NOT a trained estimator."""
    deps = frozenset(train)
    score = sum((i+1)*labels[i] for i in train) / (100*len(train))
    return {i:(score + (i+1)/1000, deps) for i in query}

def nested_base_contract(outer, groups, labels, allowed):
    validate_partition(outer, allowed, groups)
    zt = {}
    for inner in twofold_plan(outer.train, groups):
        zt.update(sentinel_response(inner.train, inner.valid, labels))
    zh = sentinel_response(outer.train, outer.valid, labels)
    assert set(zt) == set(outer.train)
    for i, (_, deps) in zt.items():
        assert deps <= set(outer.train) and i not in deps
        assert groups[i] not in {groups[j] for j in deps}
    for _, deps in zh.values(): assert deps <= set(outer.train)
    # This sum is a downstream dependency sentinel, not a fitted meta-model.
    meta_sentinel = sum(zt[i][0]+labels[i] for i in outer.train)
    signature = training_key(outer.train, groups, labels, 'synthetic-fixed-v1')
    return zt, zh, meta_sentinel, signature

def training_key(train, groups, labels, recipe):
    rows=[(i,groups[i],labels[i]) for i in sorted(train)]
    return sha256(json.dumps({'recipe':recipe,'rows':rows},sort_keys=True).encode()).hexdigest()

def convex_weight(base, candidate, y):
    assert len(base) == len(candidate) == len(y) and len(y)>0
    assert all(isfinite(v) for v in list(base)+list(candidate)+list(y))
    d=[c-b for b,c in zip(base,candidate)]
    den=sum(a*a for a in d)
    return 0.0 if den==0 else min(1.0,max(0.0,sum(a*(t-b) for a,t,b in zip(d,y,base))/den))

def blend(base,candidate,w):
    assert 0<=w<=1 and len(base)==len(candidate)
    return [(1-w)*b+w*c for b,c in zip(base,candidate)]

class FixedTransform:
    """Arithmetic fixture for declared schema + fit-row-only moments."""
    def __init__(self, schema):
        assert set(schema.values()) <= {'identity','log1p_nonnegative'}
        self.schema=tuple(schema.items())
    def _row(self,row):
        assert tuple(row)==tuple(k for k,_ in self.schema)
        out=[]
        for j,(key,kind) in enumerate(self.schema):
            v=row[key]
            if v is None: v=self.medians[j]
            assert isfinite(v)
            if kind=='log1p_nonnegative':
                if v<0: raise ValueError('Negative value violates fixed nonnegative schema')
                v=log1p(v)
            out.append(v)
        return out
    def fit_moments(self, rows):
        assert rows
        self.medians=[]
        for k,_ in self.schema:
            observed=[r[k] for r in rows if r[k] is not None]
            assert all(isfinite(v) for v in observed)
            self.medians.append(median(observed) if observed else 0.0)
        z=[self._row(r) for r in rows]
        self.means=[sum(r[j] for r in z)/len(z) for j in range(len(self.schema))]
        self.scales=[sqrt(sum((r[j]-m)**2 for r in z)/len(z)) or 1.0 for j,m in enumerate(self.means)]
        return self
    def transform(self,rows):
        return [[(v-m)/s for v,m,s in zip(self._row(r),self.means,self.scales)]+[float(r[k] is None) for k,_ in self.schema] for r in rows]
