"""Same 218 imputation and BalancedKernel recipe; no flag information in D."""
import numpy as np
from compact_regression import BalancedKernel

class PolicyKernel:
    def __init__(self, blocks, bandwidth, alpha):
        self.blocks, self.bandwidth, self.alpha = blocks, bandwidth, alpha
    def fit(self, X, y):
        assert np.isfinite(X[:, 1:]).all()
        good = np.isfinite(X[:, 0])
        self.impute_ = float(np.median(X[good, 0])) if good.any() else 0.
        self.train_good_ = int(good.sum())
        self.model_ = BalancedKernel(self.blocks, self.bandwidth, self.alpha).fit(self.transform(X), y)
        return self
    def transform(self, X):
        z = X.copy()
        z[~np.isfinite(z[:, 0]), 0] = self.impute_
        assert np.isfinite(z).all()
        return z
    def predict(self, X):
        return self.model_.predict(self.transform(X))
