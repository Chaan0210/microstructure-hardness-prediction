"""Importable saved-model class for the frozen stage-4 kernel comparison."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from grain_measurement import expected_rbf


@dataclass
class DistributionKernelModel:
    location: np.ndarray
    scale: np.ndarray
    train_mean: np.ndarray
    train_var: np.ndarray
    coefficients: np.ndarray
    y_mean: float
    length: float
    alpha: float
    uncertainty_scale: float

    @classmethod
    def fit(cls, mean, var, y, length_multiplier, alpha, uncertainty_scale):
        location, scale = mean.mean(0), mean.std(0)
        scale = np.where(scale > 1e-8, scale, 1.)
        m, v = (mean-location)/scale, var/scale**2*uncertainty_scale
        length = length_multiplier*np.sqrt(mean.shape[1])
        K = expected_rbf(m, v, m, v, length)
        K[np.diag_indices_from(K)] += alpha
        y_mean = float(y.mean())
        coefficients = cho_solve(cho_factor(K, lower=True, check_finite=False), y-y_mean, check_finite=False)
        return cls(location, scale, m, v, coefficients, y_mean, length, alpha, uncertainty_scale)

    def predict(self, mean, var):
        m, v = (mean-self.location)/self.scale, var/self.scale**2*self.uncertainty_scale
        return expected_rbf(m, v, self.train_mean, self.train_var, self.length)@self.coefficients+self.y_mean
