"""Utilities for notebooks."""

# Libraries
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Class for Gaussian Basis Functions
class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Gaussian Features for 1-D input."""

    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor

    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x-y)/width
        return np.exp(-0.5*np.sum(arg**2, axis))

    def fit(self, X, y=None):
        # create n centers along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self

    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)