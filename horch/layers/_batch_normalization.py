import numpy as np
from ._module import Module
import horch as H

class BatchNorm1d(Module):

  def __init__(self, num_features, eps=1e-05):
    super(BatchNorm1d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.mean = H.Parameter(np.zeros((1, num_features), dtype=np.double))
    self.std = H.Parameter(np.ones((1, num_features), dtype=np.double))

  def forward(self, x):
    m = H.mean(x, axis=0, keepdims=True)
    v = H.var(x, axis=0, keepdims=True)
    x = (x - m) / H.sqrt(v + self.eps)
    z = self.std * x + self.mean
    return z

class BatchNorm2d(Module):

  def __init__(self, num_features, eps=1e-05):
    super(BatchNorm1d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.mean = H.Parameter(np.zeros((1, num_features), dtype=np.double))
    self.std = H.Parameter(np.ones((1, num_features), dtype=np.double))

  def forward(self, x):
    m = H.mean(x, axis=0, keepdims=True)
    v = H.var(x, axis=0, keepdims=True)
    x = (x - m) / H.sqrt(v + self.eps)
    z = self.std * x + self.mean
    return z