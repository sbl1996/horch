import numpy as np

from .operator import Operator

class CrossEntropy(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, logit, target):
    m = logit.shape[0]
    max_val = np.max(logit, axis=1, keepdims=True)
    logit -= max_val
    norm = np.sum(np.exp(logit), axis=1)
    self.norm = norm
    return np.mean(-logit[np.arange(m), target] + np.log(norm))

  def backward(self, acc, logit, target):
    m = logit.shape[0]
    g = np.zeros_like(logit)
    g[np.arange(m), target] = -1
    s = np.exp(logit) / self.norm.reshape(-1, 1)
    g += s
    return acc * g / m