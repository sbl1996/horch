import numpy as np

from .operator import Operator

class CrossEntropy(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, logits, target):
    m = logits.shape[0]
    max_val = np.max(logits, axis=1, keepdims=True)
    logits -= max_val
    norm = np.sum(np.exp(logits), axis=1)
    self.norm = norm
    return np.mean(-logits[np.arange(m), target] + np.log(norm))

  def backward(self, acc, logits, target):
    m = logits.shape[0]
    g = np.zeros_like(logits)
    g[np.arange(m), target] = -1
    s = np.exp(logits) / self.norm.reshape(-1, 1)
    g += s
    return acc * g / m