import numpy as np

from .operator import Operator

class ELU(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, alpha):
    return np.maximum(x, 0) + np.minimum(0, alpha * (np.exp(x) - 1))

  def backward(self, acc, x, alpha):
    ind = x < 0
    acc[ind] = (acc * alpha * np.exp(x))[ind]
    return acc
