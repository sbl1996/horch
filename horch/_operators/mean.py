import numpy as np

from .operator import Operator

class Mean(Operator):

  def __init__(self, parents, *args):
    super(Mean, self).__init__(parents, args)

  def forward(self, x, axis):
    return np.mean(x, keepdims=True).squeeze()

  def backward(self, acc, x, axis):
    return acc * np.ones(x.shape) / x.size