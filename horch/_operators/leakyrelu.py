import numpy as np

from .operator import Operator

class LeakyReLU(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x, negative_slope):
    return np.maximum(x, 0) + negative_slope * np.minimum(0, x)

  def backward(self, acc, x, negative_slope):
    acc[x <= 0] *= negative_slope
    return acc
