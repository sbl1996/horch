import numpy as np

from .operator import Operator

class Sqrt(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return np.sqrt(x)

  def backward(self, acc, x):
    res = self.tensor.data.copy()
    return acc / (2 * res)
