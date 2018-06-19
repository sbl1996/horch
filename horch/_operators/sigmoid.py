import numpy as np

from .operator import Operator

class Sigmoid(Operator):

  def __init__(self, parents, *args):
    super().__init__(parents, args)

  def forward(self, x):
    return 1 / (1 + np.exp(-x))

  def backward(self, acc, x):
    d = self.tensor.data.copy()
    return (1 - d) * d
